#include "common.h"
#include "matrix_io.h"
#include "encoding.h"
#include "channel.h"
#include "decoder.h"
#include "stats.h"
#include <io.h> 

FILE *datasetFile = NULL;

#if AS_ML_MODE
extern long long dbg_stagnation_events;
extern long long dbg_as_matched;
extern long long dbg_ml_fired;
extern long long dbg_ml_escaped;

static void AppendMlOutcomeSummaryCsvPerAlpha(
  const char *csvPath,
  const char *resultFile,
  int nbMonteCarlo,
  int maxDecoderIterations,
  int nbFramesStop,
  float alphaValue,
  float alphaMax,
  float alphaMin,
  float alphaStep,
  long long framesTested,
  long long framesDecodedClean,
  long long framesBaselineOnlyDecoded,
  long long framesMlNeeded,
  long long framesMlEffective,
  long long framesMlNotEffective,
  long long stagnationEvents,
  long long mlCalls,
  long long mlEscapes)
{
  FILE *f;
  int csvExists;
  double effectivenessPct;

  if (csvPath == NULL || resultFile == NULL) {
    return;
  }

  csvExists = (access(csvPath, 0) == 0);
  f = fopen(csvPath, "a");
  if (f == NULL) {
    fprintf(stderr, "WARNING: cannot open ML summary CSV: %s\n", csvPath);
    return;
  }

  if (!csvExists) {
    fprintf(f,
      "result_file,nb_monte_carlo,max_iterations,nbframes_stop,"
      "alpha,alpha_max,alpha_min,alpha_step,"
      "frames_tested,frames_decoded_clean,frames_baseline_only_decoded,"
      "frames_ml_needed,frames_ml_effective,frames_ml_not_effective,"
      "ml_effectiveness_pct,stagnation_events,ml_calls,ml_escapes\n");
  }

  effectivenessPct = (framesMlNeeded > 0)
    ? (100.0 * (double)framesMlEffective / (double)framesMlNeeded)
    : 0.0;

  fprintf(f,
    "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%lld,%lld,%lld\n",
    resultFile,
    nbMonteCarlo,
    maxDecoderIterations,
    nbFramesStop,
    alphaValue,
    alphaMax,
    alphaMin,
    alphaStep,
    framesTested,
    framesDecodedClean,
    framesBaselineOnlyDecoded,
    framesMlNeeded,
    framesMlEffective,
    framesMlNotEffective,
    effectivenessPct,
    stagnationEvents,
    mlCalls,
    mlEscapes);

  fclose(f);
}
#endif

int main(int argc, char *argv[])
{
  SparseMatrixData matrix;
  BaseMatrixData base;
  EncodingTransformData encoding;

  FILE *fout = NULL;
  char resultFileName[512];

  int NbMonteCarlo;
  int maxDecoderIterations;
  int NBframes;
  int nb;

  float alpha;
  float alpha_max;
  float alpha_min;
  float alpha_step;

  int *codeword = NULL;
  int *receivedword = NULL;
  int *workVector = NULL;
  int *decodedBits = NULL;
  int *bitEnergy = NULL;
  int *checkNodeSyndrome = NULL;
  int *layerVariableBuffer = NULL;
  int *shiftedLayerVariableBuffer = NULL;

  long long overallFramesTested = 0;
  long long overallFramesDecodedClean = 0;
#if AS_ML_MODE
  long long overallFramesBaselineOnlyDecoded = 0;
  long long overallFramesMlNeeded = 0;
  long long overallFramesMlEffective = 0;
  long long overallFramesMlNotEffective = 0;
#endif

  if (argc < 7) {
    fprintf(stderr, "Usage: %s <NbMonteCarlo> <NbIter> <MatrixFile> <BaseMatrixPrefix> <ResultFile> <alpha> [other args...]\n", argv[0]);
    return 1;
  }

  NbMonteCarlo = atoi(argv[1]);
  maxDecoderIterations = atoi(argv[2]);
  NBframes = (argc > 7) ? atoi(argv[7]) : 0;
  alpha = (float)atof(argv[6]);
  alpha_max = (argc > 9) ? (float)atof(argv[9]) : alpha;
  alpha_min = (argc > 10) ? (float)atof(argv[10]) : (alpha - 1.0f);
  alpha_step = (argc > 11) ? (float)atof(argv[11]) : 1.0f;

  if (NbMonteCarlo <= 0) {
    fprintf(stderr, "NbMonteCarlo must be > 0\n");
    return 1;
  }
  if (maxDecoderIterations <= 0) {
    fprintf(stderr, "NbIter must be > 0\n");
    return 1;
  }
  if (alpha < 0.0f || alpha > 1.0f) {
    fprintf(stderr, "alpha must be in [0, 1]\n");
    return 1;
  }
  if (alpha_step <= 0.0f) {
    fprintf(stderr, "alpha_step must be > 0\n");
    return 1;
  }

  if (LoadSparseMatrixData(argv[3], &matrix) != 0) {
    return 1;
  }

  if (LoadBaseMatrixData(argv[4], &base) != 0) {
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  if (BuildEncodingTransformFromSparseMatrix(&matrix, &encoding) != 0) {
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  printf("Matrix Loaded\n");
  printf("Graph Built\n");
  printf("Base Matrix Loaded\n");
  printf("Encoding Transform Built\n");
  printf("BaseRows=%d, BaseCols=%d, Circulant=%d\n", base.RowBlockCount, base.ColBlockCount, base.CirculantSize);
  printf("EncodingRank=%d\n", encoding.Rank);
  printf("\n");
  printf("Base Matrix:\n");
  {
    int m;
    int k;
    for (m = 0; m < base.RowBlockCount; m++) {
      for (k = 0; k < base.ColBlockCount; k++) {
        printf("%d  ", base.ShiftMatrix[m][k]);
      }
      printf("\n");
    }
  }

#if GDBF
  printf("-------------------------La-GDBF--------------------------------------------------\n");
#endif

#if PGDBF
  printf("-------------------------La-P-GDBF--------------------------------------------------\n");
#endif

  snprintf(resultFileName, sizeof(resultFileName), "%s.res", argv[5]);
  fout = fopen(resultFileName, "a+");
  if (fout == NULL) {
    fprintf(stderr, "Output file %s error!.. Abort\n", resultFileName);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  PrintStatsHeader(fout);

  workVector = (int *)calloc(matrix.N, sizeof(int));
  codeword = (int *)calloc(matrix.N, sizeof(int));
  receivedword = (int *)calloc(matrix.N, sizeof(int));
  decodedBits = (int *)calloc(matrix.N, sizeof(int));
  bitEnergy = (int *)calloc(matrix.N, sizeof(int));
  checkNodeSyndrome = (int *)calloc(base.CirculantSize, sizeof(int));
  layerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));
  shiftedLayerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));

#if AS_TRAIN_MODE
  {
    int datasetExists = (access("data/dataset.csv", 0) == 0);
    datasetFile = fopen("data/dataset.csv", "a");
    if (!datasetFile) {
        printf("ERROR opening dataset file\n");
        return 1;
    }
    if (!datasetExists)
        fprintf(datasetFile, "E0,E1,E2,E3,E4,E5,F0,F1,F2,F3,F4,F5,L0,L1,L2,L3,L4,L5\n");
  }
#endif

  if (workVector == NULL || codeword == NULL || receivedword == NULL || decodedBits == NULL ||
      bitEnergy == NULL || checkNodeSyndrome == NULL || layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding/decoder buffers.\n");
    free(workVector);
    free(codeword);
    free(receivedword);
    free(decodedBits);
    free(bitEnergy);
    free(checkNodeSyndrome);
    free(layerVariableBuffer);
    free(shiftedLayerVariableBuffer);
    fclose(fout);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  srand(RAND_SEED);

  for (alpha = alpha_max; alpha > alpha_min; alpha -= alpha_step) {
    SimulationStats stats;
#if AS_ML_MODE
    long long alphaFramesTested = 0;
    long long alphaFramesDecodedClean = 0;
    long long alphaFramesBaselineOnlyDecoded = 0;
    long long alphaFramesMlNeeded = 0;
    long long alphaFramesMlEffective = 0;
    long long alphaFramesMlNotEffective = 0;
    long long alphaStagnationBefore = dbg_stagnation_events;
    long long alphaMlCallsBeforeAll = dbg_ml_fired;
    long long alphaMlEscapesBeforeAll = dbg_ml_escaped;
#endif
    ResetSimulationStats(&stats);

    for (nb = 0; nb < NbMonteCarlo; nb++) {
      int isCodeword;
      int usedIterations;
      int frameBitErrors;
    #if AS_ML_MODE
      long long mlCallsBefore = dbg_ml_fired;
      long long mlEscapesBefore = dbg_ml_escaped;
    #endif

      EncodeRandomCodeword(
        encoding.Rank,
        matrix.N,
        encoding.SystematicMatrix,
        encoding.ColumnPermutation,
        workVector,
        codeword);

      AddBscNoise(codeword, receivedword, matrix.N, alpha);

      if (DecodeFrameGdbf(
            &base,
            receivedword,
            codeword,
            matrix.N,
            maxDecoderIterations,
            decodedBits,
            bitEnergy,
            checkNodeSyndrome,
            layerVariableBuffer,
            shiftedLayerVariableBuffer,
            &isCodeword,
            &usedIterations,
            &frameBitErrors) != 0) {
        fprintf(stderr, "Decoder error.\n");
        free(workVector);
        free(codeword);
        free(receivedword);
        free(decodedBits);
        free(bitEnergy);
        free(checkNodeSyndrome);
        free(layerVariableBuffer);
        free(shiftedLayerVariableBuffer);
        fclose(fout);
        FreeEncodingTransformData(&encoding);
        FreeBaseMatrixData(&base);
        FreeSparseMatrixData(&matrix);
        return 1;
      }

      {
        int cleanDecoded = (isCodeword && frameBitErrors == 0);
        overallFramesTested++;
#if AS_ML_MODE
        alphaFramesTested++;
#endif
        if (cleanDecoded) {
          overallFramesDecodedClean++;
#if AS_ML_MODE
          alphaFramesDecodedClean++;
#endif
        }

#if AS_ML_MODE
        {
          long long frameMlCalls = dbg_ml_fired - mlCallsBefore;
          long long frameMlEscapes = dbg_ml_escaped - mlEscapesBefore;

          if (frameMlCalls > 0) {
            overallFramesMlNeeded++;
            alphaFramesMlNeeded++;
            if (cleanDecoded && frameMlEscapes > 0) {
              overallFramesMlEffective++;
              alphaFramesMlEffective++;
            } else {
              overallFramesMlNotEffective++;
              alphaFramesMlNotEffective++;
            }
          } else if (cleanDecoded) {
            overallFramesBaselineOnlyDecoded++;
            alphaFramesBaselineOnlyDecoded++;
          }
        }
#endif
      }

      UpdateSimulationStats(&stats, frameBitErrors, isCodeword, usedIterations, maxDecoderIterations);

      if (stats.nbtestedframes % 1000000 == 0) {
        PrintStatsLine(alpha, &stats, matrix.N, fout);
      }

      if (NBframes > 0 && stats.NbTotalErrors == NBframes) {
        break;
      }
    }

    PrintStatsLine(alpha, &stats, matrix.N, fout);

#if AS_ML_MODE
    AppendMlOutcomeSummaryCsvPerAlpha(
      "results/ml_outcome_summary_per_alpha.csv",
      resultFileName,
      NbMonteCarlo,
      maxDecoderIterations,
      NBframes,
      alpha,
      alpha_max,
      alpha_min,
      alpha_step,
      alphaFramesTested,
      alphaFramesDecodedClean,
      alphaFramesBaselineOnlyDecoded,
      alphaFramesMlNeeded,
      alphaFramesMlEffective,
      alphaFramesMlNotEffective,
      dbg_stagnation_events - alphaStagnationBefore,
      dbg_ml_fired - alphaMlCallsBeforeAll,
      dbg_ml_escaped - alphaMlEscapesBeforeAll);
#endif
  }

  free(workVector);
  free(codeword);
  free(receivedword);
  free(decodedBits);
  free(bitEnergy);
  free(checkNodeSyndrome);
  free(layerVariableBuffer);
  free(shiftedLayerVariableBuffer);

  fclose(fout);
  FreeEncodingTransformData(&encoding);
  FreeBaseMatrixData(&base);
  FreeSparseMatrixData(&matrix);

  if (datasetFile) fclose(datasetFile);

#if AS_ML_MODE
  printf("\n=== ML Escape Diagnostics ===\n");
  printf("  Stagnation events detected : %lld\n", dbg_stagnation_events);
  printf("  ML predict() calls         : %lld\n", dbg_ml_fired);
  printf("  ML escapes applied         : %lld\n", dbg_ml_escaped);

  if (overallFramesTested > 0) {
    printf("\n=== Frame-Level Outcome Summary ===\n");
    printf("  Frames tested                          : %lld\n", overallFramesTested);
    printf("  Frames decoded clean                   : %lld\n", overallFramesDecodedClean);
    printf("  Decoded by baseline GDBF only          : %lld\n", overallFramesBaselineOnlyDecoded);
    printf("  Trap frames where ML was needed        : %lld\n", overallFramesMlNeeded);
    printf("  Trap frames where ML escape was useful : %lld\n", overallFramesMlEffective);
    printf("  Trap frames where ML did not help      : %lld\n", overallFramesMlNotEffective);
    printf("  ML effectiveness on trap frames        : %.2f%%\n",
           (overallFramesMlNeeded > 0)
             ? (100.0 * (double)overallFramesMlEffective / (double)overallFramesMlNeeded)
             : 0.0);
  }

  printf("  CSV summary written to               : results/ml_outcome_summary_per_alpha.csv\n");
#endif

  return 0;
}
