#include "common.h"
#include "matrix_io.h"
#include "encoding.h"
#include "channel.h"
#include "decoder.h"
#include "decoder_framework.h"
#include "stats.h"
#include "app/logging.h"
#include "app/args_and_config.h"
#include <io.h>
#include <direct.h>
#include <unistd.h>

FILE *datasetFile = NULL;

#define AUX_EQ_HIST_MAX 256

/* Utility functions for file I/O and path handling */
static void EnsureDir(const char *path)
{
  char tmp[512];
  char *p;

  strncpy(tmp, path, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = '\0';

  for (p = tmp + 1; *p; p++) {
    if (*p == '/' || *p == '\\') {
      char c = *p;
      *p = '\0';
      _mkdir(tmp);
      *p = c;
    }
  }
  _mkdir(tmp);
}

static const char *DecoderTypeToName(DecoderType type)
{
  switch (type) {
    case DECODER_TYPE_FEEDBACK_SHIFT:
      return "feedback_shift";
    case DECODER_TYPE_ML_FEEDBACK:
      return "ml_feedback";
    case DECODER_TYPE_PGDBF:
      return "pgdbf";
    case DECODER_TYPE_ML:
      return "ml";
    case DECODER_TYPE_GDBF:
    default:
      return "baseline";
  }
}

static const char *DecoderTypeToBanner(DecoderType type)
{
  switch (type) {
    case DECODER_TYPE_PGDBF:
      return "-------------------------La-P-GDBF--------------------------------------------------\n";
    case DECODER_TYPE_ML:
      return "-------------------------La-ML-GDBF-------------------------------------------------\n";
    case DECODER_TYPE_FEEDBACK_SHIFT:
      return "-------------------------La-FEEDBACK-SHIFT-------------------------------------------\n";
    case DECODER_TYPE_ML_FEEDBACK:
      return "-------------------------La-ML-FEEDBACK----------------------------------------------\n";
    case DECODER_TYPE_GDBF:
    default:
      return "-------------------------La-GDBF--------------------------------------------------\n";
  }
}

int main(int argc, char *argv[])
{
  SparseMatrixData matrix;
  BaseMatrixData base;
  EncodingTransformData encoding;

  FILE *fout = NULL;
  char codeName[256];
  char matrixFilePath[512];
  char baseMatrixPrefix[512];
  char resultDir[512];
  char resultFileName[512];
  char mlSummaryCsvPath[512];
  char mlFailureCsvPath[512];
  FILE *mlFailureFile = NULL;
  const char *runType;
  DecoderConfig decoderConfig;
  DecoderRuntimeStats decoderRuntimeStats;
  char decoderConfigPath[512] = "configs/decoder/default.cfg";
  int hasDecoderConfigPath = 0;

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
  int *lastBitEnergyHistory = NULL;
  int *mlProposedIndices = NULL;
  int *checkNodeSyndrome = NULL;
  int *layerVariableBuffer = NULL;
  int *shiftedLayerVariableBuffer = NULL;

  long long overallFramesTested = 0;
  long long overallFramesDecodedClean = 0;
  long long overallFramesBaselineOnlyDecoded = 0;
  long long overallFramesMlNeeded = 0;
  long long overallFramesMlEscaped = 0;
  long long overallFramesMlNoEscape = 0;
  long long overallFramesMlInvokedDecodedClean = 0;
  long long overallFramesWithAppliedEscape = 0;
  long long overallFramesWithAppliedEscapeDecodedClean = 0;
  long long overallEscapeActionsInCleanFrames = 0;
  long long overallFailedFramesMlInvoked = 0;
  long long overallFailedFramesNoMlInvocation = 0;
  long long overallFramesWithMlInvocation = 0;
  long long overallMlCallsPerInvokedFrameSum = 0;
  long long overallMlCallsPerInvokedFrameMin = -1;
  long long overallMlCallsPerInvokedFrameMax = 0;
  long long overallFramesWithSuccessfulMl = 0;
  long long overallCorrectedBitsPerSuccessfulFrameSum = 0;
  long long overallCorrectedBitsPerSuccessfulFrameMin = -1;
  long long overallCorrectedBitsPerSuccessfulFrameMax = 0;

  /* ---- ARGUMENT PARSING ---- */
  if (argc > 1 && strncmp(argv[1], "--", 2) == 0) {
    if (ParseNamedArgs(
          argc,
          argv,
          &NbMonteCarlo,
          &maxDecoderIterations,
          codeName,
          (int)sizeof(codeName),
          &alpha,
          &NBframes,
          &alpha_max,
          &alpha_min,
          &alpha_step,
          decoderConfigPath,
          (int)sizeof(decoderConfigPath),
          &hasDecoderConfigPath) != 0) {
      return 1;
    }
  } else {
    if (argc < 5) {
      PrintUsage(argv[0]);
      return 1;
    }

    NbMonteCarlo = atoi(argv[1]);
    maxDecoderIterations = atoi(argv[2]);
    strncpy(codeName, argv[3], sizeof(codeName) - 1);
    codeName[sizeof(codeName) - 1] = '\0';
    alpha = (float)atof(argv[4]);
    NBframes = (argc > 5) ? atoi(argv[5]) : 0;
    alpha_max = (argc > 6) ? (float)atof(argv[6]) : alpha;
    alpha_min = (argc > 7) ? (float)atof(argv[7]) : (alpha - 1.0f);
    alpha_step = (argc > 8) ? (float)atof(argv[8]) : 1.0f;
  }

  /* ---- CONFIG LOADING ---- */
  DecoderConfigInitDefaults(&decoderConfig);
  if (access(decoderConfigPath, 0) == 0) {
    if (DecoderConfigLoadFromFile(&decoderConfig, decoderConfigPath) != 0) {
      fprintf(stderr, "Failed to parse decoder config file: %s\n", decoderConfigPath);
      return 1;
    }
  } else if (hasDecoderConfigPath) {
    fprintf(stderr, "Decoder config file not found: %s\n", decoderConfigPath);
    return 1;
  }

  if (codeName[0] == '\0') {
    if (LoadCodeNameFromConfig(decoderConfigPath, codeName, (int)sizeof(codeName)) != 0) {
      fprintf(stderr, "Missing code name. Pass --code or set code=<CodeName> in %s\n", decoderConfigPath);
      return 1;
    }
  }

  DecoderConfigApplyEnv(&decoderConfig);
  memset(&decoderRuntimeStats, 0, sizeof(decoderRuntimeStats));

  snprintf(matrixFilePath, sizeof(matrixFilePath), "codes/%s/%s_Dform", codeName, codeName);
  snprintf(baseMatrixPrefix, sizeof(baseMatrixPrefix), "codes/%s/%s_Base", codeName, codeName);

  /* ---- INPUT VALIDATION ---- */
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

  /* ---- MATRIX LOADING ---- */
  if (LoadSparseMatrixData(matrixFilePath, &matrix) != 0) {
    return 1;
  }

  if (LoadBaseMatrixData(baseMatrixPrefix, &base) != 0) {
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
  printf("DecoderConfig=%s\n", decoderConfigPath);
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

  printf("%s", DecoderTypeToBanner(decoderConfig.decoderType));

  /* ---- RESULT DIRECTORY SETUP ---- */
  if (decoderConfig.enableDatasetCollection) {
    runType = "collect";
  } else {
    runType = DecoderTypeToName(decoderConfig.decoderType);
  }
  snprintf(resultDir, sizeof(resultDir), "results/%s/%s", codeName, runType);
  EnsureDir(resultDir);
  snprintf(resultFileName, sizeof(resultFileName), "%s/simulation.res", resultDir);
  snprintf(mlSummaryCsvPath, sizeof(mlSummaryCsvPath), "%s/ml_outcome_summary.csv", resultDir);
  snprintf(mlFailureCsvPath, sizeof(mlFailureCsvPath), "%s/ml_failure_cases.csv", resultDir);

  fout = fopen(resultFileName, "a+");
  if (fout == NULL) {
    fprintf(stderr, "Output file %s error!.. Abort\n", resultFileName);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  PrintStatsHeader(
    fout,
    decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
    base.CirculantSize);

  if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int csvExists = (access(mlFailureCsvPath, 0) == 0);
    int recreateCsv = 0;

    if (csvExists) {
      FILE *hdr = fopen(mlFailureCsvPath, "r");
      if (hdr != NULL) {
        char headerLine[1024];
        if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
          if (strstr(headerLine, "error_vn_indices") == NULL ||
              strstr(headerLine, "error_vn_last5_energies") == NULL ||
              strstr(headerLine, "global_max_last5_energies") == NULL ||
              strstr(headerLine, "global_max_vn_indices_last5") == NULL ||
              strstr(headerLine, "ml_proposed_vn_indices") == NULL ||
              strstr(headerLine, "ml_proposed_error_overlap") == NULL) {
            recreateCsv = 1;
          }
        }
        fclose(hdr);
      }
    }

    mlFailureFile = fopen(mlFailureCsvPath, recreateCsv ? "w" : "a");
    if (mlFailureFile == NULL) {
      fprintf(stderr, "WARNING: cannot open ML failure CSV: %s\n", mlFailureCsvPath);
    } else if (!csvExists || recreateCsv) {
      fprintf(mlFailureFile,
        "alpha,frame_index_alpha,frame_index_global,used_iterations,frame_bit_errors,is_codeword,ml_invoked,ml_calls_in_frame,error_vn_indices,error_vn_last5_energies,global_max_last5_energies,global_max_vn_indices_last5,ml_proposed_vn_indices,ml_proposed_error_overlap\n");
    }
  }

  /* ---- MEMORY ALLOCATION ---- */
  workVector = (int *)calloc(matrix.N, sizeof(int));
  codeword = (int *)calloc(matrix.N, sizeof(int));
  receivedword = (int *)calloc(matrix.N, sizeof(int));
  decodedBits = (int *)calloc(matrix.N, sizeof(int));
  bitEnergy = (int *)calloc(matrix.N, sizeof(int));
  lastBitEnergyHistory = (int *)calloc((size_t)matrix.N * 5u, sizeof(int));
  mlProposedIndices = (int *)calloc(matrix.N, sizeof(int));
  checkNodeSyndrome = (int *)calloc(base.CirculantSize, sizeof(int));
  layerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));
  shiftedLayerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));

  if (decoderConfig.enableDatasetCollection) {
  {
    char datasetDir[512];
    char datasetPath[512];
    int datasetExists;
    int recreateDataset = 0;
    int featureDim = DecoderFeatureDimension(&decoderConfig);
    int featureCount = decoderConfig.candidateCount * featureDim;
    int labelCount = decoderConfig.candidateCount;
    snprintf(datasetDir, sizeof(datasetDir), "datasets/%s", codeName);
    EnsureDir(datasetDir);
    snprintf(datasetPath, sizeof(datasetPath), "%s/dataset.csv", datasetDir);
    datasetExists = (access(datasetPath, 0) == 0);

    if (datasetExists) {
      FILE *headerFile = fopen(datasetPath, "r");
      if (headerFile) {
        char headerLine[8192];
        if (fgets(headerLine, sizeof(headerLine), headerFile)) {
          int colCount = 1;
          int ci;
          int expectedCols = featureCount + labelCount;
          for (ci = 0; headerLine[ci] != '\0'; ci++) {
            if (headerLine[ci] == ',') {
              colCount++;
            }
          }
          if (colCount != expectedCols || strncmp(headerLine, "C0_", 3) != 0) {
            recreateDataset = 1;
          }
        }
        fclose(headerFile);
      }
    }

    datasetFile = fopen(datasetPath, recreateDataset ? "w" : "a");
    if (!datasetFile) {
        printf("ERROR opening dataset file\n");
        return 1;
    }
    if (!datasetExists || recreateDataset)
    {
      int fi;
      int li;
      for (fi = 0; fi < decoderConfig.candidateCount; fi++) {
        int fj;
        for (fj = 0; fj < featureDim; fj++) {
          const char *fname = DecoderFeatureName(&decoderConfig, fj);
          fprintf(datasetFile, "C%d_%s,", fi, fname);
        }
      }
      for (li = 0; li < labelCount; li++) {
        fprintf(datasetFile, "Y%d%s", li, (li < labelCount - 1) ? "," : "\n");
      }
    }
  }
  }

  if (workVector == NULL || codeword == NULL || receivedword == NULL || decodedBits == NULL ||
      lastBitEnergyHistory == NULL ||
      mlProposedIndices == NULL ||
      bitEnergy == NULL || checkNodeSyndrome == NULL || layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding/decoder buffers.\n");
    free(workVector);
    free(codeword);
    free(receivedword);
    free(decodedBits);
    free(lastBitEnergyHistory);
    free(mlProposedIndices);
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

  /* ---- MAIN SIMULATION LOOP ---- */
  for (alpha = alpha_max; alpha > alpha_min; alpha -= alpha_step) {
    SimulationStats stats;
    long long alphaFeedbackSuccessAuxEqHist[AUX_EQ_HIST_MAX];
    long long alphaFeedbackSuccessAuxEqOverflow = 0;
    long long alphaNoFeedbackSuccess = 0;
    long long alphaFeedbackSuccess = 0;
    long long alphaNotManagedDecode = 0;
    long long alphaFramesTested = 0;
    long long alphaFramesDecodedClean = 0;
    long long alphaFramesBaselineOnlyDecoded = 0;
    long long alphaFramesMlNeeded = 0;
    long long alphaFramesMlEscaped = 0;
    long long alphaFramesMlNoEscape = 0;
    long long alphaFramesMlInvokedDecodedClean = 0;
    long long alphaFailedFramesMlInvoked = 0;
    long long alphaFailedFramesNoMlInvocation = 0;
    long long alphaStagnationBefore = decoderRuntimeStats.stagnationEvents;
    long long alphaMlCallsBeforeAll = decoderRuntimeStats.modelInferenceCalls;
    long long alphaMlEscapesBeforeAll = decoderRuntimeStats.mlEscapes;
    memset(alphaFeedbackSuccessAuxEqHist, 0, sizeof(alphaFeedbackSuccessAuxEqHist));
    ResetSimulationStats(&stats);

    for (nb = 0; nb < NbMonteCarlo; nb++) {
      int isCodeword;
      int cleanDecoded;
      int usedIterations;
      int frameBitErrors;
      int addedAuxEquations = 0;
      int shiftMatrixGenerations = 0;
      int maxEnergyBitsBeforeFeedbackMin = 0;
      int maxEnergyBitsBeforeFeedbackMax = 0;
      long long maxEnergyBitsBeforeFeedbackSum = 0;
      int maxEnergyBitsBeforeFeedbackCount = 0;
      int unsuccessfulRoundsToSyndrome0 = -1;
      int lastBitEnergyHistoryCount = 0;
      int mlProposedCount = 0;
      long long mlCallsBefore = decoderRuntimeStats.modelInferenceCalls;
      long long mlEscapesBefore = decoderRuntimeStats.mlEscapes;
      long long mlCorrectedBitsBefore = decoderRuntimeStats.mlCorrectedBits;

      if (decoderConfig.quantumOnlySyndrome) {
        EncodeRandomCodeword(
          encoding.Rank,
          matrix.N,
          encoding.SystematicMatrix,
          encoding.ColumnPermutation,
          workVector,
          codeword);

        AddBscNoise(codeword, receivedword, matrix.N, alpha);
      } else {
        int *x = workVector;
        for (int i = 0; i < matrix.N - matrix.M; i++) {
          x[i] = rand() & 1;
        }
        EncodeSplitCodeword(&matrix, x, codeword);

        AddBscNoiseSplit(codeword, receivedword, matrix.N - matrix.M, matrix.M, alpha);
      }

      if (DecodeFrameWithConfig(
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
            &frameBitErrors,
            &addedAuxEquations,
            &shiftMatrixGenerations,
            &maxEnergyBitsBeforeFeedbackMin,
            &maxEnergyBitsBeforeFeedbackMax,
            &maxEnergyBitsBeforeFeedbackSum,
            &maxEnergyBitsBeforeFeedbackCount,
            &unsuccessfulRoundsToSyndrome0,
            lastBitEnergyHistory,
            &lastBitEnergyHistoryCount,
            mlProposedIndices,
            &mlProposedCount,
            matrix.N,
            &decoderConfig,
            &decoderRuntimeStats,
            datasetFile,
            nb,
            alpha) != 0) {
        fprintf(stderr, "Decoder error.\n");
        free(workVector);
        free(codeword);
        free(receivedword);
        free(decodedBits);
        free(lastBitEnergyHistory);
        free(mlProposedIndices);
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
        cleanDecoded = (isCodeword && frameBitErrors == 0);
        overallFramesTested++;
        alphaFramesTested++;
        if (cleanDecoded) {
          overallFramesDecodedClean++;
          alphaFramesDecodedClean++;
        }

        long long frameMlCalls = decoderRuntimeStats.modelInferenceCalls - mlCallsBefore;
        long long frameMlEscapes = decoderRuntimeStats.mlEscapes - mlEscapesBefore;
        long long frameMlCorrectedBits = decoderRuntimeStats.mlCorrectedBits - mlCorrectedBitsBefore;

        {
          if (frameMlEscapes > 0) {
            overallFramesWithAppliedEscape++;
            if (cleanDecoded) {
              overallFramesWithAppliedEscapeDecodedClean++;
              overallEscapeActionsInCleanFrames += frameMlEscapes;
            }
          }

          if (frameMlCalls > 0) {
            overallFramesMlNeeded++;
            alphaFramesMlNeeded++;
            overallFramesWithMlInvocation++;
            overallMlCallsPerInvokedFrameSum += frameMlCalls;
            if (overallMlCallsPerInvokedFrameMin < 0 || frameMlCalls < overallMlCallsPerInvokedFrameMin) {
              overallMlCallsPerInvokedFrameMin = frameMlCalls;
            }
            if (frameMlCalls > overallMlCallsPerInvokedFrameMax) {
              overallMlCallsPerInvokedFrameMax = frameMlCalls;
            }

            if (frameMlEscapes > 0) {
              overallFramesMlEscaped++;
              alphaFramesMlEscaped++;
              overallFramesWithSuccessfulMl++;
              overallCorrectedBitsPerSuccessfulFrameSum += frameMlCorrectedBits;
              if (overallCorrectedBitsPerSuccessfulFrameMin < 0 || frameMlCorrectedBits < overallCorrectedBitsPerSuccessfulFrameMin) {
                overallCorrectedBitsPerSuccessfulFrameMin = frameMlCorrectedBits;
              }
              if (frameMlCorrectedBits > overallCorrectedBitsPerSuccessfulFrameMax) {
                overallCorrectedBitsPerSuccessfulFrameMax = frameMlCorrectedBits;
              }
            } else {
              overallFramesMlNoEscape++;
              alphaFramesMlNoEscape++;
            }

            if (cleanDecoded) {
              overallFramesMlInvokedDecodedClean++;
              alphaFramesMlInvokedDecodedClean++;
            } else {
              overallFailedFramesMlInvoked++;
              alphaFailedFramesMlInvoked++;
            }
          } else {
            if (cleanDecoded) {
              overallFramesBaselineOnlyDecoded++;
              alphaFramesBaselineOnlyDecoded++;
            } else {
              overallFailedFramesNoMlInvocation++;
              alphaFailedFramesNoMlInvocation++;
            }
          }

          if (!cleanDecoded && mlFailureFile != NULL) {
            char errorVnIndices[16384];
            char errorVnEnergies[65536];
            char globalMaxEnergies[1024];
            char globalMaxVnIndices[65536];
            char mlProposedVnIndices[16384];
            char mlProposedErrorOverlap[16384];
            BuildErrorVnIndexList(decodedBits, codeword, matrix.N, errorVnIndices, (int)sizeof(errorVnIndices));
            BuildErrorVnEnergyHistoryList(
              decodedBits,
              codeword,
              matrix.N,
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              errorVnEnergies,
              (int)sizeof(errorVnEnergies));
            BuildGlobalMaxEnergyHistoryList(
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              matrix.N - matrix.M,
              globalMaxEnergies,
              (int)sizeof(globalMaxEnergies));
            BuildGlobalMaxVnIndexHistoryList(
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              matrix.N - matrix.M,
              globalMaxVnIndices,
              (int)sizeof(globalMaxVnIndices));
            BuildIndexListFromBuffer(
              mlProposedIndices,
              mlProposedCount,
              mlProposedVnIndices,
              (int)sizeof(mlProposedVnIndices));
            BuildMlProposedErrorOverlapList(
              mlProposedIndices,
              mlProposedCount,
              decodedBits,
              codeword,
              matrix.N,
              mlProposedErrorOverlap,
              (int)sizeof(mlProposedErrorOverlap));
            fprintf(
              mlFailureFile,
              "%.6f,%d,%lld,%d,%d,%d,%d,%lld,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
              alpha,
              nb + 1,
              overallFramesTested,
              usedIterations,
              frameBitErrors,
              isCodeword,
              (frameMlCalls > 0) ? 1 : 0,
              frameMlCalls,
              errorVnIndices,
              errorVnEnergies,
              globalMaxEnergies,
              globalMaxVnIndices,
              mlProposedVnIndices,
              mlProposedErrorOverlap);
          }
        }

        if (addedAuxEquations < 0) {
          addedAuxEquations = 0;
        }

        if (cleanDecoded) {
          if (unsuccessfulRoundsToSyndrome0 >= 0) {
            alphaFeedbackSuccess++;
            if (shiftMatrixGenerations < AUX_EQ_HIST_MAX) {
              alphaFeedbackSuccessAuxEqHist[shiftMatrixGenerations]++;
            } else {
              alphaFeedbackSuccessAuxEqOverflow++;
            }
          } else {
            alphaNoFeedbackSuccess++;
          }
        } else {
          alphaNotManagedDecode++;
        }

        UpdateSimulationStats(
          &stats,
          frameBitErrors,
          isCodeword,
          usedIterations,
          maxDecoderIterations,
          addedAuxEquations,
          unsuccessfulRoundsToSyndrome0,
          maxEnergyBitsBeforeFeedbackMin,
          maxEnergyBitsBeforeFeedbackMax,
          maxEnergyBitsBeforeFeedbackSum,
          maxEnergyBitsBeforeFeedbackCount,
          (int)frameMlCalls);
      }

      if (stats.nbtestedframes % 1000000 == 0) {
        PrintStatsLine(
          alpha,
          &stats,
          matrix.N,
          fout,
          decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
          base.CirculantSize);
      }

      if (NBframes > 0 && stats.NbTotalErrors == NBframes) {
        break;
      }
    }

    if ((stats.nbtestedframes % 1000000) != 0) {
      PrintStatsLine(
        alpha,
        &stats,
        matrix.N,
        fout,
        decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
        base.CirculantSize);
    }

    if (decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
      PrintFeedbackShiftDistributionSummary(
        fout,
        alpha,
        alphaFramesTested,
        alphaNoFeedbackSuccess,
        alphaFeedbackSuccess,
        alphaNotManagedDecode,
        alphaFeedbackSuccessAuxEqHist,
        alphaFeedbackSuccessAuxEqOverflow);
    }
    
    if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
      AppendMlOutcomeSummaryCsvPerAlpha(
        mlSummaryCsvPath,
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
        alphaFramesMlEscaped,
        alphaFramesMlNoEscape,
        alphaFramesMlInvokedDecodedClean,
        decoderRuntimeStats.stagnationEvents - alphaStagnationBefore,
        decoderRuntimeStats.modelInferenceCalls - alphaMlCallsBeforeAll,
        decoderRuntimeStats.mlEscapes - alphaMlEscapesBeforeAll);

      (void)alphaFailedFramesMlInvoked;
      (void)alphaFailedFramesNoMlInvocation;
    }
  }

  /* ---- CLEANUP ---- */
  free(workVector);
  free(codeword);
  free(receivedword);
  free(decodedBits);
  free(lastBitEnergyHistory);
  free(mlProposedIndices);
  free(bitEnergy);
  free(checkNodeSyndrome);
  free(layerVariableBuffer);
  free(shiftedLayerVariableBuffer);

  fclose(fout);
  FreeEncodingTransformData(&encoding);
  FreeBaseMatrixData(&base);
  FreeSparseMatrixData(&matrix);

  if (datasetFile) fclose(datasetFile);
  if (mlFailureFile) fclose(mlFailureFile);

  /* ---- FINAL DIAGNOSTICS ---- */
  if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
    printf("\n=== ML Escape Diagnostics ===\n");
    printf("  Stagnation events detected : %lld\n", decoderRuntimeStats.stagnationEvents);
    printf("  ML counterfactual skips    : %lld\n", decoderRuntimeStats.mlCounterfactualSkips);
    printf("  ML predict() calls         : %lld\n", decoderRuntimeStats.modelInferenceCalls);
    printf("  ML escapes applied         : %lld\n", decoderRuntimeStats.mlEscapes);
    printf("  Bits corrected by ML       : %lld\n", decoderRuntimeStats.mlCorrectedBits);

    if (overallFramesTested > 0) {
      printf("\n=== Frame-Level Outcome Summary ===\n");
      printf("  Frames tested                          : %lld\n", overallFramesTested);
      printf("  Frames decoded clean                   : %lld\n", overallFramesDecodedClean);
      printf("  Decoded by baseline GDBF only          : %lld\n", overallFramesBaselineOnlyDecoded);
      printf("  Stuck-triggered frames (ML invoked)    : %lld\n", overallFramesMlNeeded);
      printf("  ML escape success (applied escapes)    : %.2f%%\n",
             (decoderRuntimeStats.mlEscapes > 0)
               ? (100.0 * (double)overallEscapeActionsInCleanFrames / (double)decoderRuntimeStats.mlEscapes)
               : 0.0);

      printf("\n=== Failure Breakdown ===\n");
      printf("  Failed frames with ML invoked          : %lld\n", overallFailedFramesMlInvoked);
      printf("  Failed frames with no ML invocation    : %lld\n", overallFailedFramesNoMlInvocation);

      printf("\n=== ML Calls Per Invoked Frame ===\n");
      if (overallFramesWithMlInvocation > 0) {
        printf("  Min ML calls/frame                     : %lld\n", overallMlCallsPerInvokedFrameMin);
        printf("  Max ML calls/frame                     : %lld\n", overallMlCallsPerInvokedFrameMax);
        printf("  Avg ML calls/frame                     : %.2f\n",
               (double)overallMlCallsPerInvokedFrameSum / (double)overallFramesWithMlInvocation);
      } else {
        printf("  Min ML calls/frame                     : n/a\n");
        printf("  Max ML calls/frame                     : n/a\n");
        printf("  Avg ML calls/frame                     : n/a\n");
      }

      printf("\n=== Corrected Bits Per Successful ML Frame ===\n");
      if (overallFramesWithSuccessfulMl > 0) {
        printf("  Min corrected bits/frame               : %lld\n", overallCorrectedBitsPerSuccessfulFrameMin);
        printf("  Max corrected bits/frame               : %lld\n", overallCorrectedBitsPerSuccessfulFrameMax);
        printf("  Avg corrected bits/frame               : %.2f\n",
               (double)overallCorrectedBitsPerSuccessfulFrameSum / (double)overallFramesWithSuccessfulMl);
      } else {
        printf("  Min corrected bits/frame               : n/a\n");
        printf("  Max corrected bits/frame               : n/a\n");
        printf("  Avg corrected bits/frame               : n/a\n");
      }
    }

    printf("  CSV summary written to               : %s\n", mlSummaryCsvPath);
    if (mlFailureFile != NULL) {
      printf("  Failure cases CSV written to         : %s\n", mlFailureCsvPath);
    }
  }

  if (decoderConfig.enableDatasetCollection) {
    printf("\n=== Dataset Collection ===\n");
    printf("  Rows written                           : %lld\n", decoderRuntimeStats.datasetRows);
  }

  return 0;
}
