#include "common.h"
#include "matrix_io.h"
#include "encoding.h"
#include "channel.h"
#include "decoder.h"
#include "stats.h"

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

  srand((unsigned int)(time(NULL) + RAND_SEED));

  for (alpha = alpha_max; alpha > alpha_min; alpha -= alpha_step) {
    SimulationStats stats;
    ResetSimulationStats(&stats);

    for (nb = 0; nb < NbMonteCarlo; nb++) {
      int isCodeword;
      int usedIterations;
      int frameBitErrors;

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

      UpdateSimulationStats(&stats, frameBitErrors, isCodeword, usedIterations, maxDecoderIterations);

      if (stats.nbtestedframes % 1000000 == 0) {
        PrintStatsLine(alpha, &stats, matrix.N, fout);
      }

      if (NBframes > 0 && stats.NbTotalErrors == NBframes) {
        break;
      }
    }

    PrintStatsLine(alpha, &stats, matrix.N, fout);
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

  return 0;
}
