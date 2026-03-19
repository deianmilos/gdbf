#include "encoding.h"

static int **AllocateZeroIntMatrix(int rowCount, int colCount)
{
  int rowIndex;
  int **matrix = (int **)calloc(rowCount, sizeof(int *));
  if (matrix == NULL) {
    return NULL;
  }

  for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    matrix[rowIndex] = (int *)calloc(colCount, sizeof(int));
    if (matrix[rowIndex] == NULL) {
      while (rowIndex-- > 0) {
        free(matrix[rowIndex]);
      }
      free(matrix);
      return NULL;
    }
  }

  return matrix;
}

static void FreeIntMatrix(int **matrix, int rowCount)
{
  int rowIndex;
  if (matrix == NULL) {
    return;
  }

  for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    free(matrix[rowIndex]);
  }
  free(matrix);
}

static int ComputeMrbSystematicForm(int *columnPermutation, int **outputMatrix, int **workingMatrix, int rowCount, int colCount)
{
  int pivotRow, pivotSearchRow, eliminationRow, columnIndex, swapBuffer;
  int currentColumn = 0;
  int deferredColumnCount = 0;
  int dependentRowCount = 0;
  int rank;
  int *deferredColumns = (int *)calloc(colCount, sizeof(int));

  if (deferredColumns == NULL) {
    return 0;
  }

  for (pivotRow = 0; pivotRow < rowCount; pivotRow++)
  {
    if (currentColumn == colCount) {
      dependentRowCount = rowCount - pivotRow;
      break;
    }

    for (pivotSearchRow = pivotRow; pivotSearchRow < rowCount; pivotSearchRow++) {
      if (workingMatrix[pivotSearchRow][currentColumn] != 0) {
        break;
      }
    }

    if (pivotSearchRow < rowCount)
    {
      for (columnIndex = currentColumn; columnIndex < colCount; columnIndex++) {
        swapBuffer = workingMatrix[pivotRow][columnIndex];
        workingMatrix[pivotRow][columnIndex] = workingMatrix[pivotSearchRow][columnIndex];
        workingMatrix[pivotSearchRow][columnIndex] = swapBuffer;
      }

      for (eliminationRow = pivotRow + 1; eliminationRow < rowCount; eliminationRow++) {
        if (workingMatrix[eliminationRow][currentColumn] == 1) {
          for (columnIndex = currentColumn; columnIndex < colCount; columnIndex++) {
            workingMatrix[eliminationRow][columnIndex] ^= workingMatrix[pivotRow][columnIndex];
          }
        }
      }

      columnPermutation[pivotRow] = currentColumn;
    }
    else
    {
      deferredColumns[deferredColumnCount++] = currentColumn;
      pivotRow--;
    }

    currentColumn++;
  }

  rank = rowCount - dependentRowCount;

  for (columnIndex = 0; columnIndex < deferredColumnCount; columnIndex++) {
    columnPermutation[rank + columnIndex] = deferredColumns[columnIndex];
  }

  for (pivotRow = 0; pivotRow < rowCount; pivotRow++) {
    for (columnIndex = 0; columnIndex < colCount; columnIndex++) {
      outputMatrix[pivotRow][columnIndex] = workingMatrix[pivotRow][columnPermutation[columnIndex]];
    }
  }

  for (pivotRow = 0; pivotRow < rank - 1; pivotRow++)
  {
    for (eliminationRow = pivotRow + 1; eliminationRow < rank; eliminationRow++)
    {
      if (outputMatrix[pivotRow][eliminationRow] == 1) {
        for (columnIndex = eliminationRow; columnIndex < colCount; columnIndex++) {
          outputMatrix[pivotRow][columnIndex] ^= outputMatrix[eliminationRow][columnIndex];
        }
      }
    }
  }

  free(deferredColumns);
  return rank;
}

void FreeEncodingTransformData(EncodingTransformData *encoding)
{
  if (encoding == NULL) {
    return;
  }

  FreeIntMatrix(encoding->SystematicMatrix, encoding->RowCount);
  free(encoding->ColumnPermutation);
  encoding->SystematicMatrix = NULL;
  encoding->ColumnPermutation = NULL;
  encoding->RowCount = 0;
  encoding->ColCount = 0;
  encoding->Rank = 0;
}

int BuildEncodingTransformFromSparseMatrix(const SparseMatrixData *matrix, EncodingTransformData *encoding)
{
  int rowIndex, rowEntryIndex, columnIndex;
  int **denseParityCheck = NULL;

  if (matrix == NULL || encoding == NULL) {
    fprintf(stderr, "Invalid input to BuildEncodingTransformFromSparseMatrix.\n");
    return 1;
  }

  memset(encoding, 0, sizeof(*encoding));
  encoding->RowCount = matrix->M;
  encoding->ColCount = matrix->N;
  encoding->ColumnPermutation = (int *)calloc(matrix->N, sizeof(int));
  encoding->SystematicMatrix = AllocateZeroIntMatrix(matrix->M, matrix->N);
  denseParityCheck = AllocateZeroIntMatrix(matrix->M, matrix->N);
  if (encoding->ColumnPermutation == NULL || encoding->SystematicMatrix == NULL || denseParityCheck == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding transform.\n");
    goto fail;
  }

  for (columnIndex = 0; columnIndex < matrix->N; columnIndex++) {
    encoding->ColumnPermutation[columnIndex] = columnIndex;
  }

  for (rowIndex = 0; rowIndex < matrix->M; rowIndex++) {
    for (rowEntryIndex = 0; rowEntryIndex < matrix->RowDegree[rowIndex]; rowEntryIndex++) {
      denseParityCheck[rowIndex][matrix->Mat[rowIndex][rowEntryIndex]] = 1;
    }
  }

  for (columnIndex = 0; columnIndex < matrix->N; columnIndex++) {
    printf("%d ", encoding->ColumnPermutation[columnIndex]);
  }
  printf("\n");

  encoding->Rank = ComputeMrbSystematicForm(
    encoding->ColumnPermutation,
    encoding->SystematicMatrix,
    denseParityCheck,
    matrix->M,
    matrix->N);

  printf("Encoding Rank: %d\n", encoding->Rank);

  FreeIntMatrix(denseParityCheck, matrix->M);
  return 0;

fail:
  FreeIntMatrix(denseParityCheck, matrix->M);
  FreeEncodingTransformData(encoding);
  return 1;
}

void EncodeRandomCodeword(int rank, int N, int **systematicMatrix, const int *columnPermutation, int *workVector, int *codeword)
{
  int k;
  int l;

  for (k = 0; k < rank; k++) {
    workVector[k] = 0;
  }

  for (k = rank; k < N; k++) {
    workVector[k] = (int)floor(((double)(rand()) / RAND_MAX) * 2);
  }

  for (k = rank - 1; k >= 0; k--) {
    for (l = k + 1; l < N; l++) {
      workVector[k] = workVector[k] ^ (systematicMatrix[k][l] * workVector[l]);
    }
  }

  for (k = 0; k < N; k++) {
    codeword[columnPermutation[k]] = workVector[k];
  }
}
