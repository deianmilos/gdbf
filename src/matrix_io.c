#include "matrix_io.h"

void FreeSparseMatrixData(SparseMatrixData *data)
{
  int m;
  if (data == NULL) {
    return;
  }

  if (data->Mat != NULL) {
    for (m = 0; m < data->M; m++) {
      free(data->Mat[m]);
    }
    free(data->Mat);
  }
  free(data->RowDegree);
  free(data->ColumnDegree);

  data->Mat = NULL;
  data->RowDegree = NULL;
  data->ColumnDegree = NULL;
  data->M = 0;
  data->N = 0;
}

int LoadSparseMatrixData(const char *matrixFile, SparseMatrixData *data)
{
  FILE *f = NULL;
  char fileName[512];
  int m, k;

  if (matrixFile == NULL || data == NULL) {
    fprintf(stderr, "Invalid input to LoadSparseMatrixData.\n");
    return 1;
  }

  memset(data, 0, sizeof(*data));

  snprintf(fileName, sizeof(fileName), "%s_size", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    return 1;
  }

  if (fscanf(f, "%d", &data->M) != 1 || fscanf(f, "%d", &data->N) != 1) {
    fprintf(stderr, "Invalid matrix size file format: %s\n", fileName);
    fclose(f);
    return 1;
  }
  fclose(f);
  f = NULL;

  data->ColumnDegree = (int *)calloc(data->N, sizeof(int));
  data->RowDegree = (int *)calloc(data->M, sizeof(int));
  if (data->ColumnDegree == NULL || data->RowDegree == NULL) {
    fprintf(stderr, "Memory allocation failed for degree arrays.\n");
    goto fail;
  }

  snprintf(fileName, sizeof(fileName), "%s_RowDegree", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    if (fscanf(f, "%d", &data->RowDegree[m]) != 1) {
      fprintf(stderr, "Invalid row degree data in %s at row %d\n", fileName, m);
      fclose(f);
      f = NULL;
      goto fail;
    }
  }
  fclose(f);
  f = NULL;

  data->Mat = (int **)calloc(data->M, sizeof(int *));
  if (data->Mat == NULL) {
    fprintf(stderr, "Memory allocation failed for matrix pointers.\n");
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    data->Mat[m] = (int *)calloc(data->RowDegree[m], sizeof(int));
    if (data->Mat[m] == NULL) {
      fprintf(stderr, "Memory allocation failed for matrix row %d.\n", m);
      goto fail;
    }
  }

  snprintf(fileName, sizeof(fileName), "%s", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    for (k = 0; k < data->RowDegree[m]; k++) {
      if (fscanf(f, "%d", &data->Mat[m][k]) != 1) {
        fprintf(stderr, "Invalid matrix entry in %s at row %d, index %d\n", fileName, m, k);
        fclose(f);
        f = NULL;
        goto fail;
      }
    }
  }
  fclose(f);
  f = NULL;

  for (m = 0; m < data->M; m++) {
    for (k = 0; k < data->RowDegree[m]; k++) {
      if (data->Mat[m][k] < 0 || data->Mat[m][k] >= data->N) {
        fprintf(stderr, "Column index out of range at row %d, index %d: %d\n", m, k, data->Mat[m][k]);
        goto fail;
      }
      data->ColumnDegree[data->Mat[m][k]]++;
    }
  }

  return 0;

fail:
  if (f != NULL) {
    fclose(f);
  }
  FreeSparseMatrixData(data);
  return 1;
}

void FreeBaseMatrixData(BaseMatrixData *base)
{
  int row;
  if (base == NULL) {
    return;
  }

  if (base->ShiftMatrix != NULL) {
    for (row = 0; row < base->RowBlockCount; row++) {
      free(base->ShiftMatrix[row]);
    }
    free(base->ShiftMatrix);
  }

  base->ShiftMatrix = NULL;
  base->RowBlockCount = 0;
  base->ColBlockCount = 0;
  base->CirculantSize = 0;
}

int LoadBaseMatrixData(const char *baseMatrixPrefix, BaseMatrixData *base)
{
  FILE *f = NULL;
  char fileName[512];
  int row, col;

  if (baseMatrixPrefix == NULL || base == NULL) {
    fprintf(stderr, "Invalid input to LoadBaseMatrixData.\n");
    return 1;
  }

  memset(base, 0, sizeof(*base));

  snprintf(fileName, sizeof(fileName), "%s_size", baseMatrixPrefix);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    return 1;
  }
  if (fscanf(f, "%d", &base->RowBlockCount) != 1 ||
      fscanf(f, "%d", &base->ColBlockCount) != 1 ||
      fscanf(f, "%d", &base->CirculantSize) != 1) {
    fprintf(stderr, "Invalid base matrix size file format: %s\n", fileName);
    fclose(f);
    return 1;
  }
  fclose(f);
  f = NULL;

  base->ShiftMatrix = (int **)calloc(base->RowBlockCount, sizeof(int *));
  if (base->ShiftMatrix == NULL) {
    fprintf(stderr, "Memory allocation failed for base matrix row pointers.\n");
    goto fail;
  }
  for (row = 0; row < base->RowBlockCount; row++) {
    base->ShiftMatrix[row] = (int *)calloc(base->ColBlockCount, sizeof(int));
    if (base->ShiftMatrix[row] == NULL) {
      fprintf(stderr, "Memory allocation failed for base matrix row %d.\n", row);
      goto fail;
    }
  }

  snprintf(fileName, sizeof(fileName), "%s_mat", baseMatrixPrefix);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (row = 0; row < base->RowBlockCount; row++) {
    for (col = 0; col < base->ColBlockCount; col++) {
      if (fscanf(f, "%d", &base->ShiftMatrix[row][col]) != 1) {
        fprintf(stderr, "Invalid base matrix entry in %s at row %d, col %d\n", fileName, row, col);
        fclose(f);
        f = NULL;
        goto fail;
      }
    }
  }

  fclose(f);
  return 0;

fail:
  if (f != NULL) {
    fclose(f);
  }
  FreeBaseMatrixData(base);
  return 1;
}
