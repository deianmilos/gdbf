#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)<(y)?(y):(x))

#define GDBF 1
#define PGDBF 0
#define RAND_SEED 95732

typedef struct SparseMatrixData
{
  int M;
  int N;
  int *ColumnDegree;
  int *RowDegree;
  int **Mat;
} SparseMatrixData;

typedef struct
{
  int RowBlockCount;
  int ColBlockCount;
  int CirculantSize;
  int **ShiftMatrix;
} BaseMatrixData;

typedef struct
{
  int RowCount;
  int ColCount;
  int Rank;
  int *ColumnPermutation;
  int **SystematicMatrix;
} EncodingTransformData;

#endif
