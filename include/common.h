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

/* AS_ML_MODE: 0 = baseline decoder, 1 = ML-enhanced decoder */
#ifndef AS_ML_MODE
#define AS_ML_MODE 0
#endif

/* AS_TRAIN_MODE: 0 = normal decoding, 1 = collect training rows and force escape */
#ifndef AS_TRAIN_MODE
#define AS_TRAIN_MODE 0
#endif

#define MAX_ABSORBING_SETS 1000
#define ABSORBING_SET_SIZE 6
#define MAX_AS_CHECKS 48   /* max unique check nodes adjacent to a size-6 AS */

/* Physical range of bitEnergy: ±1 channel init + ±1 per row block (12 blocks)
   → max magnitude is 13.  Used to clamp and normalize to [-1, 1]. */
#define MAX_BIT_ENERGY 13

/* Safe upper bound for circulant size (Z=27 for wifin_r_1_2). */
#define MAX_CIRCULANT_SIZE 64

typedef struct {
  int unsatChecks;                   /* b: number of unsatisfied check nodes */
  int nodes[ABSORBING_SET_SIZE];     /* variable node indices (decoder-permuted) */
  int ui[ABSORBING_SET_SIZE];        /* #connections to unsatisfied check nodes (structural) */
  int d_out[ABSORBING_SET_SIZE];     /* #connections to check nodes outside the AS (structural) */
  int num_checks;                    /* number of unique check nodes adjacent to this AS */
  unsigned char check_r[MAX_AS_CHECKS];       /* row block index of each adjacent check */
  unsigned char check_p[MAX_AS_CHECKS];       /* circulant position of each adjacent check */
  unsigned char check_as_mask[MAX_AS_CHECKS]; /* bitmask: which AS nodes (bits 0-5) connect here */
} AbsorbingSetEntry;

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
