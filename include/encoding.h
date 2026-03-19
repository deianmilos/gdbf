#ifndef ENCODING_H
#define ENCODING_H

#include "common.h"

void FreeEncodingTransformData(EncodingTransformData *encoding);
int BuildEncodingTransformFromSparseMatrix(const SparseMatrixData *matrix, EncodingTransformData *encoding);

void EncodeRandomCodeword(
  int rank,
  int N,
  int **systematicMatrix,
  const int *columnPermutation,
  int *workVector,
  int *codeword);

#endif
