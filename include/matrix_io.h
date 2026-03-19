#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include "common.h"

void FreeSparseMatrixData(SparseMatrixData *data);
int LoadSparseMatrixData(const char *matrixFile, SparseMatrixData *data);

void FreeBaseMatrixData(BaseMatrixData *base);
int LoadBaseMatrixData(const char *baseMatrixPrefix, BaseMatrixData *base);

#endif
