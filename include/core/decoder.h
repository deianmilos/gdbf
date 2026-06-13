#ifndef DECODER_H
#define DECODER_H

#include "common.h"

#define X_LENGTH(codeLength, base) ((base)->ColBlockCount * (base)->CirculantSize - (base)->RowBlockCount * (base)->CirculantSize)

int DecodeFrameGdbf(
  const BaseMatrixData *base,
  const int *receivedword,
  const int *codeword,
  int codeLength,
  int maxDecoderIterations,
  int *decodedBits,
  int *bitEnergy,
  int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *isCodeword,
  int *usedIterations,
  int *frameBitErrors);

#endif
