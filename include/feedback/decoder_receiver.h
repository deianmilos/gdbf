#ifndef DECODER_RECEIVER_H
#define DECODER_RECEIVER_H

#include "common.h"

void InitDecodedFromReceived(const int *receivedword, int *decodedBits, int codeLength);

int LayerPassAndBitMetrics(
  const BaseMatrixData *base,
  const int *decodedBits,
  const int *receivedword,
  int codeLength,
  int xLength,
  int *bitEnergy,
  int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *unsatCounts,
  int *satCounts,
  int *syndromeWeightOut);

int CountBitErrors(const int *decodedBits, const int *codeword, int codeLength);

#endif
