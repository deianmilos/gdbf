#ifndef DECODER_ML_H
#define DECODER_ML_H

#include "common.h"
#include <stdint.h>

int ApplyModelMask(
  const int8_t *features,
  int featureCount,
  int *outMask,
  int candidateCount);

int ApplyModelMaskForCandidates(
  const int8_t *candidateFeatures,
  int perCandidateFeatureCount,
  int candidateCount,
  int anchorCandidateIndex,
  int *outMask);

int ComputeSyndromeWeightOnly(
  const BaseMatrixData *base,
  const int *decodedBits,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *checkNodeSyndrome);

#endif
