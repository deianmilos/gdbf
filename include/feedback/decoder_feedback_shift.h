#ifndef DECODER_FEEDBACK_SHIFT_H
#define DECODER_FEEDBACK_SHIFT_H

#include "common.h"

void RecomputeParityFromHOnly(
  const BaseMatrixData *base,
  const int *decodedBits,
  int codeLength,
  int xLength,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *checkNodeSyndrome,
  int *outParityBits,
  int outParityCount);

int BuildClassicalMismatchAndViolatedLayers(
  const BaseMatrixData *base,
  const int *recomputedParity,
  const int *classicalParity,
  int parityCount,
  int *parityMismatchBits,
  int *violatedLayer);

int ComputeLayerUnsatScore(
  const BaseMatrixData *base,
  const int *unsatCounts,
  int xLength,
  int layer);

int SelectLowestEnergyBlocksForLayer(
  const BaseMatrixData *base,
  int layer,
  const int *bitEnergy,
  int codeLength,
  int *outBlocks,
  int maxBlocks,
  int *outConnectedBlocks);

int SelectConnectedBlocksForLayer(
  const BaseMatrixData *base,
  int layer,
  int codeLength,
  int *outBlocks,
  int maxBlocks,
  int *outConnectedBlocks);

void ApplyLayerShiftWithOffsetForBlocks(
  const BaseMatrixData *base,
  int layer,
  int shiftOffset,
  int *decodedBits,
  int codeLength,
  const int *selectedBlocks,
  int selectedBlockCount);

void ApplyLayerShiftWithOffsetForBlocksExceptMaxEnergy(
  const BaseMatrixData *base,
  int layer,
  int shiftOffset,
  int *decodedBits,
  const int *bitEnergy,
  int maxEnergy,
  int codeLength,
  const int *selectedBlocks,
  int selectedBlockCount);

void ApplyLayerShift(
  const BaseMatrixData *base,
  int layer,
  int *decodedBits,
  int codeLength);

#endif
