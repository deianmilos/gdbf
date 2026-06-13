#include "decoder_feedback_shift.h"

#include <stdlib.h>

void RecomputeParityFromHOnly(
  const BaseMatrixData *base,
  const int *decodedBits,
  int codeLength,
  int xLength,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *checkNodeSyndrome,
  int *outParityBits,
  int outParityCount)
{
  int layer;
  int block;
  int i;
  int Z = base->CirculantSize;
  int hBlockCount = (Z > 0) ? (xLength / Z) : 0;

  if (outParityBits == NULL || outParityCount <= 0) {
    return;
  }

  if (hBlockCount > base->ColBlockCount) {
    hBlockCount = base->ColBlockCount;
  }

  for (layer = 0; layer < base->RowBlockCount; layer++) {
    int parityOffset = layer * Z;
    if (parityOffset >= outParityCount) {
      break;
    }

    for (i = 0; i < Z; i++) {
      checkNodeSyndrome[i] = 0;
    }

    for (block = 0; block < hBlockCount; block++) {
      int shift = base->ShiftMatrix[layer][block];
      if (shift == -1) {
        continue;
      }

      {
        int blockStart = block * Z;
        int src = shift;
        for (i = 0; i < Z; i++) {
          layerVariableBuffer[i] = decodedBits[blockStart + i];
        }
        for (i = 0; i < Z; i++) {
          shiftedLayerVariableBuffer[i] = layerVariableBuffer[src];
          src++;
          if (src == Z) src = 0;
        }
      }

      for (i = 0; i < Z; i++) {
        checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
      }
    }

    for (i = 0; i < Z; i++) {
      int outIndex = parityOffset + i;
      if (outIndex < outParityCount) {
        outParityBits[outIndex] = checkNodeSyndrome[i] & 1;
      }
    }
  }
}

int BuildClassicalMismatchAndViolatedLayers(
  const BaseMatrixData *base,
  const int *recomputedParity,
  const int *classicalParity,
  int parityCount,
  int *parityMismatchBits,
  int *violatedLayer)
{
  int layer;
  int Z = base->CirculantSize;
  int mismatchCount = 0;

  for (layer = 0; layer < base->RowBlockCount; layer++) {
    int i;
    int layerOffset = layer * Z;
    int violated = 0;

    for (i = 0; i < Z; i++) {
      int idx = layerOffset + i;
      if (idx >= parityCount) {
        break;
      }
      parityMismatchBits[idx] = (recomputedParity[idx] != classicalParity[idx]) ? 1 : 0;
      if (parityMismatchBits[idx]) {
        violated = 1;
        mismatchCount++;
      }
    }

    violatedLayer[layer] = violated;
  }

  return mismatchCount;
}

int ComputeLayerUnsatScore(
  const BaseMatrixData *base,
  const int *unsatCounts,
  int xLength,
  int layer)
{
  int block;
  int i;
  int Z = base->CirculantSize;
  int score = 0;

  if (layer < 0 || layer >= base->RowBlockCount) {
    return 0;
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[layer][block];
    if (shift == -1) {
      continue;
    }

    {
      int blockStart = block * Z;
      for (i = 0; i < Z; i++) {
        int bitIndex = blockStart + i;
        if (bitIndex >= xLength) {
          continue;
        }
        score += unsatCounts[bitIndex];
      }
    }
  }

  return score;
}

int SelectLowestEnergyBlocksForLayer(
  const BaseMatrixData *base,
  int layer,
  const int *bitEnergy,
  int codeLength,
  int *outBlocks,
  int maxBlocks,
  int *outConnectedBlocks)
{
  int block;
  int i;
  int connected = 0;
  int selected;
  int *candidateBlocks;
  int *candidateScore;
  int Z;

  if (base == NULL || bitEnergy == NULL || outBlocks == NULL || maxBlocks <= 0 ||
      layer < 0 || layer >= base->RowBlockCount) {
    if (outConnectedBlocks != NULL) {
      *outConnectedBlocks = 0;
    }
    return 0;
  }

  Z = base->CirculantSize;
  if (Z <= 0) {
    if (outConnectedBlocks != NULL) {
      *outConnectedBlocks = 0;
    }
    return 0;
  }

  candidateBlocks = (int *)calloc((size_t)base->ColBlockCount, sizeof(int));
  candidateScore = (int *)calloc((size_t)base->ColBlockCount, sizeof(int));
  if (candidateBlocks == NULL || candidateScore == NULL) {
    free(candidateBlocks);
    free(candidateScore);
    if (outConnectedBlocks != NULL) {
      *outConnectedBlocks = 0;
    }
    return 0;
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[layer][block];
    int start = block * Z;
    int sum = 0;

    if (shift == -1) {
      continue;
    }
    if (start + Z > codeLength) {
      continue;
    }

    for (i = 0; i < Z; i++) {
      sum += bitEnergy[start + i];
    }

    candidateBlocks[connected] = block;
    candidateScore[connected] = sum;
    connected++;
  }

  if (outConnectedBlocks != NULL) {
    *outConnectedBlocks = connected;
  }

  if (connected == 0) {
    free(candidateBlocks);
    free(candidateScore);
    return 0;
  }

  for (i = 0; i < connected - 1; i++) {
    int j;
    int bestIdx = i;
    for (j = i + 1; j < connected; j++) {
      if (candidateScore[j] < candidateScore[bestIdx]) {
        bestIdx = j;
      }
    }
    if (bestIdx != i) {
      int tmpScore = candidateScore[i];
      int tmpBlock = candidateBlocks[i];
      candidateScore[i] = candidateScore[bestIdx];
      candidateBlocks[i] = candidateBlocks[bestIdx];
      candidateScore[bestIdx] = tmpScore;
      candidateBlocks[bestIdx] = tmpBlock;
    }
  }

  selected = (connected + 1) / 2;
  if (selected < 1) {
    selected = 1;
  }
  if (selected > maxBlocks) {
    selected = maxBlocks;
  }

  for (i = 0; i < selected; i++) {
    outBlocks[i] = candidateBlocks[i];
  }

  free(candidateBlocks);
  free(candidateScore);
  return selected;
}

int SelectConnectedBlocksForLayer(
  const BaseMatrixData *base,
  int layer,
  int codeLength,
  int *outBlocks,
  int maxBlocks,
  int *outConnectedBlocks)
{
  int block;
  int selected = 0;
  int connected = 0;
  int Z;

  if (base == NULL || outBlocks == NULL || maxBlocks <= 0 || layer < 0 || layer >= base->RowBlockCount) {
    if (outConnectedBlocks != NULL) {
      *outConnectedBlocks = 0;
    }
    return 0;
  }

  Z = base->CirculantSize;
  if (Z <= 0) {
    if (outConnectedBlocks != NULL) {
      *outConnectedBlocks = 0;
    }
    return 0;
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[layer][block];
    int start = block * Z;
    if (shift == -1 || start + Z > codeLength) {
      continue;
    }
    if (selected < maxBlocks) {
      outBlocks[selected] = block;
      selected++;
    }
    connected++;
  }

  if (outConnectedBlocks != NULL) {
    *outConnectedBlocks = connected;
  }
  return selected;
}

void ApplyLayerShiftWithOffsetForBlocks(
  const BaseMatrixData *base,
  int layer,
  int shiftOffset,
  int *decodedBits,
  int codeLength,
  const int *selectedBlocks,
  int selectedBlockCount)
{
  int block;
  int i;
  int Z = base->CirculantSize;
  int *tmp;

  if (base == NULL || decodedBits == NULL || layer < 0 || layer >= base->RowBlockCount || Z <= 0) {
    return;
  }

  shiftOffset %= Z;
  if (shiftOffset < 0) {
    shiftOffset += Z;
  }
  if (shiftOffset == 0) {
    return;
  }

  tmp = (int *)calloc((size_t)Z, sizeof(int));
  if (tmp == NULL) {
    return;
  }

  if (selectedBlocks != NULL && selectedBlockCount > 0) {
    for (block = 0; block < selectedBlockCount; block++) {
      int colBlock = selectedBlocks[block];
      int shift = base->ShiftMatrix[layer][colBlock];
      int start = colBlock * Z;
      if (colBlock < 0 || colBlock >= base->ColBlockCount || shift == -1 || start + Z > codeLength) {
        continue;
      }
      for (i = 0; i < Z; i++) {
        tmp[i] = decodedBits[start + i];
      }
      for (i = 0; i < Z; i++) {
        int src = (i - shiftOffset + Z) % Z;
        decodedBits[start + i] = tmp[src];
      }
    }
  } else {
    for (block = 0; block < base->ColBlockCount; block++) {
      int shift = base->ShiftMatrix[layer][block];
      int start = block * Z;
      if (shift == -1 || start + Z > codeLength) {
        continue;
      }
      for (i = 0; i < Z; i++) {
        tmp[i] = decodedBits[start + i];
      }
      for (i = 0; i < Z; i++) {
        int src = (i - shiftOffset + Z) % Z;
        decodedBits[start + i] = tmp[src];
      }
    }
  }

  free(tmp);
}

void ApplyLayerShiftWithOffsetForBlocksExceptMaxEnergy(
  const BaseMatrixData *base,
  int layer,
  int shiftOffset,
  int *decodedBits,
  const int *bitEnergy,
  int maxEnergy,
  int codeLength,
  const int *selectedBlocks,
  int selectedBlockCount)
{
  int block;
  int i;
  int Z = base->CirculantSize;
  int *tmp;

  if (base == NULL || decodedBits == NULL || bitEnergy == NULL || layer < 0 || layer >= base->RowBlockCount || Z <= 0) {
    return;
  }

  shiftOffset %= Z;
  if (shiftOffset < 0) {
    shiftOffset += Z;
  }
  if (shiftOffset == 0) {
    return;
  }

  tmp = (int *)calloc((size_t)Z, sizeof(int));
  if (tmp == NULL) {
    return;
  }

  for (block = 0; block < selectedBlockCount; block++) {
    int colBlock = selectedBlocks[block];
    int shift = base->ShiftMatrix[layer][colBlock];
    int start = colBlock * Z;
    if (colBlock < 0 || colBlock >= base->ColBlockCount || shift == -1 || start + Z > codeLength) {
      continue;
    }

    for (i = 0; i < Z; i++) {
      tmp[i] = decodedBits[start + i];
    }

    for (i = 0; i < Z; i++) {
      int src = (i - shiftOffset + Z) % Z;
      int idx = start + i;
      if (bitEnergy[idx] != maxEnergy) {
        decodedBits[idx] = tmp[src];
      }
    }
  }

  free(tmp);
}

void ApplyLayerShift(
  const BaseMatrixData *base,
  int layer,
  int *decodedBits,
  int codeLength)
{
  ApplyLayerShiftWithOffsetForBlocks(base, layer, 1, decodedBits, codeLength, NULL, 0);
}
