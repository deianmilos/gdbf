#include "decoder_receiver.h"

void InitDecodedFromReceived(const int *receivedword, int *decodedBits, int codeLength)
{
  int i;
  for (i = 0; i < codeLength; i++) {
    decodedBits[i] = receivedword[i];
  }
}

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
  int *syndromeWeightOut)
{
  int layer;
  int block;
  int i;
  int syndromeFlag = 0;
  int Z = base->CirculantSize;
  int syndromeWeight = 0;

  for (i = 0; i < codeLength; i++) {
    bitEnergy[i] = (decodedBits[i] ^ receivedword[i]) ? 1 : -1;
  }

  for (i = 0; i < xLength; i++) {
    unsatCounts[i] = 0;
    satCounts[i] = 0;
  }

  for (layer = 0; layer < base->RowBlockCount; layer++) {
    for (i = 0; i < Z; i++) {
      checkNodeSyndrome[i] = 0;
    }

    for (block = 0; block < base->ColBlockCount; block++) {
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

    for (block = 0; block < base->ColBlockCount; block++) {
      int shift = base->ShiftMatrix[layer][block];
      int blockStart;
      int dst;
      if (shift == -1) {
        continue;
      }

      blockStart = block * Z;
      dst = shift;
      for (i = 0; i < Z; i++) {
        layerVariableBuffer[dst] = checkNodeSyndrome[i];
        dst++;
        if (dst == Z) dst = 0;
      }

      for (i = 0; i < Z; i++) {
        int bitIndex = blockStart + i;
        if (bitIndex < codeLength) {
          bitEnergy[bitIndex] += (layerVariableBuffer[i] == 0) ? -1 : 1;
        }
      }
    }

    for (block = 0; block < base->ColBlockCount; block++) {
      int shift = base->ShiftMatrix[layer][block];
      int blockStart;
      if (shift == -1) {
        continue;
      }

      blockStart = block * Z;
      for (i = 0; i < Z; i++) {
        int bitIndex = blockStart + i;
        int checkIndex;
        if (bitIndex >= xLength) {
          continue;
        }
        checkIndex = (i - shift + Z) % Z;
        if (checkNodeSyndrome[checkIndex]) {
          unsatCounts[bitIndex]++;
        } else {
          satCounts[bitIndex]++;
        }
      }
    }

    for (i = 0; i < Z; i++) {
      if (checkNodeSyndrome[i]) {
        syndromeFlag = 1;
        syndromeWeight++;
      }
    }
  }

  *syndromeWeightOut = syndromeWeight;
  return syndromeFlag;
}

int CountBitErrors(const int *decodedBits, const int *codeword, int codeLength)
{
  int i;
  int errors = 0;
  for (i = 0; i < codeLength; i++) {
    if (decodedBits[i] != codeword[i]) {
      errors++;
    }
  }
  return errors;
}
