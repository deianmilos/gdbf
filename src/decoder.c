#include "decoder.h"

static void InitializeDecoderBitsFromReceived(const int *receivedword, int *decodedBits, int length)
{
  int i;
  for (i = 0; i < length; i++) {
    decodedBits[i] = receivedword[i];
  }
}

static void InitializeBitEnergyFromDecisions(const int *decodedBits, const int *receivedword, int *bitEnergy, int length)
{
  int n;
  for (n = 0; n < length; n++) {
    bitEnergy[n] = (decodedBits[n] ^ receivedword[n]) ? 1 : -1;
  }
}

static void ResetLayerCheckNodeSyndrome(int *checkNodeSyndrome, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    checkNodeSyndrome[i] = 0;
  }
}

static void ShiftLayerVariableBuffer(const int *inputBuffer, int *shiftedBuffer, int circulantSize, int shiftOffset)
{
  int i;
  int sourceIndex = shiftOffset;

  for (i = 0; i < circulantSize; i++) {
    shiftedBuffer[i] = inputBuffer[sourceIndex];
    sourceIndex++;
    if (sourceIndex == circulantSize) {
      sourceIndex = 0;
    }
  }
}

static void AccumulateCheckNodeSyndrome(int *checkNodeSyndrome, const int *shiftedLayerVariableBuffer, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
  }
}

static void BackProjectSyndromeToLayerBuffer(
  const int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int circulantSize,
  int shiftOffset)
{
  int i;
  int targetIndex = shiftOffset;

  for (i = 0; i < circulantSize; i++) {
    layerVariableBuffer[targetIndex] = checkNodeSyndrome[i];
    targetIndex++;
    if (targetIndex == circulantSize) {
      targetIndex = 0;
    }
  }
}

static void AccumulateLayerContributionToBitEnergy(
  const int *layerVariableBuffer,
  int *bitEnergy,
  int blockStart,
  int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    bitEnergy[blockStart + i] += (layerVariableBuffer[i] == 0) ? -1 : 1;
  }
}

static int LayerHasParityViolation(const int *checkNodeSyndrome, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    if (checkNodeSyndrome[i] != 0) {
      return 1;
    }
  }
  return 0;
}

static int FindMaxBitEnergy(const int *bitEnergy, int length)
{
  int n;
  int maxEnergy = -255;
  for (n = 0; n < length; n++) {
    if (bitEnergy[n] > maxEnergy) {
      maxEnergy = bitEnergy[n];
    }
  }
  return maxEnergy;
}

static void FlipBitsAtMaxEnergy(int *decodedBits, const int *bitEnergy, int length, int maxEnergy)
{
  int n;
  for (n = 0; n < length; n++) {
    if (bitEnergy[n] == maxEnergy) {
      decodedBits[n] = 1 - decodedBits[n];
    }
  }
}

static int CountBitErrors(const int *decodedBits, const int *codeword, int length)
{
  int n;
  int errorCount = 0;
  for (n = 0; n < length; n++) {
    if (decodedBits[n] != codeword[n]) {
      errorCount++;
    }
  }
  return errorCount;
}

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
  int *frameBitErrors)
{
  int decoderIteration;
  int layerIndex;
  int columnBlockIndex;

  if (base == NULL || receivedword == NULL || codeword == NULL ||
      decodedBits == NULL || bitEnergy == NULL || checkNodeSyndrome == NULL ||
      layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL ||
      isCodeword == NULL || usedIterations == NULL || frameBitErrors == NULL) {
    return 1;
  }

  *isCodeword = 0;
  *usedIterations = maxDecoderIterations;
  InitializeDecoderBitsFromReceived(receivedword, decodedBits, codeLength);

  for (decoderIteration = 0; decoderIteration < maxDecoderIterations; decoderIteration++) {
    int syndromeFlag = 0;

    InitializeBitEnergyFromDecisions(decodedBits, receivedword, bitEnergy, codeLength);

    for (layerIndex = 0; layerIndex < base->RowBlockCount; layerIndex++) {
      ResetLayerCheckNodeSyndrome(checkNodeSyndrome, base->CirculantSize);

      for (columnBlockIndex = 0; columnBlockIndex < base->ColBlockCount; columnBlockIndex++) {
        int shiftOffset = base->ShiftMatrix[layerIndex][columnBlockIndex];
        if (shiftOffset != -1) {
          int blockStart = columnBlockIndex * base->CirculantSize;
          int circulantIndex;

          for (circulantIndex = 0; circulantIndex < base->CirculantSize; circulantIndex++) {
            layerVariableBuffer[circulantIndex] = decodedBits[blockStart + circulantIndex];
          }

          ShiftLayerVariableBuffer(
            layerVariableBuffer,
            shiftedLayerVariableBuffer,
            base->CirculantSize,
            shiftOffset);

          AccumulateCheckNodeSyndrome(
            checkNodeSyndrome,
            shiftedLayerVariableBuffer,
            base->CirculantSize);
        }
      }

      for (columnBlockIndex = 0; columnBlockIndex < base->ColBlockCount; columnBlockIndex++) {
        int shiftOffset = base->ShiftMatrix[layerIndex][columnBlockIndex];
        if (shiftOffset != -1) {
          int blockStart = columnBlockIndex * base->CirculantSize;

          BackProjectSyndromeToLayerBuffer(
            checkNodeSyndrome,
            layerVariableBuffer,
            base->CirculantSize,
            shiftOffset);

          AccumulateLayerContributionToBitEnergy(
            layerVariableBuffer,
            bitEnergy,
            blockStart,
            base->CirculantSize);
        }
      }

      if (syndromeFlag == 0 && LayerHasParityViolation(checkNodeSyndrome, base->CirculantSize)) {
        syndromeFlag = 1;
      }
    }

    if (syndromeFlag == 0) {
      *isCodeword = 1;
      *usedIterations = decoderIteration + 1;
      break;
    }

    {
      int maxEnergy = FindMaxBitEnergy(bitEnergy, codeLength);
      FlipBitsAtMaxEnergy(decodedBits, bitEnergy, codeLength, maxEnergy);
    }
  }

  *frameBitErrors = CountBitErrors(decodedBits, codeword, codeLength);
  return 0;
}
