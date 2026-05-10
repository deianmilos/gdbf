#include "decoder.h"
#include <stdio.h>
#include <math.h>

#if AS_ML_MODE
#include "../model/as_model_quantized.h"
#endif

extern FILE *datasetFile;

/* Diagnostic counters (ML mode only) */
long long dbg_stagnation_events = 0;
long long dbg_as_matched        = 0;
long long dbg_ml_fired          = 0;
long long dbg_ml_escaped        = 0;

enum {
  STAGNATION_TRIGGER = 3,
  OSCILLATION_TRIGGER = 5,
  ENERGY_HIST_LEN = 8,
  PREVIOUS_ENERGY_UNSET = -9999
};

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

/* Select the K bits with the lowest bitEnergy at stagnation time.
   These are the most uncertain bits regardless of which trapping/absorbing set is active.
   Works without a catalog and in real (non-simulation) deployment. */
static void SelectWorstBits(const int *bitEnergy, int codeLength,
                             int *worstIdx, int K)
{
  /* Initialize with first K indices */
  for (int k = 0; k < K; k++) worstIdx[k] = k;
  /* Maintain the K HIGHEST-energy indices: these are the bits the decoder
     keeps trying to flip — i.e., the bits stuck in the absorbing/trapping set. */
  for (int k = 0; k < K; k++) {
    for (int i = k + 1; i < K; i++) {
      if (bitEnergy[worstIdx[i]] > bitEnergy[worstIdx[k]]) {
        int tmp = worstIdx[k]; worstIdx[k] = worstIdx[i]; worstIdx[i] = tmp;
      }
    }
  }
  /* Scan remaining bits */
  for (int n = K; n < codeLength; n++) {
    if (bitEnergy[n] > bitEnergy[worstIdx[K-1]]) {
      worstIdx[K-1] = n;
      /* Bubble new entry into sorted position */
      for (int k = K-1; k > 0 && bitEnergy[worstIdx[k]] > bitEnergy[worstIdx[k-1]]; k--) {
        int tmp = worstIdx[k]; worstIdx[k] = worstIdx[k-1]; worstIdx[k-1] = tmp;
      }
    }
  }
}

#if AS_TRAIN_MODE
static void SaveASTrainingData(
    const int *decodedBits,
    const int *receivedword,
    const int *codeword,
    const int *bitEnergy,
    const int *worstIdx)
{
  /* E0..E5: raw bit-energy (integer, clamped to [-MAX_BIT_ENERGY, MAX_BIT_ENERGY]) */
  for (int j = 0; j < ABSORBING_SET_SIZE; j++) {
    int e = bitEnergy[worstIdx[j]];
    if (e >  MAX_BIT_ENERGY) e =  MAX_BIT_ENERGY;
    if (e < -MAX_BIT_ENERGY) e = -MAX_BIT_ENERGY;
    fprintf(datasetFile, "%d,", e);
  }
  /* F0..F5: channel-vs-decoder disagreement {0,1} */
  for (int j = 0; j < ABSORBING_SET_SIZE; j++)
    fprintf(datasetFile, "%d,", decodedBits[worstIdx[j]] ^ receivedword[worstIdx[j]]);
  /* L0..L5: true error labels {0,1} */
  for (int j = 0; j < ABSORBING_SET_SIZE; j++) {
    int label = (decodedBits[worstIdx[j]] != codeword[worstIdx[j]]) ? 1 : 0;
    fprintf(datasetFile, "%d%s", label, (j < ABSORBING_SET_SIZE - 1) ? "," : "\n");
  }
  fflush(datasetFile);
}
#endif

static void ResetEscapeTracking(
  int *stagnationCounter,
  int *oscillationCounter,
  int *energyHistCount,
  int *previousMaxEnergy)
{
  *stagnationCounter = 0;
  *oscillationCounter = 0;
  *energyHistCount = 0;
  *previousMaxEnergy = PREVIOUS_ENERGY_UNSET;
}

static void UpdateOscillationState(
  int maxEnergy,
  int *energyHistory,
  int *energyHistCount,
  int *oscillationCounter,
  int *alreadyEscaped)
{
  int inHistory = 0;
  int h;

  for (h = 0; h < *energyHistCount; h++) {
    if (energyHistory[h] == maxEnergy) {
      inHistory = 1;
      break;
    }
  }

  if (inHistory) {
    (*oscillationCounter)++;
  } else {
    *oscillationCounter = 0;
    *alreadyEscaped = 0;
  }

  if (*energyHistCount < ENERGY_HIST_LEN) {
    energyHistory[*energyHistCount] = maxEnergy;
    (*energyHistCount)++;
  } else {
    for (h = 0; h < ENERGY_HIST_LEN - 1; h++) {
      energyHistory[h] = energyHistory[h + 1];
    }
    energyHistory[ENERGY_HIST_LEN - 1] = maxEnergy;
  }
}

#if AS_ML_MODE
static void BuildWorstBitMlFeatures(
  const int *decodedBits,
  const int *receivedword,
  const int *bitEnergy,
  const int *worstIdx,
  int8_t *features)
{
  int j;
  for (j = 0; j < ABSORBING_SET_SIZE; j++) {
    int bitIndex = worstIdx[j];
    int energy = bitEnergy[bitIndex];

    if (energy > MAX_BIT_ENERGY) energy = MAX_BIT_ENERGY;
    if (energy < -MAX_BIT_ENERGY) energy = -MAX_BIT_ENERGY;

    features[j] = (int8_t)energy;
    features[j + ABSORBING_SET_SIZE] =
      (int8_t)(decodedBits[bitIndex] ^ receivedword[bitIndex]);
  }
}

static void ApplyMlFlipMask(int *decodedBits, const int *worstIdx, const int *flipMask)
{
  int j;
  for (j = 0; j < ABSORBING_SET_SIZE; j++) {
    if (flipMask[j]) {
      decodedBits[worstIdx[j]] = 1 - decodedBits[worstIdx[j]];
    }
  }
}
#elif AS_TRAIN_MODE
static void FlipWorstBits(int *decodedBits, const int *worstIdx)
{
  int j;
  for (j = 0; j < ABSORBING_SET_SIZE; j++) {
    decodedBits[worstIdx[j]] = 1 - decodedBits[worstIdx[j]];
  }
}
#endif

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

  /* Per-frame (non-static) stagnation state */
  int previousMaxEnergy = PREVIOUS_ENERGY_UNSET;
  int stagnationCounter = 0;
  int alreadyEscaped = 0;
  int energyHistory[ENERGY_HIST_LEN];
  int energyHistCount = 0;
  int oscillationCounter = 0;

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

      /* Stagnation detection: unchanged peak energy across iterations. */
      if (maxEnergy == previousMaxEnergy) {
          stagnationCounter++;
      } else {
          stagnationCounter = 0;
          alreadyEscaped = 0;
      }
      previousMaxEnergy = maxEnergy;

      /* Oscillation detection: recurring peak energy values in short history. */
      UpdateOscillationState(
        maxEnergy,
        energyHistory,
        &energyHistCount,
        &oscillationCounter,
        &alreadyEscaped);

      {
      int isStuck =
        (stagnationCounter > STAGNATION_TRIGGER) ||
        (oscillationCounter > OSCILLATION_TRIGGER);

      /* Absorbing-set escape runs once per stuck phase. */
      if (isStuck && syndromeFlag == 1 && !alreadyEscaped) {
          dbg_stagnation_events++;
          {
          int worstIdx[ABSORBING_SET_SIZE];
          SelectWorstBits(bitEnergy, codeLength, worstIdx, ABSORBING_SET_SIZE);
          dbg_as_matched++;
#if AS_ML_MODE
          /* ML-guided escape: infer flips for the selected worst bits. */
          int8_t features[AS_QUANTIZED_INPUT_SIZE];
          int flip_mask[AS_QUANTIZED_OUTPUT_SIZE];

          BuildWorstBitMlFeatures(decodedBits, receivedword, bitEnergy, worstIdx, features);

          dbg_ml_fired++;
          if (as_quantized_predict(features, flip_mask)) {
            ApplyMlFlipMask(decodedBits, worstIdx, flip_mask);
            dbg_ml_escaped++;
            ResetEscapeTracking(
              &stagnationCounter,
              &oscillationCounter,
              &energyHistCount,
              &previousMaxEnergy);
            alreadyEscaped = 1;
            continue;
          }
          alreadyEscaped = 1;
#elif AS_TRAIN_MODE
          /* Training mode: emit one sample, then force a state change. */
          SaveASTrainingData(decodedBits, receivedword, codeword, bitEnergy, worstIdx);
          FlipWorstBits(decodedBits, worstIdx);
          ResetEscapeTracking(
            &stagnationCounter,
            &oscillationCounter,
            &energyHistCount,
            &previousMaxEnergy);
#endif
          }
      }
      }

      /* Default GDBF step when no escape action altered control flow. */
      FlipBitsAtMaxEnergy(decodedBits, bitEnergy, codeLength, maxEnergy);
  }
  }

  *frameBitErrors = CountBitErrors(decodedBits, codeword, codeLength);
  return 0;
}
