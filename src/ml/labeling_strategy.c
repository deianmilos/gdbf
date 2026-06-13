#include "labeling_strategy.h"
#include "decoder_receiver.h"
#include "decoder_perturb.h"

static int SimulateSelection(
  const BaseMatrixData *base,
  const int *receivedword,
  int codeLength,
  const int *decodedBits,
  const int *candidateIdx,
  int candidateCount,
  const int *mask,
  int rolloutIters)
{
  int Z = base->CirculantSize;
  int xLength = base->ColBlockCount * Z - base->RowBlockCount * Z;
  int *trialDecoded;
  int *bitEnergy;
  int *checkNodeSyndrome;
  int *layerVariableBuffer;
  int *shiftedLayerVariableBuffer;
  int *unsatCounts;
  int *satCounts;
  int syndromeWeight = 0;
  int iter;
  int j;

  trialDecoded = (int *)malloc((size_t)codeLength * sizeof(int));
  bitEnergy = (int *)malloc((size_t)codeLength * sizeof(int));
  checkNodeSyndrome = (int *)malloc((size_t)Z * sizeof(int));
  layerVariableBuffer = (int *)malloc((size_t)Z * sizeof(int));
  shiftedLayerVariableBuffer = (int *)malloc((size_t)Z * sizeof(int));
  unsatCounts = (int *)malloc((size_t)xLength * sizeof(int));
  satCounts = (int *)malloc((size_t)xLength * sizeof(int));

  if (trialDecoded == NULL || bitEnergy == NULL || checkNodeSyndrome == NULL ||
      layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL ||
      unsatCounts == NULL || satCounts == NULL) {
    free(trialDecoded);
    free(bitEnergy);
    free(checkNodeSyndrome);
    free(layerVariableBuffer);
    free(shiftedLayerVariableBuffer);
    free(unsatCounts);
    free(satCounts);
    return 0;
  }

  for (j = 0; j < codeLength; j++) {
    trialDecoded[j] = decodedBits[j];
  }

  for (j = 0; j < candidateCount; j++) {
    if (mask[j]) {
      int idx = candidateIdx[j];
      if (idx >= 0 && idx < codeLength) {
        trialDecoded[idx] ^= 1;
      }
    }
  }

  for (iter = 0; iter < rolloutIters; iter++) {
    int syndromeFlag = LayerPassAndBitMetrics(
      base,
      trialDecoded,
      receivedword,
      codeLength,
      xLength,
      bitEnergy,
      checkNodeSyndrome,
      layerVariableBuffer,
      shiftedLayerVariableBuffer,
      unsatCounts,
      satCounts,
      &syndromeWeight);

    if (!syndromeFlag) {
      free(trialDecoded);
      free(bitEnergy);
      free(checkNodeSyndrome);
      free(layerVariableBuffer);
      free(shiftedLayerVariableBuffer);
      free(unsatCounts);
      free(satCounts);
      return 1;
    }

    FlipAtMaxEnergyTrack(
      trialDecoded,
      bitEnergy,
      xLength,
      DECODER_TYPE_GDBF,
      0.0,
      NULL);
  }

  (void)syndromeWeight;

  free(trialDecoded);
  free(bitEnergy);
  free(checkNodeSyndrome);
  free(layerVariableBuffer);
  free(shiftedLayerVariableBuffer);
  free(unsatCounts);
  free(satCounts);
  return 0;
}

static int PopCountUnsigned(unsigned int v)
{
  int c = 0;
  while (v != 0u) {
    c += (int)(v & 1u);
    v >>= 1u;
  }
  return c;
}

static void GroundTruthLabel(
  const BaseMatrixData *base,
  const int *receivedword,
  const int *codeword,
  int codeLength,
  const int *decodedBits,
  const int *candidateIdx,
  int candidateCount,
  const LabelingConfig *config,
  int *labels)
{
  int j;

  (void)base;
  (void)receivedword;
  (void)codeLength;
  (void)config;

  for (j = 0; j < candidateCount; j++) {
    int idx = candidateIdx[j];
    labels[j] = (decodedBits[idx] != codeword[idx]) ? 1 : 0;
  }
}

static void CorrectiveMaskLabel(
  const BaseMatrixData *base,
  const int *receivedword,
  const int *codeword,
  int codeLength,
  const int *decodedBits,
  const int *candidateIdx,
  int candidateCount,
  const LabelingConfig *config,
  int *labels)
{
  int j;

  (void)base;
  (void)receivedword;
  (void)codeLength;
  (void)config;

  for (j = 0; j < candidateCount; j++) {
    int idx = candidateIdx[j];
    labels[j] = (decodedBits[idx] != codeword[idx]) ? 1 : 0;
  }
}

static void RolloutLabel(
  const BaseMatrixData *base,
  const int *receivedword,
  const int *codeword,
  int codeLength,
  const int *decodedBits,
  const int *candidateIdx,
  int candidateCount,
  const LabelingConfig *config,
  int *labels)
{
  const int candidateStart = 1;
  int minTargetFlipCount = 1;
  int maxTargetFlipCount;
  int *trialMask;
  int searchCount;
  unsigned int maskBits;
  unsigned int maskLimit;
  int j;

  (void)codeword;

  for (j = 0; j < candidateCount; j++) {
    labels[j] = 0;
  }

  if (candidateCount <= candidateStart) {
    return;
  }

  trialMask = (int *)calloc((size_t)candidateCount, sizeof(int));
  if (trialMask == NULL) {
    return;
  }

  searchCount = candidateCount - candidateStart;
  if (searchCount >= (int)(sizeof(unsigned int) * 8)) {
    free(trialMask);
    return;
  }

  maxTargetFlipCount = searchCount;
  if (config != NULL && config->rolloutTargetFlipCount > 0) {
    if (config->rolloutTargetFlipCount > searchCount) {
      free(trialMask);
      return;
    }
    minTargetFlipCount = config->rolloutTargetFlipCount;
    maxTargetFlipCount = config->rolloutTargetFlipCount;
  }

  maskLimit = 1u << searchCount;

  for (j = minTargetFlipCount; j <= maxTargetFlipCount; j++) {
    int targetFlipCount = j;

    for (maskBits = 1u; maskBits < maskLimit; maskBits++) {
      unsigned int bit;

      if (PopCountUnsigned(maskBits) != targetFlipCount) {
        continue;
      }

      for (bit = 0; bit < (unsigned int)candidateCount; bit++) {
        trialMask[bit] = 0;
      }

      for (bit = 0; bit < (unsigned int)searchCount; bit++) {
        if (maskBits & (1u << bit)) {
          trialMask[candidateStart + (int)bit] = 1;
        }
      }

      if (!SimulateSelection(
            base,
            receivedword,
            codeLength,
            decodedBits,
            candidateIdx,
            candidateCount,
            trialMask,
            config->rolloutIters)) {
        continue;
      }

      memcpy(labels, trialMask, (size_t)candidateCount * sizeof(int));
      free(trialMask);
      return;
    }
  }

  free(trialMask);
}

static const LabelingStrategy kGroundTruth = {
  "ground_truth",
  GroundTruthLabel
};

static const LabelingStrategy kRollout = {
  "rollout",
  RolloutLabel
};

static const LabelingStrategy kCorrectiveMask = {
  "corrective_mask",
  CorrectiveMaskLabel
};

const LabelingStrategy *GetLabelingStrategy(LabelingStrategyType type)
{
  switch (type) {
    case LABELING_CORRECTIVE_MASK:
      return &kCorrectiveMask;
    case LABELING_ROLLOUT:
      return &kRollout;
    case LABELING_GROUND_TRUTH:
    default:
      return &kGroundTruth;
  }
}
