#include "labeling_strategy.h"

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

static const LabelingStrategy kCorrectiveMask = {
  "corrective_mask",
  CorrectiveMaskLabel
};

const LabelingStrategy *GetLabelingStrategy(LabelingStrategyType type)
{
  (void)type;
  return &kCorrectiveMask;
}
