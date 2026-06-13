#ifndef LABELING_STRATEGY_H
#define LABELING_STRATEGY_H

#include "common.h"

typedef enum {
  LABELING_GROUND_TRUTH = 0,
  LABELING_ROLLOUT = 1,
  LABELING_CORRECTIVE_MASK = 2
} LabelingStrategyType;

typedef struct {
  LabelingStrategyType type;
  int rolloutIters;
  int rolloutTargetFlipCount;
  int convergenceBonus;
} LabelingConfig;

typedef struct {
  const char *name;
  void (*label)(
    const BaseMatrixData *base,
    const int *receivedword,
    const int *codeword,
    int codeLength,
    const int *decodedBits,
    const int *candidateIdx,
    int candidateCount,
    const LabelingConfig *config,
    int *labels);
} LabelingStrategy;

const LabelingStrategy *GetLabelingStrategy(LabelingStrategyType type);

#endif
