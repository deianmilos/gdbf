#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "common.h"
#include <stdint.h>

#define FEATURE_FLAG_ENERGY       0x01
#define FEATURE_FLAG_MISMATCH     0x02
#define FEATURE_FLAG_UNSAT        0x04
#define FEATURE_FLAG_FLIP_IMPACT  0x08
#define FEATURE_FLAG_FLIP_COUNT   0x10

#define FEATURE_FLAG_DEFAULT_SET (FEATURE_FLAG_ENERGY | FEATURE_FLAG_MISMATCH | FEATURE_FLAG_UNSAT)

typedef struct {
  int featureFlags;
  int explicitSelection;
  int candidateCount;
} FeatureExtractorConfig;

int GetFeatureDimension(const FeatureExtractorConfig *config);
const char *GetFeatureNameByLocalIndex(const FeatureExtractorConfig *config, int localFeatureIndex);

void BuildCandidateFeatureVector(
  const int *candidateIdx,
  int candidateCount,
  int maxCandidateCount,
  const int *bitEnergy,
  const int *decodedBits,
  const int *receivedword,
  const int *unsatCounts,
  const int *satCounts,
  const int *flipCounts,
  const FeatureExtractorConfig *config,
  int8_t *outFeatures);

#endif
