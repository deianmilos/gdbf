#include "feature_extractor.h"

static int ClampI8(int v)
{
  if (v > 127) return 127;
  if (v < -128) return -128;
  return v;
}

static int EffectiveFeatureMask(const FeatureExtractorConfig *config)
{
  int mask;
  if (config == NULL) {
    return FEATURE_FLAG_DEFAULT_SET;
  }

  mask = config->featureFlags;

  if (config->explicitSelection) {
    return mask;
  }

  /* Backward compatibility for old presets:
     0 => basic, FLIP_IMPACT => basic_plus_impact. */
  if (mask == 0) {
    return FEATURE_FLAG_DEFAULT_SET;
  }
  if (mask == FEATURE_FLAG_FLIP_IMPACT) {
    return FEATURE_FLAG_DEFAULT_SET | FEATURE_FLAG_FLIP_IMPACT;
  }

  return mask;
}

static int BuildFeatureOrder(const FeatureExtractorConfig *config, int *orderOut, int maxOrder)
{
  int mask = EffectiveFeatureMask(config);
  int c = 0;

  if (mask & FEATURE_FLAG_ENERGY) {
    if (c < maxOrder) orderOut[c] = FEATURE_FLAG_ENERGY;
    c++;
  }
  if (mask & FEATURE_FLAG_MISMATCH) {
    if (c < maxOrder) orderOut[c] = FEATURE_FLAG_MISMATCH;
    c++;
  }
  if (mask & FEATURE_FLAG_UNSAT) {
    if (c < maxOrder) orderOut[c] = FEATURE_FLAG_UNSAT;
    c++;
  }
  if (mask & FEATURE_FLAG_FLIP_IMPACT) {
    if (c < maxOrder) orderOut[c] = FEATURE_FLAG_FLIP_IMPACT;
    c++;
  }
  if (mask & FEATURE_FLAG_FLIP_COUNT) {
    if (c < maxOrder) orderOut[c] = FEATURE_FLAG_FLIP_COUNT;
    c++;
  }

  return c;
}

int GetFeatureDimension(const FeatureExtractorConfig *config)
{
  int order[5];
  return BuildFeatureOrder(config, order, 5);
}

const char *GetFeatureNameByLocalIndex(const FeatureExtractorConfig *config, int localFeatureIndex)
{
  int order[5];
  int n = BuildFeatureOrder(config, order, 5);

  if (localFeatureIndex < 0 || localFeatureIndex >= n) {
    return "unknown";
  }

  switch (order[localFeatureIndex]) {
    case FEATURE_FLAG_ENERGY:
      return "energy";
    case FEATURE_FLAG_MISMATCH:
      return "mismatch";
    case FEATURE_FLAG_UNSAT:
      return "unsat";
    case FEATURE_FLAG_FLIP_IMPACT:
      return "flip_impact";
    case FEATURE_FLAG_FLIP_COUNT:
      return "flip_count";
    default:
      return "unknown";
  }
}

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
  int8_t *outFeatures)
{
  int j;
  int k = 0;
  int featureDim;
  int order[5];
  int orderCount;

  if (candidateIdx == NULL || bitEnergy == NULL || decodedBits == NULL ||
      receivedword == NULL || unsatCounts == NULL || satCounts == NULL ||
      outFeatures == NULL || config == NULL) {
    return;
  }

  orderCount = BuildFeatureOrder(config, order, 5);
  featureDim = orderCount;

  for (j = 0; j < maxCandidateCount; j++) {
    int valid = (j < candidateCount);
    int idx = valid ? candidateIdx[j] : 0;
    int flipImpact = 0;
    int oi;

    for (oi = 0; oi < orderCount; oi++) {
      switch (order[oi]) {
        case FEATURE_FLAG_ENERGY:
          outFeatures[k++] = (int8_t)(valid ? ClampI8(bitEnergy[idx]) : 0);
          break;
        case FEATURE_FLAG_MISMATCH:
          outFeatures[k++] = (int8_t)(valid ? (decodedBits[idx] ^ receivedword[idx]) : 0);
          break;
        case FEATURE_FLAG_UNSAT:
          outFeatures[k++] = (int8_t)(valid ? ClampI8(unsatCounts[idx]) : 0);
          break;
        case FEATURE_FLAG_FLIP_IMPACT:
          if (valid) {
            flipImpact = unsatCounts[idx] - satCounts[idx];
          } else {
            flipImpact = 0;
          }
          outFeatures[k++] = (int8_t)ClampI8(flipImpact);
          break;
        case FEATURE_FLAG_FLIP_COUNT:
          outFeatures[k++] = (int8_t)(valid ? ClampI8((flipCounts != NULL) ? flipCounts[idx] : 0) : 0);
          break;
        default:
          break;
      }
    }
  }

  (void)featureDim;
}
