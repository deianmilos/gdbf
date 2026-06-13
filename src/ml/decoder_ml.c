#include "decoder_ml.h"

#if __has_include("../model/as_model_active_quantized.h")
#include "../model/as_model_active_quantized.h"
#define HAVE_ACTIVE_MODEL 1
#else
#define HAVE_ACTIVE_MODEL 0
#endif

int ApplyModelMask(
  const int8_t *features,
  int featureCount,
  int *outMask,
  int candidateCount)
{
#if HAVE_ACTIVE_MODEL
  int i;
  int predicted[AS_QUANTIZED_OUTPUT_SIZE] = {0};

  if (featureCount != AS_QUANTIZED_INPUT_SIZE) {
    return 0;
  }

  if (candidateCount > AS_QUANTIZED_OUTPUT_SIZE) {
    return 0;
  }

  as_quantized_predict(features, predicted);
  for (i = 0; i < candidateCount; i++) {
    outMask[i] = predicted[i] ? 1 : 0;
  }
  return 1;
#else
  (void)features;
  (void)featureCount;
  (void)outMask;
  (void)candidateCount;
  return 0;
#endif
}

int ApplyModelMaskForCandidates(
  const int8_t *candidateFeatures,
  int perCandidateFeatureCount,
  int candidateCount,
  int anchorCandidateIndex,
  int *outMask)
{
#if HAVE_ACTIVE_MODEL
  int i;
  int directInputSize;
  int compactInputSize;
  int *predicted;

  if (candidateFeatures == NULL || outMask == NULL || perCandidateFeatureCount <= 0 || candidateCount <= 0) {
    return 0;
  }

  directInputSize = perCandidateFeatureCount * candidateCount;
  compactInputSize = perCandidateFeatureCount * (candidateCount - 1);

  for (i = 0; i < candidateCount; i++) {
    outMask[i] = 0;
  }

  predicted = (int *)calloc((size_t)AS_QUANTIZED_OUTPUT_SIZE, sizeof(int));
  if (predicted == NULL) {
    return 0;
  }

  if (AS_QUANTIZED_INPUT_SIZE == directInputSize && AS_QUANTIZED_OUTPUT_SIZE <= candidateCount) {
    as_quantized_predict(candidateFeatures, predicted);
    for (i = 0; i < AS_QUANTIZED_OUTPUT_SIZE; i++) {
      outMask[i] = predicted[i] ? 1 : 0;
    }
    free(predicted);
    return 1;
  }

  if (anchorCandidateIndex >= 0 && anchorCandidateIndex < candidateCount &&
      AS_QUANTIZED_INPUT_SIZE == compactInputSize && AS_QUANTIZED_OUTPUT_SIZE <= (candidateCount - 1)) {
    int8_t *compactFeatures = (int8_t *)calloc((size_t)compactInputSize, sizeof(int8_t));
    int srcCand;
    int dstCand = 0;

    if (compactFeatures == NULL) {
      free(predicted);
      return 0;
    }

    for (srcCand = 0; srcCand < candidateCount; srcCand++) {
      int f;
      int srcBase;
      int dstBase;
      if (srcCand == anchorCandidateIndex) {
        continue;
      }
      srcBase = srcCand * perCandidateFeatureCount;
      dstBase = dstCand * perCandidateFeatureCount;
      for (f = 0; f < perCandidateFeatureCount; f++) {
        compactFeatures[dstBase + f] = candidateFeatures[srcBase + f];
      }
      dstCand++;
    }

    as_quantized_predict(compactFeatures, predicted);

    dstCand = 0;
    for (srcCand = 0; srcCand < candidateCount; srcCand++) {
      if (srcCand == anchorCandidateIndex) {
        outMask[srcCand] = 0;
      } else {
        outMask[srcCand] = (dstCand < AS_QUANTIZED_OUTPUT_SIZE && predicted[dstCand]) ? 1 : 0;
        dstCand++;
      }
    }

    free(compactFeatures);
    free(predicted);
    return 1;
  }

  free(predicted);
  return 0;
#else
  (void)candidateFeatures;
  (void)perCandidateFeatureCount;
  (void)candidateCount;
  (void)anchorCandidateIndex;
  (void)outMask;
  return 0;
#endif
}

int ComputeSyndromeWeightOnly(
  const BaseMatrixData *base,
  const int *decodedBits,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *checkNodeSyndrome)
{
  int layer;
  int block;
  int i;
  int Z = base->CirculantSize;
  int total = 0;

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

    for (i = 0; i < Z; i++) {
      if (checkNodeSyndrome[i]) {
        total++;
      }
    }
  }

  return total;
}
