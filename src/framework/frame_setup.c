/*
 * frame_setup.c — FrameState buffer allocation and teardown.
 *
 * FrameStateAlloc() allocates every dynamic buffer that DecodeFrameWithConfig
 * needs for one frame.  Caller must fill the const/config fields of FrameState
 * (base, xLength, codeLength, config, selectionConfig, featureConfig) before
 * calling this function.
 *
 * FrameStateFree() releases everything; safe to call after a partial alloc.
 */

#include "frame_state.h"
#include "feature_extractor.h"
#include "candidate_selection.h"
#include "labeling_strategy.h"
#include "stagnation_detection.h"
#include "decoder_framework.h"

#include <stdlib.h>
#include <string.h>

int FrameStateAlloc(FrameState *fs)
{
  int rowInit;

  if (fs == NULL || fs->base == NULL || fs->config == NULL) {
    return 1;
  }

  /* Stagnation state must be initialised before the allocation check */
  if (StagnationStateInit(&fs->stagnationState,
                          fs->config->stagnation.energyHistoryLen) != 0) {
    return 1;
  }

  /* Pre-compute feature/selection dimensions */
  fs->featureCount = GetFeatureDimension(&fs->featureConfig);
  fs->featureDim   = fs->featureCount * fs->selectionConfig.candidateCount;

  /* --- Decode working buffers --- */
  fs->unsatCounts  = (int *)calloc((size_t)fs->xLength, sizeof(int));
  fs->satCounts    = (int *)calloc((size_t)fs->xLength, sizeof(int));
  fs->flipCounts   = (int *)calloc((size_t)fs->xLength, sizeof(int));
  fs->candidateIdx = (int *)calloc((size_t)fs->selectionConfig.candidateCount, sizeof(int));
  fs->labels       = (int *)calloc((size_t)fs->selectionConfig.candidateCount, sizeof(int));
  fs->recentEnergyHistory =
    (int *)calloc((size_t)(FRAME_ENERGY_HISTORY_DEPTH * fs->codeLength), sizeof(int));

  /* --- Scratch buffers --- */
  fs->scratchDecodedBits              = (int *)calloc((size_t)fs->codeLength, sizeof(int));
  fs->scratchBitEnergy                = (int *)calloc((size_t)fs->codeLength, sizeof(int));
  fs->scratchCheckNodeSyndrome        = (int *)calloc((size_t)fs->base->CirculantSize, sizeof(int));
  fs->scratchLayerVariableBuffer      = (int *)calloc((size_t)fs->base->CirculantSize, sizeof(int));
  fs->scratchShiftedLayerVariableBuffer =
    (int *)calloc((size_t)fs->base->CirculantSize, sizeof(int));
  fs->scratchUnsatCounts = (int *)calloc((size_t)fs->xLength, sizeof(int));
  fs->scratchSatCounts   = (int *)calloc((size_t)fs->xLength, sizeof(int));

  /* --- Parity buffers --- */
  {
    size_t paritySize =
      (size_t)(fs->base->RowBlockCount * fs->base->CirculantSize);
    fs->recomputedParity   = (int *)calloc(paritySize, sizeof(int));
    fs->classicalParity    = (int *)calloc(paritySize, sizeof(int));
    fs->parityMismatchBits = (int *)calloc(paritySize, sizeof(int));
  }
  fs->violatedLayer = (int *)calloc((size_t)fs->base->RowBlockCount, sizeof(int));

  /* --- Feedback state buffers --- */
  fs->feedbackRowNextShift = (int *)calloc((size_t)fs->base->RowBlockCount, sizeof(int));
  fs->feedbackRowLastCol1  = (int *)malloc((size_t)fs->base->RowBlockCount * sizeof(int));
  fs->feedbackRowLastCol2  = (int *)malloc((size_t)fs->base->RowBlockCount * sizeof(int));
  fs->feedbackShiftDeltas  =
    (int *)calloc((size_t)(fs->base->RowBlockCount * fs->base->ColBlockCount), sizeof(int));

  if (fs->feedbackRowLastCol1 != NULL && fs->feedbackRowLastCol2 != NULL) {
    for (rowInit = 0; rowInit < fs->base->RowBlockCount; rowInit++) {
      fs->feedbackRowLastCol1[rowInit] = -1;
      fs->feedbackRowLastCol2[rowInit] = -1;
    }
  }

  /* --- ML feature buffer --- */
  fs->features = (int8_t *)calloc((size_t)fs->featureDim, sizeof(int8_t));

  /* --- Original ShiftMatrix backup (feedback modes only) --- */
  fs->originalShiftMatrix = NULL;
  fs->shiftMatrixSize     = 0;
  if (fs->config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
      fs->config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int row, col, idx;
    fs->shiftMatrixSize = fs->base->RowBlockCount * fs->base->ColBlockCount;
    fs->originalShiftMatrix =
      (int *)malloc((size_t)fs->shiftMatrixSize * sizeof(int));
    if (fs->originalShiftMatrix != NULL) {
      idx = 0;
      for (row = 0; row < fs->base->RowBlockCount; row++) {
        for (col = 0; col < fs->base->ColBlockCount; col++) {
          fs->originalShiftMatrix[idx++] = fs->base->ShiftMatrix[row][col];
        }
      }
    }
  }

  /* --- Verify all allocations succeeded --- */
  if (fs->unsatCounts == NULL    || fs->satCounts == NULL       ||
      fs->flipCounts == NULL     || fs->candidateIdx == NULL    ||
      fs->labels == NULL         || fs->recentEnergyHistory == NULL ||
      fs->scratchDecodedBits == NULL  || fs->scratchBitEnergy == NULL ||
      fs->scratchCheckNodeSyndrome == NULL ||
      fs->scratchLayerVariableBuffer == NULL ||
      fs->scratchShiftedLayerVariableBuffer == NULL ||
      fs->scratchUnsatCounts == NULL || fs->scratchSatCounts == NULL ||
      fs->recomputedParity == NULL   || fs->classicalParity == NULL  ||
      fs->parityMismatchBits == NULL || fs->violatedLayer == NULL    ||
      fs->feedbackRowNextShift == NULL || fs->feedbackRowLastCol1 == NULL ||
      fs->feedbackRowLastCol2 == NULL  || fs->feedbackShiftDeltas == NULL ||
      fs->features == NULL) {
    FrameStateFree(fs);
    return 1;
  }

  return 0;
}

void FrameStateFree(FrameState *fs)
{
  if (fs == NULL) return;

  free(fs->unsatCounts);           fs->unsatCounts = NULL;
  free(fs->satCounts);             fs->satCounts   = NULL;
  free(fs->flipCounts);            fs->flipCounts  = NULL;
  free(fs->candidateIdx);          fs->candidateIdx = NULL;
  free(fs->labels);                fs->labels = NULL;
  free(fs->recentEnergyHistory);   fs->recentEnergyHistory = NULL;

  free(fs->scratchDecodedBits);    fs->scratchDecodedBits = NULL;
  free(fs->scratchBitEnergy);      fs->scratchBitEnergy   = NULL;
  free(fs->scratchCheckNodeSyndrome);         fs->scratchCheckNodeSyndrome = NULL;
  free(fs->scratchLayerVariableBuffer);       fs->scratchLayerVariableBuffer = NULL;
  free(fs->scratchShiftedLayerVariableBuffer);
    fs->scratchShiftedLayerVariableBuffer = NULL;
  free(fs->scratchUnsatCounts);    fs->scratchUnsatCounts = NULL;
  free(fs->scratchSatCounts);      fs->scratchSatCounts   = NULL;

  free(fs->recomputedParity);      fs->recomputedParity   = NULL;
  free(fs->classicalParity);       fs->classicalParity    = NULL;
  free(fs->parityMismatchBits);    fs->parityMismatchBits = NULL;
  free(fs->violatedLayer);         fs->violatedLayer = NULL;

  free(fs->feedbackRowNextShift);  fs->feedbackRowNextShift = NULL;
  free(fs->feedbackRowLastCol1);   fs->feedbackRowLastCol1  = NULL;
  free(fs->feedbackRowLastCol2);   fs->feedbackRowLastCol2  = NULL;
  free(fs->feedbackShiftDeltas);   fs->feedbackShiftDeltas  = NULL;

  free(fs->features);              fs->features = NULL;
  free(fs->originalShiftMatrix);   fs->originalShiftMatrix = NULL;

  free(fs->errorIndexCorrectedIter); fs->errorIndexCorrectedIter = NULL;

  StagnationStateFree(&fs->stagnationState);
}
