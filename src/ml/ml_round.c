/*
 * ml_round.c — ML candidate-driven bit-flip step for one decode iteration.
 *
 * RunMlRound() runs the ML inference path when the decoder is stuck or when
 * the periodic ML interval fires.  It also handles dataset collection when
 * collect mode is active.
 *
 * Helper statics:
 *   WouldBaselineDecode  — counterfactual oracle: would plain GDBF converge?
 *   SaveDatasetRow       — write one feature/label row to the dataset file.
 *   CountPositiveLabels  — count non-zero labels (used as a dataset gate).
 */

#include "frame_state.h"
#include "decoder.h"
#include "decoder_ml.h"
#include "candidate_selection.h"
#include "feature_extractor.h"
#include "labeling_strategy.h"
#include "stagnation_detection.h"
#include "decoder_receiver.h"
#include "decoder_perturb.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Static helpers (used only within this translation unit)             */
/* ------------------------------------------------------------------ */

static int WouldBaselineDecode(FrameState *fs)
{
  int t;
  int syndromeWeight = 0;
  int syndromeFlag   = 1;

  memcpy(fs->scratchDecodedBits, fs->decodedBits,
         (size_t)fs->codeLength * sizeof(int));

  FlipAtMaxEnergy(fs->scratchDecodedBits, fs->bitEnergy,
                  fs->xLength, DECODER_TYPE_GDBF, 0.0);

  for (t = fs->iter + 1; t < fs->maxDecoderIterations; t++) {
    syndromeFlag = LayerPassAndBitMetrics(
      fs->base, fs->scratchDecodedBits, fs->receivedword,
      fs->codeLength, fs->xLength,
      fs->scratchBitEnergy, fs->scratchCheckNodeSyndrome,
      fs->scratchLayerVariableBuffer, fs->scratchShiftedLayerVariableBuffer,
      fs->scratchUnsatCounts, fs->scratchSatCounts,
      &syndromeWeight);

    if (!syndromeFlag) return 1;

    FlipAtMaxEnergy(fs->scratchDecodedBits, fs->scratchBitEnergy,
                    fs->xLength, DECODER_TYPE_GDBF, 0.0);
  }
  return 0;
}

static void SaveDatasetRow(FILE *datasetFile,
                           const int8_t *features, int featureCount,
                           const int *labels,       int labelCount)
{
  int i, j;
  if (datasetFile == NULL) return;
  for (i = 0; i < featureCount; i++) {
    fprintf(datasetFile, "%d,", (int)features[i]);
  }
  for (j = 0; j < labelCount; j++) {
    fprintf(datasetFile, "%d%s", labels[j], (j < labelCount - 1) ? "," : "\n");
  }
}

static int CountPositiveLabels(const int *labels, int labelCount)
{
  int i, total = 0;
  if (labels == NULL) return 0;
  for (i = 0; i < labelCount; i++) {
    if (labels[i] > 0) total++;
  }
  return total;
}

/* ------------------------------------------------------------------ */
/* Public: RunMlRound                                                   */
/* ------------------------------------------------------------------ */

MlResult RunMlRound(FrameState *fs)
{
  int periodicMlTrigger;
  int shouldRunMlPath;
  int candidateCount = 0;

  periodicMlTrigger =
    (fs->config->decoderType == DECODER_TYPE_ML ||
     fs->config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
    (fs->config->mlPeriodicInterval > 0) &&
    (((fs->iter + 1) % fs->config->mlPeriodicInterval) == 0);

  shouldRunMlPath = fs->isStuck || periodicMlTrigger;
  if (!shouldRunMlPath) return ML_NONE;

  fs->selectionStrategy->select(
    fs->base, fs->decodedBits, fs->bitEnergy, fs->unsatCounts,
    fs->xLength, &fs->selectionConfig,
    fs->candidateIdx, &candidateCount);

  if (candidateCount <= 0) return ML_NONE;

  memset(fs->labels, 0,
         (size_t)fs->selectionConfig.candidateCount * sizeof(int));

  BuildCandidateFeatureVector(
    fs->candidateIdx, candidateCount, fs->selectionConfig.candidateCount,
    fs->bitEnergy, fs->decodedBits, fs->receivedword,
    fs->unsatCounts, fs->satCounts, fs->flipCounts,
    &fs->featureConfig, fs->features);

  if (fs->config->enableDatasetCollection) {
    fs->labelingStrategy->label(
      fs->base, fs->receivedword, fs->codeword,
      fs->codeLength, fs->decodedBits,
      fs->candidateIdx, candidateCount,
      &fs->labelingConfig, fs->labels);
  }

  if (fs->config->enableDatasetCollection &&
      fs->datasetFile != NULL &&
      !fs->stuckSnapshotCollected &&
      CountPositiveLabels(fs->labels, fs->selectionConfig.candidateCount) > 0) {
    SaveDatasetRow(fs->datasetFile, fs->features, fs->featureDim,
                   fs->labels, fs->selectionConfig.candidateCount);
    fs->stuckSnapshotCollected = 1;
    if (fs->runtimeStats != NULL) fs->runtimeStats->datasetRows++;
  }

  /* ML inference path */
  if (fs->config->decoderType == DECODER_TYPE_ML ||
      fs->config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int shouldInvokeMl     = 1;
    int flipMaskApplied    = 0;
    int correctedBitsApplied = 0;
    int oldSyndrome, newSyndrome;
    int *flipMask = (int *)calloc((size_t)fs->selectionConfig.candidateCount,
                                  sizeof(int));

    if (fs->config->mlInvokeOnlyIfBaselineFails) {
      if (WouldBaselineDecode(fs)) {
        shouldInvokeMl = 0;
        if (fs->runtimeStats != NULL) fs->runtimeStats->mlCounterfactualSkips++;
      }
    }

    if (!shouldInvokeMl) {
      free(flipMask);
      return ML_NONE;
    }

    /* Record proposed candidate indices */
    if (fs->mlProposedIndices != NULL && fs->mlProposedCount != NULL &&
        fs->mlProposedCapacity > 0) {
      int j;
      for (j = 0; j < candidateCount; j++) {
        int idx = fs->candidateIdx[j], k, exists = 0;
        for (k = 0; k < *(fs->mlProposedCount); k++) {
          if (fs->mlProposedIndices[k] == idx) { exists = 1; break; }
        }
        if (!exists && *(fs->mlProposedCount) < fs->mlProposedCapacity) {
          fs->mlProposedIndices[*(fs->mlProposedCount)] = idx;
          (*(fs->mlProposedCount))++;
        }
      }
    }

    if (fs->runtimeStats != NULL) {
      if (fs->isStuck) fs->runtimeStats->stagnationEvents++;
      fs->runtimeStats->modelInferenceCalls++;
    }

    if (flipMask != NULL &&
        ApplyModelMaskForCandidates(
          fs->features, fs->featureCount,
          fs->selectionConfig.candidateCount, 0, flipMask)) {
      int j;
      oldSyndrome = ComputeSyndromeWeightOnly(
        fs->base, fs->decodedBits,
        fs->layerVariableBuffer, fs->shiftedLayerVariableBuffer,
        fs->checkNodeSyndrome);

      for (j = 0; j < candidateCount; j++) {
        if (flipMask[j]) {
          if (fs->decodedBits[fs->candidateIdx[j]] !=
              fs->codeword[fs->candidateIdx[j]]) {
            correctedBitsApplied++;
          }
          fs->decodedBits[fs->candidateIdx[j]] ^= 1;
          fs->flipCounts[fs->candidateIdx[j]]++;
          flipMaskApplied = 1;
        }
      }

      newSyndrome = ComputeSyndromeWeightOnly(
        fs->base, fs->decodedBits,
        fs->layerVariableBuffer, fs->shiftedLayerVariableBuffer,
        fs->checkNodeSyndrome);

      if (newSyndrome > oldSyndrome + fs->config->allowedWorsening) {
        for (j = 0; j < candidateCount; j++) {
          if (flipMask[j]) {
            fs->decodedBits[fs->candidateIdx[j]] ^= 1;
            if (fs->flipCounts[fs->candidateIdx[j]] > 0) {
              fs->flipCounts[fs->candidateIdx[j]]--;
            }
          }
        }
        flipMaskApplied      = 0;
        correctedBitsApplied = 0;
      }
    }

    if (flipMaskApplied) {
      if (fs->runtimeStats != NULL) {
        fs->runtimeStats->mlEscapes++;
        fs->runtimeStats->mlCorrectedBits += correctedBitsApplied;
      }
      StagnationStateReset(&fs->stagnationState);
      free(flipMask);
      return ML_CONTINUE_ITER;
    }

    free(flipMask);
  }

  return ML_NONE;
}
