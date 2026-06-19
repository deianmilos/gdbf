/*
 * ml_round.c — ML candidate-driven bit-flip step for one decode iteration.
 *
 * RunMlRound() runs the ML inference path when the decoder is stuck or when
 * the periodic ML interval fires.  It also handles dataset collection when
 * collect mode is active.
 *
 * Helper statics:
 *   WouldBaselineDecode  — counterfactual oracle: would plain GDBF converge?
 *   RecordProposedIndices — de-duplicate proposed candidate indices for stats.
 *   TryApplyMlMaskForCurrentCandidates — inference + rollback safety gate.
 */

#include "frame_state.h"
#include "decoder.h"
#include "decoder_ml.h"
#include "candidate_selection.h"
#include "feature_extractor.h"
#include "labeling_strategy.h"
#include "stagnation_detection.h"
#include "dataset_writer.h"
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

static void RecordProposedIndices(FrameState *fs, int candidateCount)
{
  int j;

  if (fs->mlProposedIndices == NULL || fs->mlProposedCount == NULL ||
      fs->mlProposedCapacity <= 0) {
    return;
  }

  for (j = 0; j < candidateCount; j++) {
    int idx = fs->candidateIdx[j];
    int k;
    int exists = 0;
    for (k = 0; k < *(fs->mlProposedCount); k++) {
      if (fs->mlProposedIndices[k] == idx) {
        exists = 1;
        break;
      }
    }
    if (!exists && *(fs->mlProposedCount) < fs->mlProposedCapacity) {
      fs->mlProposedIndices[*(fs->mlProposedCount)] = idx;
      (*(fs->mlProposedCount))++;
    }
  }
}

static int TryApplyMlMaskForCurrentCandidates(
  FrameState *fs,
  int candidateCount,
  int *correctedBitsAppliedOut)
{
  int oldSyndrome;
  int newSyndrome;
  int j;
  int applied = 0;
  int correctedBitsApplied = 0;
  int *flipMask;

  if (candidateCount <= 0 || correctedBitsAppliedOut == NULL) {
    return 0;
  }

  flipMask = (int *)calloc((size_t)fs->selectionConfig.candidateCount, sizeof(int));
  if (flipMask == NULL) {
    return 0;
  }

  if (!ApplyModelMaskForCandidates(
        fs->features, fs->featureCount,
        fs->selectionConfig.candidateCount, 0, flipMask)) {
    free(flipMask);
    return 0;
  }

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
      applied = 1;
    }
  }

  if (applied) {
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
      applied = 0;
      correctedBitsApplied = 0;
    }
  }

  free(flipMask);
  *correctedBitsAppliedOut = correctedBitsApplied;
  return applied;
}

/* ------------------------------------------------------------------ */
/* Public: RunMlRound                                                   */
/* ------------------------------------------------------------------ */

MlResult RunMlRound(FrameState *fs)
{
  int periodicMlTrigger;
  int stagnationMlTrigger;
  int oscillationMlTrigger;
  int shouldRunMlPath;
  int candidateCount = 0;

  /* Track when first stuck state is detected */
  if (fs->isStuck && fs->firstStuckIter < 0) {
    fs->firstStuckIter = fs->iter;
  }

  /* Determine trigger based on mlTriggerMode */
  if (fs->config->decoderType == DECODER_TYPE_ML ||
      fs->config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    
    switch (fs->config->mlTriggerMode) {
      case ML_TRIGGER_PERIODIC:
        /* Periodic mode: trigger only at fixed intervals */
        /* If mlStartAfterStuck is enabled, wait until after first stuck state */
        periodicMlTrigger =
          (fs->config->mlPeriodicInterval > 0) &&
          (((fs->iter + 1) % fs->config->mlPeriodicInterval) == 0) &&
          (!fs->config->mlStartAfterStuck || fs->firstStuckIter >= 0);
        stagnationMlTrigger = 0;
        oscillationMlTrigger = 0;
        break;

      case ML_TRIGGER_STATE_BASED:
        /* State-based (reactive) mode: trigger on stagnation/oscillation detection */
        periodicMlTrigger = 0;
        stagnationMlTrigger =
          (fs->stagnationState.stagnationCounter >= fs->config->stagnation.stagnationTrigger);
        oscillationMlTrigger =
          (fs->stagnationState.oscillationCounter >= fs->config->stagnation.oscillationTrigger);
        break;

      case ML_TRIGGER_NONE:
      default:
        /* ML disabled */
        periodicMlTrigger = 0;
        stagnationMlTrigger = 0;
        oscillationMlTrigger = 0;
        break;
    }
  } else {
    periodicMlTrigger = 0;
    stagnationMlTrigger = 0;
    oscillationMlTrigger = 0;
  }

  shouldRunMlPath = periodicMlTrigger || stagnationMlTrigger || oscillationMlTrigger;
  if (!shouldRunMlPath) return ML_NONE;

  /* ---- Dataset collection: one row per max-energy seed bit ----------
   * Every max-energy bit is a potential absorbing-set participant.
   * Writing one row per seed (with its check-neighborhood as candidates)
   * mirrors exactly the per-seed inference policy used at runtime, so the
   * training data and the inference inputs are always aligned.
   * The stuckSnapshotCollected flag still limits this to the first trigger
   * event within a stuck episode to avoid identical duplicate rows.
   * ------------------------------------------------------------------ */
  if (fs->config->enableDatasetCollection &&
      fs->datasetFile != NULL &&
      !fs->stuckSnapshotCollected) {
    int *seedIdx = (int *)calloc((size_t)fs->xLength, sizeof(int));
    if (seedIdx != NULL) {
      int seedCount = CollectMaxEnergySeedIndices(
        fs->bitEnergy, fs->xLength, seedIdx, fs->xLength);
      int seed;
      for (seed = 0; seed < seedCount; seed++) {
        int seedCandCount = BuildMaxEnergyChecksCandidatesForSeed(
          fs->base, fs->bitEnergy, fs->unsatCounts,
          fs->xLength, &fs->selectionConfig,
          seedIdx[seed], fs->candidateIdx);
        if (seedCandCount <= 0) continue;
        memset(fs->labels, 0,
               (size_t)fs->selectionConfig.candidateCount * sizeof(int));
        BuildCandidateFeatureVector(
          fs->candidateIdx, seedCandCount, fs->selectionConfig.candidateCount,
          fs->bitEnergy, fs->decodedBits, fs->receivedword,
          fs->unsatCounts, fs->satCounts, fs->flipCounts,
          &fs->featureConfig, fs->features);
        fs->labelingStrategy->label(
          fs->base, fs->receivedword, fs->codeword,
          fs->codeLength, fs->decodedBits,
          fs->candidateIdx, seedCandCount,
          &fs->labelingConfig, fs->labels);
        if (CountPositiveLabels(fs->labels, fs->selectionConfig.candidateCount) > 0) {
          SaveDatasetRow(fs->datasetFile, fs->features, fs->featureDim,
                         fs->labels, fs->selectionConfig.candidateCount);
          if (fs->runtimeStats != NULL) fs->runtimeStats->datasetRows++;
        }
      }
      free(seedIdx);
    }
    fs->stuckSnapshotCollected = 1;
  }

  /* Set up candidateIdx/candidateCount for the ML inference path below */
  fs->selectionStrategy->select(
    fs->base, fs->decodedBits, fs->bitEnergy, fs->unsatCounts,
    fs->xLength, &fs->selectionConfig,
    fs->candidateIdx, &candidateCount);

  if (candidateCount <= 0) return ML_NONE;

  /* ML inference path */
  if (fs->config->decoderType == DECODER_TYPE_ML ||
      fs->config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int shouldInvokeMl = 1;
    int flipMaskApplied = 0;
    int correctedBitsApplied = 0;

    if (fs->config->mlInvokeOnlyIfBaselineFails) {
      if (WouldBaselineDecode(fs)) {
        shouldInvokeMl = 0;
        if (fs->runtimeStats != NULL) fs->runtimeStats->mlCounterfactualSkips++;
      }
    }

    if (!shouldInvokeMl) {
      return ML_NONE;
    }

    if (fs->runtimeStats != NULL && fs->isStuck) {
      fs->runtimeStats->stagnationEvents++;
    }

    if (fs->config->candidateSelection == CANDIDATE_SELECTION_MAX_ENERGY_CHECKS &&
        fs->config->mlMaxEnergySeedMode == 1) {
      int *seedIdx;
      int seedCount;
      int seed;

      seedIdx = (int *)calloc((size_t)fs->xLength, sizeof(int));
      if (seedIdx == NULL) {
        return ML_NONE;
      }

      seedCount = CollectMaxEnergySeedIndices(
        fs->bitEnergy, fs->xLength, seedIdx, fs->xLength);

      for (seed = 0; seed < seedCount && !flipMaskApplied; seed++) {
        candidateCount = BuildMaxEnergyChecksCandidatesForSeed(
          fs->base, fs->bitEnergy, fs->unsatCounts,
          fs->xLength, &fs->selectionConfig,
          seedIdx[seed], fs->candidateIdx);

        if (candidateCount <= 0) {
          continue;
        }

        BuildCandidateFeatureVector(
          fs->candidateIdx, candidateCount, fs->selectionConfig.candidateCount,
          fs->bitEnergy, fs->decodedBits, fs->receivedword,
          fs->unsatCounts, fs->satCounts, fs->flipCounts,
          &fs->featureConfig, fs->features);

        RecordProposedIndices(fs, candidateCount);

        if (fs->runtimeStats != NULL) {
          fs->runtimeStats->modelInferenceCalls++;
        }

        flipMaskApplied = TryApplyMlMaskForCurrentCandidates(
          fs, candidateCount, &correctedBitsApplied);
      }

      free(seedIdx);
    } else {
      RecordProposedIndices(fs, candidateCount);

      if (fs->runtimeStats != NULL) {
        fs->runtimeStats->modelInferenceCalls++;
      }

      flipMaskApplied = TryApplyMlMaskForCurrentCandidates(
        fs, candidateCount, &correctedBitsApplied);
    }

    if (flipMaskApplied) {
      if (fs->runtimeStats != NULL) {
        fs->runtimeStats->mlEscapes++;
        fs->runtimeStats->mlCorrectedBits += correctedBitsApplied;
      }
      StagnationStateReset(&fs->stagnationState);
      return ML_CONTINUE_ITER;
    }
  }

  return ML_NONE;
}
