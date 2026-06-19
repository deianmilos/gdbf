/*
 * decoder_framework.c — Top-level frame decode orchestrator.
 *
 * DecodeFrameWithConfig() is the single public entry point.  It:
 *   1. Validates arguments and fills a FrameState from its parameters.
 *   2. Calls FrameStateAlloc() to provision working buffers.
 *   3. Runs the main decode loop, delegating to:
 *        RunFeedbackRound()  — feedback-shift sender/receiver logic
 *        RunMlRound()        — ML candidate inference and bit-flip
 *        ApplyAuxEquEnergy() — energy update for active auxiliary equations
 *   4. Writes output values, restores the shift matrix, and frees buffers.
 *
 * All per-frame working state lives in FrameState (see include/framework/frame_state.h).
 * Allocation/teardown lives in src/framework/frame_setup.c.
 */

#include "frame_state.h"
#include "decoder.h"
#include "decoder_config.h"
#include "decoder_perturb.h"
#include "decoder_receiver.h"
#include "stagnation_detection.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* ApplyAuxEquEnergy — add one auxiliary equation's energy contribution*/
/* Called each decode iteration while a feedback mask is active.       */
/* ------------------------------------------------------------------ */
static void ApplyAuxEquEnergy(
  const BaseMatrixData *base,
  const int *decodedBits,
  int codeLength, int xLength,
  int *bitEnergy, int *checkNodeSyndrome,
  int *layerVariableBuffer, int *shiftedLayerVariableBuffer,
  int row,
  int col1, int delta1,
  int col2, int delta2,
  int col3, int delta3,
  int auxWeight)
{
  int Z = base->CirculantSize;
  int block, i;

  for (i = 0; i < Z; i++) checkNodeSyndrome[i] = 0;

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[row][block];
    int blockStart, src;
    if (shift == -1) continue;
    if      (block == col1) shift = (shift + delta1 + Z) % Z;
    else if (block == col2) shift = (shift + delta2 + Z) % Z;
    else if (block == col3) shift = (shift + delta3 + Z) % Z;

    blockStart = block * Z;
    src = shift;
    for (i = 0; i < Z; i++) layerVariableBuffer[i] = decodedBits[blockStart + i];
    for (i = 0; i < Z; i++) {
      shiftedLayerVariableBuffer[i] = layerVariableBuffer[src];
      if (++src == Z) src = 0;
    }
    for (i = 0; i < Z; i++) checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[row][block];
    int blockStart, dst;
    if (shift == -1) continue;
    if      (block == col1) shift = (shift + delta1 + Z) % Z;
    else if (block == col2) shift = (shift + delta2 + Z) % Z;
    else if (block == col3) shift = (shift + delta3 + Z) % Z;

    blockStart = block * Z;
    dst = shift;
    for (i = 0; i < Z; i++) {
      layerVariableBuffer[dst] = checkNodeSyndrome[i];
      if (++dst == Z) dst = 0;
    }
    for (i = 0; i < Z; i++) {
      int bitIndex = blockStart + i;
      if (bitIndex < codeLength && bitIndex < xLength) {
        bitEnergy[bitIndex] += (layerVariableBuffer[i] == 0) ? -auxWeight : auxWeight;
      }
    }
  }
}

/* ================================================================== */
/* DecodeFrameWithConfig — public API                                  */
/* ================================================================== */
int DecodeFrameWithConfig(
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
  int *frameBitErrors,
  int *addedAuxEquations,
  int *shiftMatrixGenerations,
  int *maxEnergyBitsBeforeFeedbackMin,
  int *maxEnergyBitsBeforeFeedbackMax,
  long long *maxEnergyBitsBeforeFeedbackSum,
  int *maxEnergyBitsBeforeFeedbackCount,
  int *unsuccessfulRoundsToSyndrome0,
  int *lastBitEnergyHistory,
  int *lastBitEnergyHistoryCount,
  int *mlProposedIndices,
  int *mlProposedCount,
  int mlProposedCapacity,
  const DecoderConfig *config,
  DecoderRuntimeStats *runtimeStats,
  FILE *datasetFile,
  int frameNumber,
  float alpha)
{
  FrameState fs;
  int iter;

  if (base == NULL || receivedword == NULL || codeword == NULL ||
      decodedBits == NULL || bitEnergy == NULL || checkNodeSyndrome == NULL ||
      layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL ||
      isCodeword == NULL || usedIterations == NULL ||
      frameBitErrors == NULL || config == NULL) {
    return 1;
  }

  /* ---- Fill FrameState from function arguments ---- */
  memset(&fs, 0, sizeof(fs));

  /* Initialize firstStuckIter to -1 to indicate "never stuck" state */
  fs.firstStuckIter = -1;

  fs.base                 = base;
  fs.receivedword         = receivedword;
  fs.codeword             = codeword;
  fs.codeLength           = codeLength;
  fs.maxDecoderIterations = maxDecoderIterations;
  fs.config               = config;
  fs.runtimeStats         = runtimeStats;
  fs.datasetFile          = datasetFile;
  fs.frameNumber          = frameNumber;
  fs.alpha                = alpha;

  fs.isCodeword                       = isCodeword;
  fs.usedIterations                   = usedIterations;
  fs.frameBitErrors                   = frameBitErrors;
  fs.addedAuxEquations                = addedAuxEquations;
  fs.shiftMatrixGenerations           = shiftMatrixGenerations;
  fs.maxEnergyBitsBeforeFeedbackMin   = maxEnergyBitsBeforeFeedbackMin;
  fs.maxEnergyBitsBeforeFeedbackMax   = maxEnergyBitsBeforeFeedbackMax;
  fs.maxEnergyBitsBeforeFeedbackSum   = maxEnergyBitsBeforeFeedbackSum;
  fs.maxEnergyBitsBeforeFeedbackCount = maxEnergyBitsBeforeFeedbackCount;
  fs.unsuccessfulRoundsToSyndrome0    = unsuccessfulRoundsToSyndrome0;
  fs.lastBitEnergyHistory             = lastBitEnergyHistory;
  fs.lastBitEnergyHistoryCount        = lastBitEnergyHistoryCount;
  fs.mlProposedIndices                = mlProposedIndices;
  fs.mlProposedCount                  = mlProposedCount;
  fs.mlProposedCapacity               = mlProposedCapacity;

  fs.decodedBits              = decodedBits;
  fs.bitEnergy                = bitEnergy;
  fs.checkNodeSyndrome        = checkNodeSyndrome;
  fs.layerVariableBuffer      = layerVariableBuffer;
  fs.shiftedLayerVariableBuffer = shiftedLayerVariableBuffer;

  /* Derived lengths */
  fs.xLength = X_LENGTH(codeLength, base);
  if (fs.xLength <= 0 || fs.xLength > codeLength) return 1;
  if (config->quantumOnlySyndrome) fs.xLength = codeLength;

  /* Config-derived feedback values */
  fs.feedbackLogsEnabled          = (config->feedbackLogsEnabled != 0);
  fs.feedbackIntervalIters        = (config->feedbackMaskWindowIters > 0)
                                      ? config->feedbackMaskWindowIters : 4;
  fs.feedbackDeltaMax             = (config->feedbackDeltaMax > 0)
                                      ? config->feedbackDeltaMax : 3;
  fs.feedbackTargetRows           = (config->feedbackTargetRows > 0)
                                      ? config->feedbackTargetRows : 6;
  if (fs.feedbackTargetRows > 128) fs.feedbackTargetRows = 128;

  fs.auxWeight       = 1;
  fs.lastFeedbackIter = -1000000;
  fs.frameMaxEnergyBitsBeforeFeedbackMin = 1000000;

  /* Selection / labeling configs */
  fs.selectionConfig.type           = config->candidateSelection;
  fs.selectionConfig.candidateCount = (config->candidateCount > fs.xLength)
                                        ? fs.xLength : config->candidateCount;
  fs.featureConfig.featureFlags      = config->featureFlags;
  fs.featureConfig.explicitSelection = config->featureSelectionExplicit;
  fs.featureConfig.candidateCount    = fs.selectionConfig.candidateCount;
  fs.labelingConfig.type                   = config->labelingStrategy;
  fs.labelingConfig.convergenceBonus       = codeLength;

  fs.selectionStrategy = GetCandidateSelectionStrategy(fs.selectionConfig.type);
  fs.labelingStrategy  = GetLabelingStrategy(fs.labelingConfig.type);

  /* ---- Allocate working buffers ---- */
  if (FrameStateAlloc(&fs) != 0) return 1;

  /* ---- Initialise output pointers ---- */
  if (addedAuxEquations != NULL)              *addedAuxEquations = 0;
  if (maxEnergyBitsBeforeFeedbackMin != NULL)  *maxEnergyBitsBeforeFeedbackMin = 0;
  if (maxEnergyBitsBeforeFeedbackMax != NULL)  *maxEnergyBitsBeforeFeedbackMax = 0;
  if (maxEnergyBitsBeforeFeedbackSum != NULL)  *maxEnergyBitsBeforeFeedbackSum = 0;
  if (maxEnergyBitsBeforeFeedbackCount != NULL) *maxEnergyBitsBeforeFeedbackCount = 0;
  if (unsuccessfulRoundsToSyndrome0 != NULL)   *unsuccessfulRoundsToSyndrome0 = -1;

  /* ---- Initialise decoded bits from received word ---- */
  InitDecodedFromReceived(receivedword, decodedBits, codeLength);
  *isCodeword     = 0;
  *usedIterations = 0;
  fs.recentEnergyCount = 0;

  /* ================================================================
   * MAIN DECODE LOOP
   * ================================================================ */
  for (iter = 0; iter < maxDecoderIterations; iter++) {
    FeedbackResult fbr;
    MlResult       mlr;
    int syndromeFlag;

    fs.iter = iter;

    /* -- 1. Layer pass: update bit energies and syndrome ----------- */
    syndromeFlag = LayerPassAndBitMetrics(
      base, decodedBits, receivedword, codeLength, fs.xLength,
      bitEnergy, checkNodeSyndrome,
      layerVariableBuffer, shiftedLayerVariableBuffer,
      fs.unsatCounts, fs.satCounts, &fs.syndromeWeight);

    /* -- 2. Convergence check: stop if syndrome is zero ----------- */
    if (!syndromeFlag) {
      if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
           config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
          fs.feedbackRounds > 0) {
        FEEDBACK_LOG(fs.feedbackLogsEnabled,
                     "[receiver_calc] converged with syndrome 0 at iter=%d after %d round(s)\n",
                     iter + 1, fs.feedbackRounds);
      }
      if (unsuccessfulRoundsToSyndrome0 != NULL) {
        *unsuccessfulRoundsToSyndrome0 =
          ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
            config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
           fs.feedbackRounds > 0)
            ? (fs.feedbackRounds - 1) : -1;
      }
      *isCodeword     = 1;
      *usedIterations = iter + 1;
      break;
    }

    /* -- 3. Apply active auxiliary equation energy updates --------- */
    if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
         config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
        fs.auxMaskCount > 0 &&
        fs.auxRoundsRemaining > 0) {
      int m;
      for (m = 0; m < fs.auxMaskCount; m++) {
        ApplyAuxEquEnergy(
          base, decodedBits, codeLength, fs.xLength,
          bitEnergy, checkNodeSyndrome,
          layerVariableBuffer, shiftedLayerVariableBuffer,
          fs.auxMaskRows[m],
          fs.auxMaskCol1[m],   fs.auxMaskDelta1[m],
          fs.auxMaskCol2[m],   fs.auxMaskDelta2[m],
          fs.auxMaskCol3[m],   fs.auxMaskDelta3[m],
          fs.auxWeight);
      }
      if (fs.auxRoundsRemaining > 0)
        fs.auxRoundsRemaining--;
      FEEDBACK_LOG(fs.feedbackLogsEnabled,
                   "[receiver_calc] aux_equations_active masks=%d remaining_iter_window=%d\n",
                   fs.auxMaskCount, fs.auxRoundsRemaining);
    }

    /* -- 4. Energy tracking and stagnation detection --------------- */
    fs.maxEnergy = FindMaxEnergy(bitEnergy, fs.xLength);
    fs.maxEnergyBitIdx = 0;
    while (fs.maxEnergyBitIdx < fs.xLength &&
           bitEnergy[fs.maxEnergyBitIdx] != fs.maxEnergy) {
      fs.maxEnergyBitIdx++;
    }
    if (fs.maxEnergyBitIdx >= fs.xLength) fs.maxEnergyBitIdx = 0;
    fs.isStuck = StagnationStateUpdate(&fs.stagnationState,
                                        &config->stagnation, fs.maxEnergy);

    /* -- 5. Feedback round: propose new parity-check shifts -------- */
    fbr = RunFeedbackRound(&fs);
    if (fbr == FEEDBACK_STOP_FRAME)    break;
    if (fbr == FEEDBACK_CONTINUE_ITER) continue;

    /* -- 6. ML round: candidate-driven bit flips ------------------- */
    mlr = RunMlRound(&fs);
    if (mlr == ML_CONTINUE_ITER) continue;

    /* -- 7. Core perturbation: baseline GDBF flip ------------------ */
    FlipAtMaxEnergyTrack(
      decodedBits, bitEnergy, fs.xLength,
      config->decoderType, config->pgdbfFlipProbability,
      fs.flipCounts);
  }

  /* ================================================================
   * POST-LOOP: write outputs, restore shift matrix, free buffers
   * ================================================================ */
  *frameBitErrors = CountBitErrors(decodedBits, codeword, codeLength);

  if (addedAuxEquations != NULL)
    *addedAuxEquations = fs.auxMaskCount;
  if (shiftMatrixGenerations != NULL)
    *shiftMatrixGenerations = fs.frameShiftMatrixGenerations;
  if (maxEnergyBitsBeforeFeedbackCount != NULL)
    *maxEnergyBitsBeforeFeedbackCount = fs.frameMaxEnergyBitsBeforeFeedbackCount;
  if (maxEnergyBitsBeforeFeedbackSum != NULL)
    *maxEnergyBitsBeforeFeedbackSum = fs.frameMaxEnergyBitsBeforeFeedbackSum;
  if (maxEnergyBitsBeforeFeedbackMin != NULL)
    *maxEnergyBitsBeforeFeedbackMin =
      (fs.frameMaxEnergyBitsBeforeFeedbackCount > 0)
        ? fs.frameMaxEnergyBitsBeforeFeedbackMin : 0;
  if (maxEnergyBitsBeforeFeedbackMax != NULL)
    *maxEnergyBitsBeforeFeedbackMax =
      (fs.frameMaxEnergyBitsBeforeFeedbackCount > 0)
        ? fs.frameMaxEnergyBitsBeforeFeedbackMax : 0;

  if (lastBitEnergyHistory != NULL && lastBitEnergyHistoryCount != NULL) {
    int copyCount = (fs.recentEnergyCount < FRAME_ENERGY_HISTORY_DEPTH)
                      ? fs.recentEnergyCount : FRAME_ENERGY_HISTORY_DEPTH;
    int i, j;
    *lastBitEnergyHistoryCount = copyCount;
    for (i = 0; i < copyCount; i++) {
      int srcSlot   = (fs.recentEnergyCount - copyCount + i) % FRAME_ENERGY_HISTORY_DEPTH;
      int srcOffset = srcSlot * codeLength;
      int dstOffset = i * codeLength;
      for (j = 0; j < codeLength; j++)
        lastBitEnergyHistory[dstOffset + j] =
          fs.recentEnergyHistory[srcOffset + j];
    }
  }

  /* Restore original shift matrix snapshot for feedback-enabled modes. */
  if (fs.originalShiftMatrix != NULL) {
    int row, col, idx = 0;
    for (row = 0; row < base->RowBlockCount; row++) {
      for (col = 0; col < base->ColBlockCount; col++) {
        ((int **)base->ShiftMatrix)[row][col] = fs.originalShiftMatrix[idx++];
      }
    }
  }

  FrameStateFree(&fs);

  return 0;
}
