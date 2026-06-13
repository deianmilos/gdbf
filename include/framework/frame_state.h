#ifndef FRAME_STATE_H
#define FRAME_STATE_H

/*
 * frame_state.h — Per-frame decode working state.
 *
 * FrameState bundles every buffer, accumulator and loop variable that belongs
 * to a single call to DecodeFrameWithConfig().  Sub-functions (RunFeedbackRound,
 * RunMlRound) receive a pointer to this struct instead of a sprawling argument
 * list, and the top-level orchestrator stays short and readable.
 *
 * Lifecycle:
 *   1. Caller zeroes and fills the const/parameter fields.
 *   2. FrameStateAlloc() allocates all dynamic buffers.
 *   3. Decode loop runs, calling sub-functions as needed.
 *   4. FrameStateFree() releases every buffer (safe even after a partial alloc).
 */

#include "decoder_framework.h"   /* DecoderConfig, DecoderType, DecoderRuntimeStats */
#include <stdio.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Depth of the rolling bit-energy snapshot kept for the history log   */
/* ------------------------------------------------------------------ */
#define FRAME_ENERGY_HISTORY_DEPTH 5

/* ------------------------------------------------------------------ */
/* Feedback logging helper macro (used by framework and feedback_round) */
/* ------------------------------------------------------------------ */
#define FEEDBACK_LOG(enabled, ...) \
  do { if (enabled) { printf(__VA_ARGS__); } } while (0)

/* ------------------------------------------------------------------ */
/* Return codes from RunFeedbackRound()                                 */
/* ------------------------------------------------------------------ */
typedef enum {
  FEEDBACK_NONE         = 0,  /* feedback did not trigger this iteration        */
  FEEDBACK_STOP_FRAME   = 1,  /* equation cap reached: stop decoding this frame */
  FEEDBACK_CONTINUE_ITER = 2  /* new masks applied: skip to next decode iteration */
} FeedbackResult;

/* ------------------------------------------------------------------ */
/* Return codes from RunMlRound()                                       */
/* ------------------------------------------------------------------ */
typedef enum {
  ML_NONE         = 0,  /* ML did not apply a flip                        */
  ML_CONTINUE_ITER = 1  /* ML flipped bits: skip to next decode iteration */
} MlResult;

/* ------------------------------------------------------------------ */
/* All working state for one frame decode                               */
/* ------------------------------------------------------------------ */
typedef struct FrameState {

  /* ---- Frame parameters (set from DecodeFrameWithConfig args) ---- */
  const BaseMatrixData *base;
  const int            *receivedword;
  const int            *codeword;
  int                   codeLength;
  int                   xLength;
  int                   maxDecoderIterations;
  const DecoderConfig  *config;
  DecoderRuntimeStats  *runtimeStats;
  FILE                 *datasetFile;
  const int            *errorIndexes;
  int                   errorIndexCount;
  int                   frameNumber;
  float                 alpha;

  /* ---- Output pointers (written at frame end) ---- */
  int       *isCodeword;
  int       *usedIterations;
  int       *frameBitErrors;
  int       *addedAuxEquations;
  int       *shiftMatrixGenerations;
  int       *maxEnergyBitsBeforeFeedbackMin;
  int       *maxEnergyBitsBeforeFeedbackMax;
  long long *maxEnergyBitsBeforeFeedbackSum;
  int       *maxEnergyBitsBeforeFeedbackCount;
  int       *unsuccessfulRoundsToSyndrome0;
  int       *lastBitEnergyHistory;
  int       *lastBitEnergyHistoryCount;
  int       *mlProposedIndices;
  int       *mlProposedCount;
  int        mlProposedCapacity;

  /* ---- Caller-provided decode buffers (NOT freed by FrameStateFree) ---- */
  int *decodedBits;
  int *bitEnergy;
  int *checkNodeSyndrome;
  int *layerVariableBuffer;
  int *shiftedLayerVariableBuffer;

  /* ---- Working buffers (allocated/freed by FrameStateAlloc/Free) ---- */
  int *unsatCounts;
  int *satCounts;
  int *flipCounts;
  int *candidateIdx;
  int *labels;
  int *recentEnergyHistory;
  int  recentEnergyCount;

  int *scratchDecodedBits;
  int *scratchBitEnergy;
  int *scratchCheckNodeSyndrome;
  int *scratchLayerVariableBuffer;
  int *scratchShiftedLayerVariableBuffer;
  int *scratchUnsatCounts;
  int *scratchSatCounts;

  int *recomputedParity;
  int *classicalParity;
  int *parityMismatchBits;
  int *violatedLayer;

  /* ---- ML feature buffers ---- */
  int8_t *features;
  int     featureDim;
  int     featureCount;

  /* ---- Feedback working state ---- */
  int *feedbackRowNextShift;
  int *feedbackRowLastCol1;
  int *feedbackRowLastCol2;
  int *feedbackShiftDeltas;

  int auxMaskRows[128];
  int auxMaskCol1[128];  int auxMaskDelta1[128];
  int auxMaskCol2[128];  int auxMaskDelta2[128];
  int auxMaskCol3[128];  int auxMaskDelta3[128];
  int auxMaskCount;
  int auxRoundsRemaining;
  int auxWeight;
  int feedbackRounds;
  int lastFeedbackIter;

  /* Config-derived feedback values (pre-computed at frame start) */
  int feedbackLogsEnabled;
  int feedbackIntervalIters;
  int feedbackDeltaMax;
  int feedbackTargetRows;

  /* ---- Frame accumulators ---- */
  int       frameAddedAuxEquations;
  int       frameShiftMatrixGenerations;
  int       frameMaxEnergyBitsBeforeFeedbackMin;
  int       frameMaxEnergyBitsBeforeFeedbackMax;
  long long frameMaxEnergyBitsBeforeFeedbackSum;
  int       frameMaxEnergyBitsBeforeFeedbackCount;

  /* ---- Decode loop state ---- */
  int            iter;
  int            syndromeWeight;
  int            maxEnergy;
  int            maxEnergyBitIdx;
  int            isStuck;
  int            stuckSnapshotCollected;
  StagnationState stagnationState;

  /* ---- Candidate selection / ML infrastructure ---- */
  CandidateSelectionConfig          selectionConfig;
  FeatureExtractorConfig            featureConfig;
  LabelingConfig                    labelingConfig;
  const CandidateSelectionStrategy *selectionStrategy;
  const LabelingStrategy           *labelingStrategy;

  /* ---- Original shift matrix backup (restored after frame) ---- */
  int *originalShiftMatrix;
  int  shiftMatrixSize;

  /* ---- Error index tracking (optional) ---- */
  FILE *errorIndexLogFile;
  int  *errorIndexCorrectedIter;

} FrameState;

/* ------------------------------------------------------------------ */
/* Buffer lifecycle                                                     */
/* ------------------------------------------------------------------ */

/* Allocate all dynamic buffers.  Caller must fill const/config fields first.
 * Returns 0 on success, non-zero on allocation failure (all ptrs freed). */
int  FrameStateAlloc(FrameState *fs);

/* Free every dynamically allocated buffer.  Safe after a partial alloc. */
void FrameStateFree(FrameState *fs);

/* ------------------------------------------------------------------ */
/* Decode sub-functions                                                 */
/* ------------------------------------------------------------------ */

/* Execute one feedback round for fs->iter.
 * Modifies: auxMask*, feedbackRounds, lastFeedbackIter, frameAccumulators, etc.
 * Returns FEEDBACK_NONE / FEEDBACK_CONTINUE_ITER / FEEDBACK_STOP_FRAME. */
FeedbackResult RunFeedbackRound(FrameState *fs);

/* Execute one ML step for fs->iter (if stuck or periodic trigger).
 * May flip bits in fs->decodedBits and update fs->runtimeStats.
 * Returns ML_NONE / ML_CONTINUE_ITER. */
MlResult RunMlRound(FrameState *fs);

#endif /* FRAME_STATE_H */
