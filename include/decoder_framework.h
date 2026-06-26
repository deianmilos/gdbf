#ifndef DECODER_FRAMEWORK_H
#define DECODER_FRAMEWORK_H

#include "common.h"
#include "candidate_selection.h"
#include "feature_extractor.h"
#include "labeling_strategy.h"
#include "stagnation_detection.h"

typedef enum {
  DECODER_TYPE_GDBF = 0,
  DECODER_TYPE_PGDBF = 1,
  DECODER_TYPE_ML = 2,
  DECODER_TYPE_FEEDBACK_SHIFT = 3,
  DECODER_TYPE_ML_FEEDBACK = 4
} DecoderType;

typedef enum {
  ML_TRIGGER_NONE = 0,
  ML_TRIGGER_PERIODIC = 1,
  ML_TRIGGER_STATE_BASED = 2
} MLTriggerMode;

typedef struct {
  DecoderType decoderType;
  CandidateSelectionType candidateSelection;
  LabelingStrategyType labelingStrategy;
  int candidateCount;
  int featureFlags;
  int featureSelectionExplicit;
  int enableDatasetCollection;
  double pgdbfFlipProbability;
  int allowedWorsening;
  int feedbackTriggerIter;
  int feedbackMaskWindowIters;
  int feedbackTargetRows;
  int feedbackPrioritySeverityW;
  int feedbackPriorityPersistenceW;
  int feedbackPriorityFailW;
  int feedbackShiftColumnsPerRow;
  int feedbackSenderStrictObservability;
  int feedbackLogsEnabled;
  int feedbackRowSelectionMode;  /* 0 = max severity (default), 1 = min non-zero severity */
  int feedbackTargetColumnMode;  /* 0 = next participating column, 1 = participating column with minimum energy */
  int mlInvokeOnlyIfBaselineFails;
  MLTriggerMode mlTriggerMode;        /* NONE | PERIODIC | STATE_BASED */
  int mlPeriodicInterval;             /* periodic: ML every N iterations */
  int mlSingleTriggerIter;            /* one-shot ML trigger at this 1-based iteration (0 disables) */
  int mlStagnationPeriodicInterval;   /* state-based: periodic ML while stagnating (0 disables) */
  int mlOscillationPeriodicInterval;  /* state-based: periodic ML while oscillating (0 disables) */
  int mlStartAfterStuck;              /* 1 = enable periodic ML only after first stuck state, 0 = disable (default) */
  int mlMaxEnergySeedMode; /* 0 = single max-energy seed, 1 = run inference for all max-energy seeds */
  char resultsExperimentName[128]; /* Optional extra subfolder under results/<code>/<mode>/ */
  int syndromeIncludesParityBits; /* 0 = syndrome over information bits only, 1 = syndrome over full codeword (includes parity bits) */
  StagnationConfig stagnation;
} DecoderConfig;

typedef struct {
  long long stagnationEvents;
  long long mlCounterfactualSkips;
  long long modelInferenceCalls;
  long long mlEscapes;
  long long mlCorrectedBits;
  long long datasetRows;
} DecoderRuntimeStats;

void DecoderConfigInitDefaults(DecoderConfig *config);
int DecoderConfigLoadFromFile(DecoderConfig *config, const char *filePath);
void DecoderConfigApplyEnv(DecoderConfig *config);
int DecoderFeatureDimension(const DecoderConfig *config);
const char *DecoderFeatureName(const DecoderConfig *config, int localFeatureIndex);

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
  int *shiftMatrixGenerations,  /* Number of layer shift matrix proposals (each layer = 1 generation) */
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
  float alpha);

#endif
