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

typedef struct {
  DecoderType decoderType;
  CandidateSelectionType candidateSelection;
  LabelingStrategyType labelingStrategy;
  int candidateCount;
  int featureFlags;
  int featureSelectionExplicit;
  int enableDatasetCollection;
  int rolloutIters;
  int rolloutTargetFlipCount;
  double pgdbfFlipProbability;
  int allowedWorsening;
  int feedbackTriggerIter;
  int feedbackMaskWindowIters;
  int feedbackDeltaMax;
  int feedbackTargetRows;
  int feedbackPrioritySeverityW;
  int feedbackPriorityPersistenceW;
  int feedbackPriorityFailW;
  int feedbackShiftColumnsPerRow;
  int feedbackSenderStrictObservability;
  int feedbackLogsEnabled;
  int feedbackRowSelectionMode;  /* 0 = max severity (default), 1 = min non-zero severity */
  int feedbackShiftSourceMode;   /* 0 = fixed (sat-check guided + fallback), 1 = random, 2 = fixed-number */
  int feedbackShiftFixedDelta;   /* used when feedbackShiftSourceMode == 2 */
  int mlInvokeOnlyIfBaselineFails;
  int mlPeriodicInterval;
  int errorIndexesLoggingEnabled;
  int quantumOnlySyndrome;
  StagnationConfig stagnation;
  char errorIndexesPath[512];
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
  const int *errorIndexes,
  int errorIndexCount,
  int frameNumber,
  float alpha);

#endif
