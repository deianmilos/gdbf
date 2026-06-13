#include "decoder_framework.h"
#include "decoder.h"
#include "decoder_ml.h"
#include "decoder_feedback_shift.h"
#include "decoder_receiver.h"
#include "decoder_perturb.h"
#include "decoder_config.h"

#include <stdint.h>

#define FEEDBACK_LOG(enabled, ...) do { if (enabled) { printf(__VA_ARGS__); } } while (0)

static void SaveDatasetRow(FILE *datasetFile, const int8_t *features, int featureCount, const int *labels, int labelCount)
{
  int i;
  int j;

  if (datasetFile == NULL) {
    return;
  }

  for (i = 0; i < featureCount; i++) {
    fprintf(datasetFile, "%d,", (int)features[i]);
  }
  for (j = 0; j < labelCount; j++) {
    fprintf(datasetFile, "%d%s", labels[j], (j < labelCount - 1) ? "," : "\n");
  }
}

static int CountPositiveLabels(const int *labels, int labelCount)
{
  int i;
  int total = 0;

  if (labels == NULL) {
    return 0;
  }

  for (i = 0; i < labelCount; i++) {
    if (labels[i] > 0) {
      total++;
    }
  }

  return total;
}

static int WouldBaselineDecodeFromCurrentState(
  const BaseMatrixData *base,
  const int *receivedword,
  int codeLength,
  int xLength,
  int currentIter,
  int maxDecoderIterations,
  const int *decodedBits,
  const int *bitEnergy,
  int *scratchDecodedBits,
  int *scratchBitEnergy,
  int *scratchCheckNodeSyndrome,
  int *scratchLayerVariableBuffer,
  int *scratchShiftedLayerVariableBuffer,
  int *scratchUnsatCounts,
  int *scratchSatCounts)
{
  int t;
  int syndromeWeight = 0;
  int syndromeFlag = 1;

  memcpy(scratchDecodedBits, decodedBits, (size_t)codeLength * sizeof(int));

  /* Baseline path would flip now and continue. */
  FlipAtMaxEnergy(
    scratchDecodedBits,
    bitEnergy,
    xLength,
    DECODER_TYPE_GDBF,
    0.0);

  for (t = currentIter + 1; t < maxDecoderIterations; t++) {
    syndromeFlag = LayerPassAndBitMetrics(
      base,
      scratchDecodedBits,
      receivedword,
      codeLength,
      xLength,
      scratchBitEnergy,
      scratchCheckNodeSyndrome,
      scratchLayerVariableBuffer,
      scratchShiftedLayerVariableBuffer,
      scratchUnsatCounts,
      scratchSatCounts,
      &syndromeWeight);

    if (!syndromeFlag) {
      return 1;
    }

    FlipAtMaxEnergy(
      scratchDecodedBits,
      scratchBitEnergy,
      xLength,
      DECODER_TYPE_GDBF,
      0.0);
  }

  return 0;
}

static void ApplyAuxiliaryEquationEnergy(
  const BaseMatrixData *base,
  const int *decodedBits,
  int codeLength,
  int xLength,
  int *bitEnergy,
  int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int row,
  int col1,
  int delta1,
  int col2,
  int delta2,
  int col3,
  int delta3,
  int auxWeight)
{
  int Z = base->CirculantSize;
  int block;
  int i;

  for (i = 0; i < Z; i++) {
    checkNodeSyndrome[i] = 0;
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[row][block];
    int blockStart;
    int src;

    if (shift == -1) {
      continue;
    }

    if (block == col1) {
      shift = (shift + delta1 + Z) % Z;
    } else if (block == col2) {
      shift = (shift + delta2 + Z) % Z;
    } else if (block == col3) {
      shift = (shift + delta3 + Z) % Z;
    }

    blockStart = block * Z;
    src = shift;
    for (i = 0; i < Z; i++) {
      layerVariableBuffer[i] = decodedBits[blockStart + i];
    }
    for (i = 0; i < Z; i++) {
      shiftedLayerVariableBuffer[i] = layerVariableBuffer[src];
      src++;
      if (src == Z) {
        src = 0;
      }
    }

    for (i = 0; i < Z; i++) {
      checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
    }
  }

  for (block = 0; block < base->ColBlockCount; block++) {
    int shift = base->ShiftMatrix[row][block];
    int blockStart;
    int dst;

    if (shift == -1) {
      continue;
    }

    if (block == col1) {
      shift = (shift + delta1 + Z) % Z;
    } else if (block == col2) {
      shift = (shift + delta2 + Z) % Z;
    } else if (block == col3) {
      shift = (shift + delta3 + Z) % Z;
    }

    blockStart = block * Z;
    dst = shift;
    for (i = 0; i < Z; i++) {
      layerVariableBuffer[dst] = checkNodeSyndrome[i];
      dst++;
      if (dst == Z) {
        dst = 0;
      }
    }

    for (i = 0; i < Z; i++) {
      int bitIndex = blockStart + i;
      if (bitIndex < codeLength && bitIndex < xLength) {
        bitEnergy[bitIndex] += (layerVariableBuffer[i] == 0) ? -auxWeight : auxWeight;
      }
    }
  }
}

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
  float alpha)
{
  const int kEnergyHistoryDepth = 5;
  int feedbackRounds = 0;
  int lastFeedbackIter = -1000000;
  int feedbackIntervalIters;
  int frameAddedAuxEquations = 0;
  int frameShiftMatrixGenerations = 0;  /* Count of layer shift matrix proposals (4 rows per generation) */
  int frameMaxEnergyBitsBeforeFeedbackMin = 1000000;
  int frameMaxEnergyBitsBeforeFeedbackMax = 0;
  long long frameMaxEnergyBitsBeforeFeedbackSum = 0;
  int frameMaxEnergyBitsBeforeFeedbackCount = 0;
  int auxMaskRows[128];
  int auxMaskCol1[128];
  int auxMaskDelta1[128];
  int auxMaskCol2[128];
  int auxMaskDelta2[128];
  int auxMaskCol3[128];
  int auxMaskDelta3[128];
  int auxMaskCount = 0;
  int auxRoundsRemaining = 0;
  int auxWeight = 1;
  int xLength;
  int iter;
  int *unsatCounts;
  int *satCounts;
  int *flipCounts;
  int *candidateIdx;
  int *labels;
  int *recentEnergyHistory;
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
  int *feedbackRowNextShift;
  int *feedbackRowLastCol1;
  int *feedbackRowLastCol2;
  int *feedbackShiftDeltas;
  int recentEnergyCount;
  int8_t *features;
  int featureDim;
  int featureCount;
  int feedbackLogsEnabled;
  int feedbackDeltaMax;
  int feedbackTargetRows;
  int *errorIndexCorrectedIter;
  StagnationState stagnationState;
  CandidateSelectionConfig selectionConfig;
  FeatureExtractorConfig featureConfig;
  LabelingConfig labelingConfig;
  const CandidateSelectionStrategy *selectionStrategy;
  const LabelingStrategy *labelingStrategy;
  
  /* Backup original ShiftMatrix for restoration after frame */
  int *originalShiftMatrix = NULL;
  int shiftMatrixSize = 0;

  if (base == NULL || receivedword == NULL || codeword == NULL || decodedBits == NULL ||
      bitEnergy == NULL || checkNodeSyndrome == NULL || layerVariableBuffer == NULL ||
      shiftedLayerVariableBuffer == NULL || isCodeword == NULL || usedIterations == NULL ||
      frameBitErrors == NULL || config == NULL) {
    return 1;
  }

  feedbackLogsEnabled = (config->feedbackLogsEnabled != 0);
  feedbackIntervalIters = (config->feedbackMaskWindowIters > 0) ? config->feedbackMaskWindowIters : 4;
  feedbackDeltaMax = (config->feedbackDeltaMax > 0) ? config->feedbackDeltaMax : 3;
  feedbackTargetRows = (config->feedbackTargetRows > 0) ? config->feedbackTargetRows : 6;
  if (feedbackTargetRows > 128) {
    feedbackTargetRows = 128;
  }
  errorIndexCorrectedIter = NULL;

  if (addedAuxEquations != NULL) {
    *addedAuxEquations = 0;
  }
  if (maxEnergyBitsBeforeFeedbackMin != NULL) {
    *maxEnergyBitsBeforeFeedbackMin = 0;
  }
  if (maxEnergyBitsBeforeFeedbackMax != NULL) {
    *maxEnergyBitsBeforeFeedbackMax = 0;
  }
  if (maxEnergyBitsBeforeFeedbackSum != NULL) {
    *maxEnergyBitsBeforeFeedbackSum = 0;
  }
  if (maxEnergyBitsBeforeFeedbackCount != NULL) {
    *maxEnergyBitsBeforeFeedbackCount = 0;
  }
  if (unsuccessfulRoundsToSyndrome0 != NULL) {
    *unsuccessfulRoundsToSyndrome0 = -1;
  }

  xLength = X_LENGTH(codeLength, base);
  if (xLength <= 0 || xLength > codeLength) {
    return 1;
  }

  // In quantum-only syndrome mode, all N bits are candidates for flipping
  // (parity bits are not anchored from classical channel)
  if (config->quantumOnlySyndrome) {
    xLength = codeLength;
  }

  if (StagnationStateInit(&stagnationState, config->stagnation.energyHistoryLen) != 0) {
    return 1;
  }

  selectionConfig.type = config->candidateSelection;
  selectionConfig.candidateCount = (config->candidateCount > xLength) ? xLength : config->candidateCount;
  featureConfig.featureFlags = config->featureFlags;
  featureConfig.explicitSelection = config->featureSelectionExplicit;
  featureConfig.candidateCount = selectionConfig.candidateCount;
  labelingConfig.type = config->labelingStrategy;
  labelingConfig.rolloutIters = config->rolloutIters;
  labelingConfig.rolloutTargetFlipCount = config->rolloutTargetFlipCount;
  labelingConfig.convergenceBonus = codeLength;
  
  /* Allocate working buffers */
  unsatCounts = (int *)calloc((size_t)xLength, sizeof(int));
  satCounts = (int *)calloc((size_t)xLength, sizeof(int));
  flipCounts = (int *)calloc((size_t)xLength, sizeof(int));
  candidateIdx = (int *)calloc((size_t)selectionConfig.candidateCount, sizeof(int));
  labels = (int *)calloc((size_t)selectionConfig.candidateCount, sizeof(int));
  recentEnergyHistory = (int *)calloc((size_t)(kEnergyHistoryDepth * codeLength), sizeof(int));
  
  scratchDecodedBits = (int *)calloc((size_t)codeLength, sizeof(int));
  scratchBitEnergy = (int *)calloc((size_t)codeLength, sizeof(int));
  scratchCheckNodeSyndrome = (int *)calloc((size_t)base->CirculantSize, sizeof(int));
  scratchLayerVariableBuffer = (int *)calloc((size_t)base->CirculantSize, sizeof(int));
  scratchShiftedLayerVariableBuffer = (int *)calloc((size_t)base->CirculantSize, sizeof(int));
  scratchUnsatCounts = (int *)calloc((size_t)xLength, sizeof(int));
  scratchSatCounts = (int *)calloc((size_t)xLength, sizeof(int));
  
  recomputedParity = (int *)calloc((size_t)(base->RowBlockCount * base->CirculantSize), sizeof(int));
  classicalParity = (int *)calloc((size_t)(base->RowBlockCount * base->CirculantSize), sizeof(int));
  parityMismatchBits = (int *)calloc((size_t)(base->RowBlockCount * base->CirculantSize), sizeof(int));
  violatedLayer = (int *)calloc((size_t)base->RowBlockCount, sizeof(int));
  
  feedbackRowNextShift = (int *)calloc((size_t)base->RowBlockCount, sizeof(int));
  feedbackRowLastCol1 = (int *)malloc((size_t)base->RowBlockCount * sizeof(int));
  feedbackRowLastCol2 = (int *)malloc((size_t)base->RowBlockCount * sizeof(int));
  feedbackShiftDeltas = (int *)calloc((size_t)(base->RowBlockCount * base->ColBlockCount), sizeof(int));

  if (feedbackRowLastCol1 != NULL && feedbackRowLastCol2 != NULL) {
    int rowInit;
    for (rowInit = 0; rowInit < base->RowBlockCount; rowInit++) {
      feedbackRowLastCol1[rowInit] = -1;
      feedbackRowLastCol2[rowInit] = -1;
    }
  }
  
  featureCount = GetFeatureDimension(&featureConfig);
  featureDim = featureCount * selectionConfig.candidateCount;
  features = (int8_t *)calloc((size_t)featureDim, sizeof(int8_t));
  
  /* Backup original ShiftMatrix for feedback-shift mode */
  if (config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
      config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    shiftMatrixSize = base->RowBlockCount * base->ColBlockCount;
    originalShiftMatrix = (int *)malloc((size_t)shiftMatrixSize * sizeof(int));
    if (originalShiftMatrix != NULL) {
      int row, col, i;
      i = 0;
      for (row = 0; row < base->RowBlockCount; row++) {
        for (col = 0; col < base->ColBlockCount; col++) {
          originalShiftMatrix[i++] = base->ShiftMatrix[row][col];
        }
      }
    }
  }
  
  if (unsatCounts == NULL || satCounts == NULL || flipCounts == NULL || candidateIdx == NULL || labels == NULL ||
      recentEnergyHistory == NULL || scratchDecodedBits == NULL || scratchBitEnergy == NULL ||
      scratchCheckNodeSyndrome == NULL || scratchLayerVariableBuffer == NULL ||
      scratchShiftedLayerVariableBuffer == NULL || scratchUnsatCounts == NULL ||
      scratchSatCounts == NULL || recomputedParity == NULL || classicalParity == NULL ||
      parityMismatchBits == NULL || violatedLayer == NULL || feedbackRowNextShift == NULL ||
      feedbackRowLastCol1 == NULL || feedbackRowLastCol2 == NULL ||
      feedbackShiftDeltas == NULL || features == NULL) {
    StagnationStateFree(&stagnationState);
    free(unsatCounts);
    free(satCounts);
    free(flipCounts);
    free(candidateIdx);
    free(labels);
    free(recentEnergyHistory);
    free(scratchDecodedBits);
    free(scratchBitEnergy);
    free(scratchCheckNodeSyndrome);
    free(scratchLayerVariableBuffer);
    free(scratchShiftedLayerVariableBuffer);
    free(scratchUnsatCounts);
    free(scratchSatCounts);
    free(recomputedParity);
    free(classicalParity);
    free(parityMismatchBits);
    free(violatedLayer);
    free(feedbackRowNextShift);
    free(feedbackRowLastCol1);
    free(feedbackRowLastCol2);
    free(feedbackShiftDeltas);
    free(features);
    return 1;
  }
  
  selectionStrategy = GetCandidateSelectionStrategy(selectionConfig.type);
  labelingStrategy = GetLabelingStrategy(labelingConfig.type);
  
  int syndromeWeight = 0;
  int maxEnergyBitIdx = 0;
  int maxEnergy = 0;
  int isStuck = 0;
  int stuckSnapshotCollected = 0;

  InitDecodedFromReceived(receivedword, decodedBits, codeLength);

  *isCodeword = 0;
  *usedIterations = 0;
  recentEnergyCount = 0;

  // Setup error index logging (append mode, don't overwrite)
  FILE *errorIndexLogFile = NULL;
  int fileExists = 0;
  if (config->errorIndexesLoggingEnabled && errorIndexes != NULL && errorIndexCount > 0) {
    char logFilename[512];
    snprintf(logFilename, sizeof(logFilename), "results/error_index_tracking.csv");
    
    // Check if file already exists
    FILE *checkFile = fopen(logFilename, "r");
    if (checkFile != NULL) {
      fileExists = 1;
      fclose(checkFile);
    }
    
    // Open in append mode
    errorIndexLogFile = fopen(logFilename, "a");
    
    if (errorIndexLogFile != NULL) {
      int ei;
      errorIndexCorrectedIter = (int *)calloc((size_t)errorIndexCount, sizeof(int));
      if (errorIndexCorrectedIter == NULL) {
        fclose(errorIndexLogFile);
        errorIndexLogFile = NULL;
      }

      for (ei = 0; errorIndexCorrectedIter != NULL && ei < errorIndexCount; ei++) {
        errorIndexCorrectedIter[ei] = -1;
      }

      // Write header only on first frame
      if (errorIndexLogFile != NULL && !fileExists) {
        fprintf(errorIndexLogFile, "max_energy,\tframe_id,\titeration");
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\tbit_%d_value", errorIndexes[ei]);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\tbit_%d_energy", errorIndexes[ei]);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\tbit_%d_target", errorIndexes[ei]);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\tbit_%d_corrected_iter", errorIndexes[ei]);
        }
        fprintf(errorIndexLogFile, "\n");
      }
      
      // Row 1: transmitted codeword bits for this frame
      if (errorIndexLogFile != NULL) {
        fprintf(errorIndexLogFile, "-,\t%d,\ttx", frameNumber);
        for (ei = 0; ei < errorIndexCount; ei++) {
          int idx = errorIndexes[ei];
          int targetVal = 0;
          if (idx >= 0 && idx < codeLength) {
            targetVal = codeword[idx];
          }
          fprintf(errorIndexLogFile, ",\t%d", targetVal);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\t-");
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          int idx = errorIndexes[ei];
          int targetVal = 0;
          if (idx >= 0 && idx < codeLength) {
            targetVal = codeword[idx];
          }
          fprintf(errorIndexLogFile, ",\t%d", targetVal);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\t0");
        }
        fprintf(errorIndexLogFile, "\n");

        // Row 2: received bits before any decoder iteration
        fprintf(errorIndexLogFile, "-,\t%d,\trx", frameNumber);
        for (ei = 0; ei < errorIndexCount; ei++) {
          int idx = errorIndexes[ei];
          int recvVal = 0;
          int targetVal = 0;
          if (idx >= 0 && idx < codeLength) {
            recvVal = receivedword[idx];
            targetVal = codeword[idx];
            if (errorIndexCorrectedIter != NULL && recvVal == targetVal) {
              errorIndexCorrectedIter[ei] = 0;
            }
          }
          fprintf(errorIndexLogFile, ",\t%d", recvVal);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\t-");
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          int idx = errorIndexes[ei];
          int targetVal = 0;
          if (idx >= 0 && idx < codeLength) {
            targetVal = codeword[idx];
          }
          fprintf(errorIndexLogFile, ",\t%d", targetVal);
        }
        for (ei = 0; ei < errorIndexCount; ei++) {
          fprintf(errorIndexLogFile, ",\t%d", (errorIndexCorrectedIter != NULL) ? errorIndexCorrectedIter[ei] : -1);
        }
        fprintf(errorIndexLogFile, "\n");
      }
    }
  }

  for (iter = 0; iter < maxDecoderIterations; iter++) {
    int syndromeFlag;

    syndromeFlag = LayerPassAndBitMetrics(
      base,
      decodedBits,
      receivedword,
      codeLength,
      xLength,
      bitEnergy,
      checkNodeSyndrome,
      layerVariableBuffer,
      shiftedLayerVariableBuffer,
      unsatCounts,
      satCounts,
      &syndromeWeight);

    // Log error index bits and energies for this iteration
    if (errorIndexLogFile != NULL) {
      int rowMaxEnergy = FindMaxEnergy(bitEnergy, xLength);
      fprintf(errorIndexLogFile, "%d,\t%d,\tit%d", rowMaxEnergy, frameNumber, iter);
      int ei;
      for (ei = 0; ei < errorIndexCount; ei++) {
        int idx = errorIndexes[ei];
        if (idx >= 0 && idx < codeLength) {
          int targetVal = codeword[idx];
          if (errorIndexCorrectedIter != NULL &&
              errorIndexCorrectedIter[ei] == -1 &&
              decodedBits[idx] == targetVal) {
            errorIndexCorrectedIter[ei] = iter;
          }
          fprintf(errorIndexLogFile, ",\t%d", decodedBits[idx]);
        } else {
          fprintf(errorIndexLogFile, ",\t-");
        }
      }
      for (ei = 0; ei < errorIndexCount; ei++) {
        int idx = errorIndexes[ei];
        if (idx >= 0 && idx < codeLength) {
          fprintf(errorIndexLogFile, ",\t%d", bitEnergy[idx]);
        } else {
          fprintf(errorIndexLogFile, ",\t-");
        }
      }
      for (ei = 0; ei < errorIndexCount; ei++) {
        int idx = errorIndexes[ei];
        if (idx >= 0 && idx < codeLength) {
          fprintf(errorIndexLogFile, ",\t%d", codeword[idx]);
        } else {
          fprintf(errorIndexLogFile, ",\t-");
        }
      }
      for (ei = 0; ei < errorIndexCount; ei++) {
        if (errorIndexCorrectedIter != NULL) {
          fprintf(errorIndexLogFile, ",\t%d", errorIndexCorrectedIter[ei]);
        } else {
          fprintf(errorIndexLogFile, ",\t-");
        }
      }
      fprintf(errorIndexLogFile, "\n");
    }

    if (!syndromeFlag) {
       if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
         config->decoderType == DECODER_TYPE_ML_FEEDBACK) && feedbackRounds > 0) {
        FEEDBACK_LOG(feedbackLogsEnabled,
                     "[receiver_calc] converged with syndrome 0 at iter=%d after %d sender/receiver round(s)\n",
                     iter + 1, feedbackRounds);
      }
      if (unsuccessfulRoundsToSyndrome0 != NULL) {
           if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
             config->decoderType == DECODER_TYPE_ML_FEEDBACK) && feedbackRounds > 0) {
          *unsuccessfulRoundsToSyndrome0 = feedbackRounds - 1;
        } else {
          *unsuccessfulRoundsToSyndrome0 = -1;
        }
      }
      *isCodeword = 1;
      *usedIterations = iter + 1;
      break;
    }

        if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
          config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
          auxMaskCount > 0 &&
          auxRoundsRemaining > 0) {
      int m;
      for (m = 0; m < auxMaskCount; m++) {
        ApplyAuxiliaryEquationEnergy(
          base,
          decodedBits,
          codeLength,
          xLength,
          bitEnergy,
          checkNodeSyndrome,
          layerVariableBuffer,
          shiftedLayerVariableBuffer,
          auxMaskRows[m],
            auxMaskCol1[m],
            auxMaskDelta1[m],
            auxMaskCol2[m],
            auxMaskDelta2[m],
            auxMaskCol3[m],
            auxMaskDelta3[m],
          auxWeight);
      }
      if (auxRoundsRemaining > 0) {
        auxRoundsRemaining--;
      }
      FEEDBACK_LOG(feedbackLogsEnabled,
           "[receiver_calc] aux_equations_active masks=%d remaining_iter_window=%d\n",
           auxMaskCount, auxRoundsRemaining);
    }

    maxEnergy = FindMaxEnergy(bitEnergy, xLength);
    maxEnergyBitIdx = 0;
    while (maxEnergyBitIdx < xLength && bitEnergy[maxEnergyBitIdx] != maxEnergy) {
      maxEnergyBitIdx++;
    }
    if (maxEnergyBitIdx >= xLength) {
      maxEnergyBitIdx = 0;
    }
    isStuck = StagnationStateUpdate(&stagnationState, &config->stagnation, maxEnergy);

    {
      int periodicMlTrigger =
        (config->decoderType == DECODER_TYPE_ML || config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
        (config->mlPeriodicInterval > 0) &&
        (((iter + 1) % config->mlPeriodicInterval) == 0);
      int shouldRunMlPath = isStuck || periodicMlTrigger;

    /* Feedback-shift: Modify Base Matrix (ShiftMatrix) to create new parity check equations */
        if ((config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
          config->decoderType == DECODER_TYPE_ML_FEEDBACK) &&
      iter >= config->feedbackTriggerIter &&
      (feedbackRounds == 0 || (iter - lastFeedbackIter) >= feedbackIntervalIters)) {
      int Z = base->CirculantSize;
      int violatedRows[128];
      int violatedSeverity[128];
      int violatedScore[128];
      int violatedCount = 0;
      int rowIdx;
      int colBlock;
      int totalShifts = 0;
      int deltaMaxForZ;
      if (Z <= 1) {
        continue;
      }
      deltaMaxForZ = feedbackDeltaMax;
      if (deltaMaxForZ > (Z - 1)) {
        deltaMaxForZ = Z - 1;
      }
      if (deltaMaxForZ < 1) {
        deltaMaxForZ = 1;
      }

      /* Step 1: Receiver computes syndrome to identify violated checks */
      for (rowIdx = 0; rowIdx < base->RowBlockCount; rowIdx++) {
        int block;
        int i;
        int rowUnsatCount = 0;
        
        /* Compute syndrome for this specific row */
        for (i = 0; i < Z; i++) {
          checkNodeSyndrome[i] = 0;
        }
        
        for (block = 0; block < base->ColBlockCount; block++) {
          int shift = base->ShiftMatrix[rowIdx][block];
          if (shift == -1) continue;
          
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
          
          for (i = 0; i < Z; i++) {
            checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
          }
        }
        
        /* Check if this row has violated checks */
        for (i = 0; i < Z; i++) {
          if (checkNodeSyndrome[i]) {
            rowUnsatCount++;
          }
        }
        
        if (rowUnsatCount > 0) {
          if (violatedCount < 128) {
            violatedRows[violatedCount++] = rowIdx;
            violatedSeverity[violatedCount - 1] = rowUnsatCount;
          }
        }
      }

      for (rowIdx = 0; rowIdx < violatedCount; rowIdx++) {
        /* Severity-only priority: rows with more unsatisfied checks are requested first. */
        violatedScore[rowIdx] = violatedSeverity[rowIdx];
      }

      if (violatedCount > 1) {
        int a;
        for (a = 0; a < violatedCount - 1; a++) {
          int b;
          int best = a;
          for (b = a + 1; b < violatedCount; b++) {
            if (violatedScore[b] > violatedScore[best]) {
              best = b;
            }
          }
          if (best != a) {
            int tmpRow = violatedRows[a];
            int tmpSeverity = violatedSeverity[a];
            int tmpScore = violatedScore[a];
            violatedRows[a] = violatedRows[best];
            violatedSeverity[a] = violatedSeverity[best];
            violatedScore[a] = violatedScore[best];
            violatedRows[best] = tmpRow;
            violatedSeverity[best] = tmpSeverity;
            violatedScore[best] = tmpScore;
          }
        }
      }

            FEEDBACK_LOG(feedbackLogsEnabled,
                         "[feedback][uplink] iter=%d syndrome_weight=%d violated_rows=%d\n",
                         iter, syndromeWeight, violatedCount);
            FEEDBACK_LOG(feedbackLogsEnabled,
                         "[receiver->sender][round] round=%d iter=%d window=%d\n",
                         feedbackRounds + 1, iter, feedbackIntervalIters);

      if (violatedCount > 0 && originalShiftMatrix != NULL) {
        int maxEnergyBitTieCount = 0;
        int bitIdx;

        for (bitIdx = 0; bitIdx < xLength; bitIdx++) {
          if (bitEnergy[bitIdx] == maxEnergy) {
            maxEnergyBitTieCount++;
          }
        }
        if (maxEnergyBitTieCount < frameMaxEnergyBitsBeforeFeedbackMin) {
          frameMaxEnergyBitsBeforeFeedbackMin = maxEnergyBitTieCount;
        }
        if (maxEnergyBitTieCount > frameMaxEnergyBitsBeforeFeedbackMax) {
          frameMaxEnergyBitsBeforeFeedbackMax = maxEnergyBitTieCount;
        }
        frameMaxEnergyBitsBeforeFeedbackSum += maxEnergyBitTieCount;
        frameMaxEnergyBitsBeforeFeedbackCount++;

        FEEDBACK_LOG(feedbackLogsEnabled,
                     "[receiver_calc] max_energy_bits_before_feedback=%d (request_count=%d)\n",
                     maxEnergyBitTieCount,
                     frameMaxEnergyBitsBeforeFeedbackCount);

        /* Step 2: MAX ENERGY VN-GUIDED STRATEGY
         * Strategy:
         * 1. Select a super-layer of 4 consecutive rows
         * 2. For each row in that layer, choose the participating column with highest energy bit support
         * 3. Propose one random shift for that selected column */

        int targetRows[128];               /* Violated rows where maxEnergyCol participates */
        int targetRowScore[128];
        int targetRowCount = 0;
        int i;
        
        
        /* RECEIVER -> SENDER: Uplink communication */
        FEEDBACK_LOG(feedbackLogsEnabled,
                     "\n[receiver->sender][uplink] iter=%d syndrome_weight=%d violated_rows=%d\n",
                     iter, syndromeWeight, violatedCount);
        FEEDBACK_LOG(feedbackLogsEnabled, "[receiver->sender][uplink] violated row indices: ");
        for (i = 0; i < violatedCount && i < 5; i++) {
          FEEDBACK_LOG(feedbackLogsEnabled, "%d ", violatedRows[i]);
        }
        if (violatedCount > 5) FEEDBACK_LOG(feedbackLogsEnabled, "...");
        FEEDBACK_LOG(feedbackLogsEnabled, "\n");
        
        /* RECEIVER CALC: Identify max energy VN and its column */
        FEEDBACK_LOG(feedbackLogsEnabled,
               "\n[receiver_calc] max_energy_bit_idx=%d -> col_block=%d\n",
           maxEnergy, maxEnergy / Z);
        
        /* SENDER CALC: propose masks for multiple violated rows at once. */
        {
          int maxProposedRows = (violatedCount < feedbackTargetRows) ? violatedCount : feedbackTargetRows;
          int proposedRows[128];
          int proposedCol1[128];
          int proposedDelta1[128];
          int proposedOldShift1[128];
          int proposedNewShift1[128];
          int proposedCol2[128];
          int proposedDelta2[128];
          int proposedOldShift2[128];
          int proposedNewShift2[128];
          int proposedCol3[128];
          int proposedDelta3[128];
          int proposedOldShift3[128];
          int proposedNewShift3[128];
          int proposedScore[128];
          int proposedCount = 0;

          /* LAYER-BASED SELECTION: Pick a group of 4 consecutive rows (layer 0-3, 4-7, or 8-11).
           * Scoring: Sum severity of violated rows in each layer.
           * mode=0 (max): select layer with HIGHEST total severity.
           * mode=1 (min): select layer with LOWEST total severity (but > 0). */
          int layerRowStart = 0;
          int layerRowEnd = 4;
          int selectMin = (config->feedbackRowSelectionMode == 1);
          int bestLayer = 0;
          int bestScore = -1;
          
          {
            int layer;
            
            for (layer = 0; layer < 3; layer++) {
              int rowStart = layer * 4;
              int rowEnd = rowStart + 4;
              int layerScore = 0;
              int k;
              
              /* Sum severity of violated rows in this layer */
              for (k = 0; k < violatedCount; k++) {
                if (violatedRows[k] >= rowStart && violatedRows[k] < rowEnd) {
                  layerScore += violatedSeverity[k];
                }
              }
              
              /* Update best layer based on mode */
              if (bestScore < 0) {
                bestScore = layerScore;
                bestLayer = layer;
              } else if (selectMin) {
                /* min mode: pick layer with lowest non-zero score */
                if (layerScore > 0 && layerScore < bestScore) {
                  bestScore = layerScore;
                  bestLayer = layer;
                }
              } else {
                /* max mode: pick layer with highest score */
                if (layerScore > bestScore) {
                  bestScore = layerScore;
                  bestLayer = layer;
                }
              }
            }
          }

          layerRowStart = bestLayer * 4;
          layerRowEnd = layerRowStart + 4;
          
          FEEDBACK_LOG(feedbackLogsEnabled,
                       "\n[sender_calc] selected layer [%d-%d] (mode=%s, score=%d)\n",
                       layerRowStart, layerRowEnd - 1, selectMin ? "min" : "max", bestScore);

          /* Propose masks for all 4 rows in the selected layer */
          for (int layerRow = layerRowStart; layerRow < layerRowEnd; layerRow++) {
            targetRows[targetRowCount++] = layerRow;
            
            /* Find severity score for this row if it's violated, else 0 */
            int score = 0;
            int k;
            for (k = 0; k < violatedCount; k++) {
              if (violatedRows[k] == layerRow) {
                score = violatedSeverity[k];
                break;
              }
            }
            targetRowScore[targetRowCount - 1] = score;
          }

          /* ANCHOR-MAX + SAT-CHECK TARGET SHIFT STRATEGY
           * For each row in the selected layer:
           *   1) Find the participating info-column containing the global max-energy bit
           *      => this is the ANCHOR column; it is kept UNCHANGED (no shift proposed on it).
           *   2) Find the NEXT participating (ShiftMatrix != -1) info-column after the anchor
           *      (circular), which is the TARGET column.
           *   3) For the TARGET column, scan non-selected rows that are:
           *        a) not in the selected layer
           *        b) satisfied (not in violatedRows)
           *        c) have ShiftMatrix[otherRow][targetCol] != -1
           *   4) Among those satisfied reference rows pick the one with smallest resulting delta
           *      (tiebreak: smallest row index).
           *      delta = (ShiftMatrix[refRow][targetCol] - ShiftMatrix[row][targetCol] + Z) % Z
           *   5) If no satisfied reference exists, fall back to local max->min inside the
           *      TARGET column.
           *   6) Apply shift on the TARGET column only (one column per row). */

          {
            int infoColBlocks = (xLength + Z - 1) / Z;
            int shiftSourceRandom = (config->feedbackShiftSourceMode == 1);

            FEEDBACK_LOG(feedbackLogsEnabled,
                         "[sender_calc] layer=%d anchor-max target-shift (mode=%s, info_cols=%d, global_max_energy=%d)\n",
                         bestLayer, shiftSourceRandom ? "random" : "fixed", infoColBlocks, maxEnergy);

            for (i = 0; i < targetRowCount; i++) {
              int row = targetRows[i];
              int col;

              /* Step 1: find ANCHOR column — the participating column that contains
               * the global max-energy bit in this row. */
              int anchorCol    = -1;
              int anchorMaxPos = -1;
              for (col = 0; col < infoColBlocks; col++) {
                int bitStart, bitEnd, bit;
                if (base->ShiftMatrix[row][col] == -1) continue;
                bitStart = col * Z;
                bitEnd   = bitStart + Z;
                if (bitStart >= xLength) continue;
                if (bitEnd   >  xLength) bitEnd = xLength;
                for (bit = bitStart; bit < bitEnd; bit++) {
                  if (bitEnergy[bit] == maxEnergy) {
                    anchorCol    = col;
                    anchorMaxPos = bit - bitStart;
                    break;
                  }
                }
                if (anchorCol >= 0) break;
              }

              if (anchorCol < 0) {
                FEEDBACK_LOG(feedbackLogsEnabled,
                             "[sender_calc] row=%d skipped (no participating col with global max-energy)\n", row);
                continue;
              }

              /* Step 2: find TARGET column — next participating col after anchor (circular). */
              int targetCol = -1;
              {
                int step;
                for (step = 1; step <= infoColBlocks; step++) {
                  int cand = (anchorCol + step) % infoColBlocks;
                  if (cand != anchorCol && base->ShiftMatrix[row][cand] != -1) {
                    int bs = cand * Z;
                    if (bs < xLength) { targetCol = cand; break; }
                  }
                }
              }

              if (targetCol < 0) {
                FEEDBACK_LOG(feedbackLogsEnabled,
                             "[sender_calc] row=%d skipped (no next participating target col)\n", row);
                continue;
              }

              int curShift = base->ShiftMatrix[row][targetCol];

              int chosenDelta;
              if (shiftSourceRandom) {
                if (Z > 1) {
                  chosenDelta = 1 + (rand() % (Z - 1));
                } else {
                  chosenDelta = 0;
                }
                FEEDBACK_LOG(feedbackLogsEnabled,
                             "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                             "delta=%d (random, cur_shift=%d->%d)\n",
                             row, anchorCol, targetCol, chosenDelta,
                             curShift, (curShift + chosenDelta) % Z);
              } else {
                /* Step 3+4: find best satisfied reference for targetCol from non-selected rows. */
                int refShift = -1, refDelta = -1, refRow = -1;
                {
                  int otherRow;
                  for (otherRow = 0; otherRow < base->RowBlockCount; otherRow++) {
                    int d, isSatisfied, k;
                    if (otherRow >= layerRowStart && otherRow < layerRowEnd) continue;
                    if (base->ShiftMatrix[otherRow][targetCol] == -1) continue;
                    isSatisfied = 1;
                    for (k = 0; k < violatedCount; k++)
                      if (violatedRows[k] == otherRow) { isSatisfied = 0; break; }
                    if (!isSatisfied) continue;
                    d = (base->ShiftMatrix[otherRow][targetCol] - curShift + Z) % Z;
                    if (d == 0) continue;
                    if (refShift < 0 || d < refDelta ||
                        (d == refDelta && otherRow < refRow)) {
                      refShift = base->ShiftMatrix[otherRow][targetCol];
                      refDelta = d;
                      refRow   = otherRow;
                    }
                  }
                }

                /* Step 5: fallback — local max->min within targetCol. */
                if (refDelta > 0) {
                  chosenDelta = refDelta;
                  FEEDBACK_LOG(feedbackLogsEnabled,
                               "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                               "ref_row=%d ref_shift=%d delta=%d (sat-guided, cur_shift=%d->%d)\n",
                               row, anchorCol, targetCol, refRow, refShift, chosenDelta,
                               curShift, (curShift + chosenDelta) % Z);
                } else {
                  int bitStart = targetCol * Z;
                  int bitEnd   = bitStart + Z;
                  int bit, blockMaxE, blockMinE, blockMaxPos = 0, step;
                  if (bitEnd > xLength) bitEnd = xLength;
                  blockMaxE = blockMinE = bitEnergy[bitStart];
                  for (bit = bitStart + 1; bit < bitEnd; bit++) {
                    if (bitEnergy[bit] > blockMaxE) blockMaxE = bitEnergy[bit];
                    if (bitEnergy[bit] < blockMinE) blockMinE = bitEnergy[bit];
                  }
                  for (bit = bitStart; bit < bitEnd; bit++)
                    if (bitEnergy[bit] == blockMaxE) { blockMaxPos = bit - bitStart; break; }
                  chosenDelta = 1;
                  for (step = 1; step <= Z; step++) {
                    int cPos = (blockMaxPos + step) % Z;
                    if (bitStart + cPos < bitEnd && bitEnergy[bitStart + cPos] == blockMinE) {
                      chosenDelta = step; break;
                    }
                  }
                  FEEDBACK_LOG(feedbackLogsEnabled,
                               "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                               "delta=%d (fallback local max->min, cur_shift=%d->%d)\n",
                               row, anchorCol, targetCol, chosenDelta,
                               curShift, (curShift + chosenDelta) % Z);
                }
              }

              proposedRows[proposedCount]      = row;
              proposedCol1[proposedCount]      = targetCol;
              proposedDelta1[proposedCount]    = chosenDelta;
              proposedOldShift1[proposedCount] = curShift;
              proposedNewShift1[proposedCount] = (curShift + chosenDelta) % Z;
              proposedCol2[proposedCount]      = -1;
              proposedDelta2[proposedCount]    = 0;
              proposedOldShift2[proposedCount] = -1;
              proposedNewShift2[proposedCount] = -1;
              proposedCol3[proposedCount]      = -1;
              proposedDelta3[proposedCount]    = 0;
              proposedOldShift3[proposedCount] = -1;
              proposedNewShift3[proposedCount] = -1;
              proposedScore[proposedCount]     = targetRowScore[i];
              proposedCount++;

              feedbackRowLastCol1[row] = targetCol;
              feedbackRowLastCol2[row] = -1;
              feedbackRowNextShift[row] = chosenDelta;
              feedbackShiftDeltas[row * base->ColBlockCount + targetCol] = chosenDelta + 1;
            }
          }


          FEEDBACK_LOG(feedbackLogsEnabled,
                       "\n[sender->receiver][downlink] LAYER MASKS rows=%d\n",
                       proposedCount);
          for (i = 0; i < proposedCount; i++) {
               if (proposedCol3[i] >= 0) {
                 FEEDBACK_LOG(feedbackLogsEnabled,
                              "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d col2=%d delta2=%+d col3=%d delta3=%+d\n",
                              proposedRows[i], proposedCol1[i], proposedDelta1[i], proposedCol2[i], proposedDelta2[i], proposedCol3[i], proposedDelta3[i]);
               } else if (proposedCol2[i] >= 0) {
                 FEEDBACK_LOG(feedbackLogsEnabled,
                              "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d col2=%d delta2=%+d\n",
                              proposedRows[i], proposedCol1[i], proposedDelta1[i], proposedCol2[i], proposedDelta2[i]);
               } else {
                 FEEDBACK_LOG(feedbackLogsEnabled,
                              "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d\n",
                              proposedRows[i], proposedCol1[i], proposedDelta1[i]);
               }
          }

          {
            int newMasksAdded = 0;
            auxMaskCount = 0;
          for (i = 0; i < proposedCount; i++) {
            int row = proposedRows[i];
            int shiftedCol1 = proposedCol1[i];
            int shiftedCol2 = proposedCol2[i];
            int shiftedCol3 = proposedCol3[i];
            int shiftCount = 1;
            if (shiftedCol2 >= 0) {
              shiftCount++;
            }
            if (shiftedCol3 >= 0) {
              shiftCount++;
            }

            if (auxMaskCount < 128) {
              auxMaskRows[auxMaskCount] = row;
              auxMaskCol1[auxMaskCount] = shiftedCol1;
              auxMaskDelta1[auxMaskCount] = proposedDelta1[i];
              auxMaskCol2[auxMaskCount] = shiftedCol2;
              auxMaskDelta2[auxMaskCount] = proposedDelta2[i];
              auxMaskCol3[auxMaskCount] = shiftedCol3;
              auxMaskDelta3[auxMaskCount] = proposedDelta3[i];
              auxMaskCount++;
              newMasksAdded++;
              totalShifts += shiftCount;
            }

            /* feedbackRowLastCol/NextShift/ShiftDeltas already set in proposal loop */

            if (shiftedCol3 >= 0) {
              FEEDBACK_LOG(feedbackLogsEnabled,
                           "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) col2=%d delta2=%+d (%d->%d) col3=%d delta3=%+d (%d->%d) score=%d\n",
                           row,
                           shiftedCol1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                           shiftedCol2, proposedDelta2[i], proposedOldShift2[i], proposedNewShift2[i],
                           shiftedCol3, proposedDelta3[i], proposedOldShift3[i], proposedNewShift3[i],
                           proposedScore[i]);
            } else if (shiftedCol2 >= 0) {
              FEEDBACK_LOG(feedbackLogsEnabled,
                           "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) col2=%d delta2=%+d (%d->%d) score=%d\n",
                           row,
                           shiftedCol1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                           shiftedCol2, proposedDelta2[i], proposedOldShift2[i], proposedNewShift2[i],
                           proposedScore[i]);
            } else {
              FEEDBACK_LOG(feedbackLogsEnabled,
                           "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) score=%d\n",
                           row,
                           shiftedCol1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                           proposedScore[i]);
            }
          }
          auxRoundsRemaining = feedbackIntervalIters;
          if (newMasksAdded > 0) {
            frameShiftMatrixGenerations++;
          }
          frameAddedAuxEquations += newMasksAdded;
           FEEDBACK_LOG(feedbackLogsEnabled,
                    "[receiver_calc] generation=%d aux_equation_set_size=%d active_for_next_%d_iterations\n\n",
                    frameShiftMatrixGenerations, auxMaskCount, auxRoundsRemaining);

          }
        }

          FEEDBACK_LOG(feedbackLogsEnabled,
                  "[receiver_calc] total shifts applied: %d across %d row(s)\n",
                  totalShifts, targetRowCount);
          FEEDBACK_LOG(feedbackLogsEnabled,
                  "[receiver_calc] continuing GDBF with original H + auxiliary equations...\n\n");
        
         feedbackRounds++;
         lastFeedbackIter = iter;
         FEEDBACK_LOG(feedbackLogsEnabled,
                      "[sender->receiver][round] round=%d mask_sent, receiver_will_run_next_%d_iterations\n",
                      feedbackRounds, feedbackIntervalIters);
        StagnationStateReset(&stagnationState);
        continue;  /* Continue decoding with new equations */
      }
    }

    if (shouldRunMlPath) {
      int candidateCount = 0;
      selectionStrategy->select(
        base,
        decodedBits,
        bitEnergy,
        unsatCounts,
        xLength,
        &selectionConfig,
        candidateIdx,
        &candidateCount);

      if (candidateCount > 0) {
        memset(labels, 0, (size_t)selectionConfig.candidateCount * sizeof(int));

        BuildCandidateFeatureVector(
          candidateIdx,
          candidateCount,
          selectionConfig.candidateCount,
          bitEnergy,
          decodedBits,
          receivedword,
          unsatCounts,
          satCounts,
          flipCounts,
          &featureConfig,
          features);

        if (config->enableDatasetCollection) {
          labelingStrategy->label(
            base,
            receivedword,
            codeword,
            codeLength,
            decodedBits,
            candidateIdx,
            candidateCount,
            &labelingConfig,
            labels);
        }

        if (config->enableDatasetCollection && datasetFile != NULL && !stuckSnapshotCollected &&
            CountPositiveLabels(labels, selectionConfig.candidateCount) > 0) {
          SaveDatasetRow(datasetFile, features, featureDim, labels, selectionConfig.candidateCount);
          stuckSnapshotCollected = 1;
          if (runtimeStats != NULL) {
            runtimeStats->datasetRows++;
          }
        }

        /* ML-only path: candidate-driven model inference and rollback-safe mask application. */
        if (config->decoderType == DECODER_TYPE_ML || config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
          int shouldInvokeMl = 1;
          int flipMaskApplied = 0;
          int correctedBitsApplied = 0;
          int oldSyndrome;
          int newSyndrome;
          int *flipMask = (int *)calloc((size_t)selectionConfig.candidateCount, sizeof(int));

          if (config->mlInvokeOnlyIfBaselineFails) {
            if (WouldBaselineDecodeFromCurrentState(
                  base,
                  receivedword,
                  codeLength,
                  xLength,
                  iter,
                  maxDecoderIterations,
                  decodedBits,
                  bitEnergy,
                  scratchDecodedBits,
                  scratchBitEnergy,
                  scratchCheckNodeSyndrome,
                  scratchLayerVariableBuffer,
                  scratchShiftedLayerVariableBuffer,
                  scratchUnsatCounts,
                  scratchSatCounts)) {
              shouldInvokeMl = 0;
              if (runtimeStats != NULL) {
                runtimeStats->mlCounterfactualSkips++;
              }
            }
          }

          if (!shouldInvokeMl) {
            free(flipMask);
            goto skip_ml_inference;
          }

          if (mlProposedIndices != NULL && mlProposedCount != NULL && mlProposedCapacity > 0) {
            int j;
            for (j = 0; j < candidateCount; j++) {
              int idx = candidateIdx[j];
              int k;
              int exists = 0;
              for (k = 0; k < *mlProposedCount; k++) {
                if (mlProposedIndices[k] == idx) {
                  exists = 1;
                  break;
                }
              }
              if (!exists && *mlProposedCount < mlProposedCapacity) {
                mlProposedIndices[*mlProposedCount] = idx;
                (*mlProposedCount)++;
              }
            }
          }

          if (runtimeStats != NULL) {
            if (isStuck) {
              runtimeStats->stagnationEvents++;
            }
            runtimeStats->modelInferenceCalls++;
          }

              if (flipMask != NULL && ApplyModelMaskForCandidates(
                features,
                featureCount,
                selectionConfig.candidateCount,
                0,
                flipMask)) {
            int j;
            oldSyndrome = ComputeSyndromeWeightOnly(
              base,
              decodedBits,
              layerVariableBuffer,
              shiftedLayerVariableBuffer,
              checkNodeSyndrome);

            for (j = 0; j < candidateCount; j++) {
              if (flipMask[j]) {
                if (decodedBits[candidateIdx[j]] != codeword[candidateIdx[j]]) {
                  correctedBitsApplied++;
                }
                decodedBits[candidateIdx[j]] ^= 1;
                flipCounts[candidateIdx[j]]++;
                flipMaskApplied = 1;
              }
            }

            newSyndrome = ComputeSyndromeWeightOnly(
              base,
              decodedBits,
              layerVariableBuffer,
              shiftedLayerVariableBuffer,
              checkNodeSyndrome);

            if (newSyndrome > oldSyndrome + config->allowedWorsening) {
              for (j = 0; j < candidateCount; j++) {
                if (flipMask[j]) {
                  decodedBits[candidateIdx[j]] ^= 1;
                  if (flipCounts[candidateIdx[j]] > 0) {
                    flipCounts[candidateIdx[j]]--;
                  }
                }
              }
              flipMaskApplied = 0;
              correctedBitsApplied = 0;
            }
          }

          if (flipMaskApplied) {
            if (runtimeStats != NULL) {
              runtimeStats->mlEscapes++;
              runtimeStats->mlCorrectedBits += correctedBitsApplied;
            }
            StagnationStateReset(&stagnationState);
            free(flipMask);
            continue;
          }

          free(flipMask);

skip_ml_inference:
          ;
        }
      }
    }
    }

    /* Core perturbation step (gdbf/pgdbf baseline behavior used by all decoder types). */
    FlipAtMaxEnergyTrack(
      decodedBits,
      bitEnergy,
      xLength,
      config->decoderType,
      config->pgdbfFlipProbability,
      flipCounts);
  }

  *frameBitErrors = CountBitErrors(decodedBits, codeword, codeLength);
  if (addedAuxEquations != NULL) {
    /* Report currently active auxiliary equations at end-of-frame.
     * In non-accumulate mode this is the active super-layer mask size (expected 4). */
    *addedAuxEquations = auxMaskCount;
  }
  if (shiftMatrixGenerations != NULL) {
    *shiftMatrixGenerations = frameShiftMatrixGenerations;
  }
  if (maxEnergyBitsBeforeFeedbackCount != NULL) {
    *maxEnergyBitsBeforeFeedbackCount = frameMaxEnergyBitsBeforeFeedbackCount;
  }
  if (maxEnergyBitsBeforeFeedbackSum != NULL) {
    *maxEnergyBitsBeforeFeedbackSum = frameMaxEnergyBitsBeforeFeedbackSum;
  }
  if (maxEnergyBitsBeforeFeedbackMin != NULL) {
    *maxEnergyBitsBeforeFeedbackMin =
      (frameMaxEnergyBitsBeforeFeedbackCount > 0) ? frameMaxEnergyBitsBeforeFeedbackMin : 0;
  }
  if (maxEnergyBitsBeforeFeedbackMax != NULL) {
    *maxEnergyBitsBeforeFeedbackMax =
      (frameMaxEnergyBitsBeforeFeedbackCount > 0) ? frameMaxEnergyBitsBeforeFeedbackMax : 0;
  }

  if (lastBitEnergyHistory != NULL && lastBitEnergyHistoryCount != NULL) {
    int copyCount = (recentEnergyCount < kEnergyHistoryDepth) ? recentEnergyCount : kEnergyHistoryDepth;
    int i;
    int j;

    *lastBitEnergyHistoryCount = copyCount;

    for (i = 0; i < copyCount; i++) {
      int srcSlot = (recentEnergyCount - copyCount + i) % kEnergyHistoryDepth;
      int srcOffset = srcSlot * codeLength;
      int dstOffset = i * codeLength;
      for (j = 0; j < codeLength; j++) {
        lastBitEnergyHistory[dstOffset + j] = recentEnergyHistory[srcOffset + j];
      }
    }
  }

  if (config->decoderType == DECODER_TYPE_FEEDBACK_SHIFT ||
      config->decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int Z = base->CirculantSize;
    int layer;
    BaseMatrixData *mutableBase = (BaseMatrixData *)base;
    if (Z > 0) {
      for (layer = 0; layer < base->RowBlockCount; layer++) {
        int block;
        for (block = 0; block < base->ColBlockCount; block++) {
          int offset = feedbackShiftDeltas[layer * base->ColBlockCount + block] % Z;
          int s = mutableBase->ShiftMatrix[layer][block];
          if (s != -1 && offset != 0) {
            mutableBase->ShiftMatrix[layer][block] = (s - offset + Z) % Z;
          }
        }
      }
    }
  }

  /* Restore original ShiftMatrix if it was backed up */
  if (originalShiftMatrix != NULL) {
    int i, row, col;
    for (row = 0; row < base->RowBlockCount; row++) {
      for (col = 0; col < base->ColBlockCount; col++) {
        i = row * base->ColBlockCount + col;
        ((int **)base->ShiftMatrix)[row][col] = originalShiftMatrix[i];
      }
    }
    free(originalShiftMatrix);
  }

  free(unsatCounts);
  free(satCounts);
  free(flipCounts);
  free(candidateIdx);
  free(labels);
  free(recentEnergyHistory);
  free(scratchDecodedBits);
  free(scratchBitEnergy);
  free(scratchCheckNodeSyndrome);
  free(scratchLayerVariableBuffer);
  free(scratchShiftedLayerVariableBuffer);
  free(scratchUnsatCounts);
  free(scratchSatCounts);
  free(recomputedParity);
  free(classicalParity);
  free(parityMismatchBits);
  free(violatedLayer);
  free(feedbackRowNextShift);
  free(feedbackRowLastCol1);
  free(feedbackRowLastCol2);
  free(feedbackShiftDeltas);
  free(features);
  StagnationStateFree(&stagnationState);

  if (errorIndexLogFile != NULL) {
    // Blank line between frames for readability.
    fprintf(errorIndexLogFile, "\n");
    fclose(errorIndexLogFile);
  }
  free(errorIndexCorrectedIter);

  // Log frame convergence status to summary file
  if (config->errorIndexesLoggingEnabled && frameNumber >= 0) {
    FILE *summaryFile = fopen("results/frame_summary.csv", "a");
    if (summaryFile != NULL) {
      // Check if file is new (empty) to write header
      fseek(summaryFile, 0, SEEK_END);
      if (ftell(summaryFile) == 0) {
        fprintf(summaryFile, "frame_id,alpha,status,iterations,bit_errors\n");
      }
      
      // Log frame result: "success" or "trapped"
      const char *status = (*isCodeword) ? "success" : "trapped";
      fprintf(summaryFile, "%d,%.5f,%s,%d,%d\n", frameNumber, alpha, status, *usedIterations, *frameBitErrors);
      fclose(summaryFile);
    }
  }

  return 0;
}
