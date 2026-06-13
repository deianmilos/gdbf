/*
 * feedback_round.c — Feedback-shift sender/receiver logic for one decode iteration.
 *
 * RunFeedbackRound() implements the full per-iteration feedback scheduler:
 *   1. Check whether feedback conditions are met (trigger iter, window).
 *   2. Compute which rows have violated parity checks (receiver syndrome).
 *   3. Select the best super-layer (4 consecutive rows).
 *   4. For each row, identify the anchor column (max-energy bit) and the next
 *      participating column (target), then compute the shift delta using the
 *      configured source mode (fixed = sat-check guided, random = rand delta).
 *   5. Commit the proposed masks to fs->auxMask* arrays.
 *   6. Return FEEDBACK_STOP_FRAME, FEEDBACK_CONTINUE_ITER, or FEEDBACK_NONE.
 */

#include "frame_state.h"
#include "stagnation_detection.h"

#include <stdlib.h>   /* rand() */

FeedbackResult RunFeedbackRound(FrameState *fs)
{
  int Z;
  int violatedRows[128];
  int violatedSeverity[128];
  int violatedScore[128];
  int violatedCount = 0;
  int rowIdx;
  int totalShifts = 0;
  int deltaMaxForZ;

  /* --- Trigger guard -------------------------------------------- */
  if ((fs->config->decoderType != DECODER_TYPE_FEEDBACK_SHIFT &&
       fs->config->decoderType != DECODER_TYPE_ML_FEEDBACK) ||
      fs->iter < fs->config->feedbackTriggerIter ||
      (fs->feedbackRounds > 0 &&
       (fs->iter - fs->lastFeedbackIter) < fs->feedbackIntervalIters)) {
    return FEEDBACK_NONE;
  }

  Z = fs->base->CirculantSize;
  if (Z <= 1) {
    return FEEDBACK_CONTINUE_ITER;
  }

  deltaMaxForZ = fs->feedbackDeltaMax;
  if (deltaMaxForZ > (Z - 1)) deltaMaxForZ = Z - 1;
  if (deltaMaxForZ < 1)       deltaMaxForZ = 1;

  /* --- Step 1: Receiver — compute violated rows ------------------- */
  for (rowIdx = 0; rowIdx < fs->base->RowBlockCount; rowIdx++) {
    int block, i, rowUnsatCount = 0;

    for (i = 0; i < Z; i++) {
      fs->checkNodeSyndrome[i] = 0;
    }

    for (block = 0; block < fs->base->ColBlockCount; block++) {
      int shift = fs->base->ShiftMatrix[rowIdx][block];
      int blockStart, src;
      if (shift == -1) continue;

      blockStart = block * Z;
      src = shift;
      for (i = 0; i < Z; i++) {
        fs->layerVariableBuffer[i] = fs->decodedBits[blockStart + i];
      }
      for (i = 0; i < Z; i++) {
        fs->shiftedLayerVariableBuffer[i] = fs->layerVariableBuffer[src];
        src++;
        if (src == Z) src = 0;
      }
      for (i = 0; i < Z; i++) {
        fs->checkNodeSyndrome[i] ^= fs->shiftedLayerVariableBuffer[i];
      }
    }

    for (i = 0; i < Z; i++) {
      if (fs->checkNodeSyndrome[i]) rowUnsatCount++;
    }

    if (rowUnsatCount > 0 && violatedCount < 128) {
      violatedRows[violatedCount]    = rowIdx;
      violatedSeverity[violatedCount] = rowUnsatCount;
      violatedCount++;
    }
  }

  /* Severity-only row ranking (selection sort descending) */
  for (rowIdx = 0; rowIdx < violatedCount; rowIdx++) {
    violatedScore[rowIdx] = violatedSeverity[rowIdx];
  }
  if (violatedCount > 1) {
    int a;
    for (a = 0; a < violatedCount - 1; a++) {
      int b, best = a;
      for (b = a + 1; b < violatedCount; b++) {
        if (violatedScore[b] > violatedScore[best]) best = b;
      }
      if (best != a) {
        int tr = violatedRows[a], ts = violatedSeverity[a], tc = violatedScore[a];
        violatedRows[a]     = violatedRows[best];
        violatedSeverity[a] = violatedSeverity[best];
        violatedScore[a]    = violatedScore[best];
        violatedRows[best]     = tr;
        violatedSeverity[best] = ts;
        violatedScore[best]    = tc;
      }
    }
  }

  FEEDBACK_LOG(fs->feedbackLogsEnabled,
               "[feedback][uplink] iter=%d syndrome_weight=%d violated_rows=%d\n",
               fs->iter, fs->syndromeWeight, violatedCount);
  FEEDBACK_LOG(fs->feedbackLogsEnabled,
               "[receiver->sender][round] round=%d iter=%d window=%d\n",
               fs->feedbackRounds + 1, fs->iter, fs->feedbackIntervalIters);

  if (violatedCount > 0 && fs->originalShiftMatrix != NULL) {
    int maxEnergyBitTieCount = 0;
    int bitIdx;
    int i;

    /* Track max-energy-bits-before-feedback statistics */
    for (bitIdx = 0; bitIdx < fs->xLength; bitIdx++) {
      if (fs->bitEnergy[bitIdx] == fs->maxEnergy) maxEnergyBitTieCount++;
    }
    if (maxEnergyBitTieCount < fs->frameMaxEnergyBitsBeforeFeedbackMin)
      fs->frameMaxEnergyBitsBeforeFeedbackMin = maxEnergyBitTieCount;
    if (maxEnergyBitTieCount > fs->frameMaxEnergyBitsBeforeFeedbackMax)
      fs->frameMaxEnergyBitsBeforeFeedbackMax = maxEnergyBitTieCount;
    fs->frameMaxEnergyBitsBeforeFeedbackSum += maxEnergyBitTieCount;
    fs->frameMaxEnergyBitsBeforeFeedbackCount++;

    FEEDBACK_LOG(fs->feedbackLogsEnabled,
                 "[receiver_calc] max_energy_bits_before_feedback=%d (request_count=%d)\n",
                 maxEnergyBitTieCount, fs->frameMaxEnergyBitsBeforeFeedbackCount);

    /* --- Step 2: Receiver -> Sender uplink log -------------------- */
    {
      int targetRows[128];
      int targetRowScore[128];
      int targetRowCount = 0;

      int proposedRows[128];
      int proposedCol1[128];  int proposedDelta1[128];
      int proposedOldShift1[128]; int proposedNewShift1[128];
      int proposedCol2[128];  int proposedDelta2[128];
      int proposedOldShift2[128]; int proposedNewShift2[128];
      int proposedCol3[128];  int proposedDelta3[128];
      int proposedOldShift3[128]; int proposedNewShift3[128];
      int proposedScore[128];
      int proposedCount = 0;

      FEEDBACK_LOG(fs->feedbackLogsEnabled,
                   "\n[receiver->sender][uplink] iter=%d syndrome_weight=%d violated_rows=%d\n",
                   fs->iter, fs->syndromeWeight, violatedCount);
      FEEDBACK_LOG(fs->feedbackLogsEnabled,
                   "[receiver->sender][uplink] violated row indices: ");
      for (i = 0; i < violatedCount && i < 5; i++) {
        FEEDBACK_LOG(fs->feedbackLogsEnabled, "%d ", violatedRows[i]);
      }
      if (violatedCount > 5) FEEDBACK_LOG(fs->feedbackLogsEnabled, "...");
      FEEDBACK_LOG(fs->feedbackLogsEnabled, "\n");

      FEEDBACK_LOG(fs->feedbackLogsEnabled,
                   "\n[receiver_calc] max_energy_bit_idx=%d -> col_block=%d\n",
                   fs->maxEnergy, fs->maxEnergy / Z);

      /* --- Step 3: Layer-based selection -------------------------- */
      {
        int selectMin  = (fs->config->feedbackRowSelectionMode == 1);
        int bestLayer  = 0;
        int bestScore  = -1;
        int layer;

        for (layer = 0; layer < 3; layer++) {
          int rowStart   = layer * 4;
          int rowEnd     = rowStart + 4;
          int layerScore = 0;
          int k;
          for (k = 0; k < violatedCount; k++) {
            if (violatedRows[k] >= rowStart && violatedRows[k] < rowEnd) {
              layerScore += violatedSeverity[k];
            }
          }
          if (bestScore < 0) {
            bestScore = layerScore; bestLayer = layer;
          } else if (selectMin) {
            if (layerScore > 0 && layerScore < bestScore) {
              bestScore = layerScore; bestLayer = layer;
            }
          } else {
            if (layerScore > bestScore) {
              bestScore = layerScore; bestLayer = layer;
            }
          }
        }

        {
          int layerRowStart = bestLayer * 4;
          int layerRowEnd   = layerRowStart + 4;
          int layerRow;

          FEEDBACK_LOG(fs->feedbackLogsEnabled,
                       "\n[sender_calc] selected layer [%d-%d] (mode=%s, score=%d)\n",
                       layerRowStart, layerRowEnd - 1,
                       selectMin ? "min" : "max", bestScore);

          /* Collect all 4 rows in the selected layer as target rows */
          for (layerRow = layerRowStart; layerRow < layerRowEnd; layerRow++) {
            int score = 0, k;
            targetRows[targetRowCount] = layerRow;
            for (k = 0; k < violatedCount; k++) {
              if (violatedRows[k] == layerRow) {
                score = violatedSeverity[k]; break;
              }
            }
            targetRowScore[targetRowCount] = score;
            targetRowCount++;
          }

          /* --- Step 4: ANCHOR-MAX + SAT-CHECK TARGET SHIFT STRATEGY -- */
          {
            int infoColBlocks    = (fs->xLength + Z - 1) / Z;
            int shiftSourceRandom = (fs->config->feedbackShiftSourceMode == 1);
            int shiftSourceFixedNumber = (fs->config->feedbackShiftSourceMode == 2);
            const char *shiftModeLabel = shiftSourceRandom ? "random" : (shiftSourceFixedNumber ? "fixed_number" : "fixed");

            FEEDBACK_LOG(fs->feedbackLogsEnabled,
                         "[sender_calc] layer=%d anchor-max target-shift (mode=%s, info_cols=%d, global_max_energy=%d)\n",
                         bestLayer, shiftModeLabel,
                         infoColBlocks, fs->maxEnergy);

            for (i = 0; i < targetRowCount; i++) {
              int row = targetRows[i];
              int col;
              int anchorCol = -1;
              int targetCol = -1;
              int curShift;
              int chosenDelta;

              /* Find ANCHOR column: first participating col containing the
               * global max-energy bit */
              for (col = 0; col < infoColBlocks; col++) {
                int bitStart, bitEnd, bit;
                if (fs->base->ShiftMatrix[row][col] == -1) continue;
                bitStart = col * Z;
                bitEnd   = bitStart + Z;
                if (bitStart >= fs->xLength) continue;
                if (bitEnd   >  fs->xLength) bitEnd = fs->xLength;
                for (bit = bitStart; bit < bitEnd; bit++) {
                  if (fs->bitEnergy[bit] == fs->maxEnergy) {
                    anchorCol = col;
                    break;
                  }
                }
                if (anchorCol >= 0) break;
              }

              if (anchorCol < 0) {
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[sender_calc] row=%d skipped (no participating col with global max-energy)\n",
                             row);
                continue;
              }

              /* Find TARGET column: next participating col after anchor (circular) */
              {
                int step;
                for (step = 1; step <= infoColBlocks; step++) {
                  int cand = (anchorCol + step) % infoColBlocks;
                  if (cand != anchorCol && fs->base->ShiftMatrix[row][cand] != -1) {
                    int bs = cand * Z;
                    if (bs < fs->xLength) { targetCol = cand; break; }
                  }
                }
              }

              if (targetCol < 0) {
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[sender_calc] row=%d skipped (no next participating target col)\n",
                             row);
                continue;
              }

              curShift = fs->base->ShiftMatrix[row][targetCol];

              /* Compute shift delta */
              if (shiftSourceFixedNumber) {
                if (Z > 1) {
                  chosenDelta = fs->config->feedbackShiftFixedDelta % Z;
                  if (chosenDelta == 0) {
                    chosenDelta = 1;
                  }
                } else {
                  chosenDelta = 0;
                }
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                             "delta=%d (fixed_number=%d, cur_shift=%d->%d)\n",
                             row, anchorCol, targetCol, chosenDelta,
                             fs->config->feedbackShiftFixedDelta,
                             curShift, (curShift + chosenDelta) % Z);
              } else if (shiftSourceRandom) {
                chosenDelta = (Z > 1) ? (1 + (rand() % (Z - 1))) : 0;
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                             "delta=%d (random, cur_shift=%d->%d)\n",
                             row, anchorCol, targetCol, chosenDelta,
                             curShift, (curShift + chosenDelta) % Z);
              } else {
                /* Fixed mode: find sat-check guided delta */
                int refShift = -1, refDelta = -1, refRow = -1;
                int otherRow;
                for (otherRow = 0; otherRow < fs->base->RowBlockCount; otherRow++) {
                  int d, isSatisfied, k;
                  if (otherRow >= layerRowStart && otherRow < layerRowEnd) continue;
                  if (fs->base->ShiftMatrix[otherRow][targetCol] == -1) continue;
                  isSatisfied = 1;
                  for (k = 0; k < violatedCount; k++) {
                    if (violatedRows[k] == otherRow) { isSatisfied = 0; break; }
                  }
                  if (!isSatisfied) continue;
                  d = (fs->base->ShiftMatrix[otherRow][targetCol] - curShift + Z) % Z;
                  if (d == 0) continue;
                  if (refShift < 0 || d < refDelta ||
                      (d == refDelta && otherRow < refRow)) {
                    refShift = fs->base->ShiftMatrix[otherRow][targetCol];
                    refDelta = d;
                    refRow   = otherRow;
                  }
                }

                if (refDelta > 0) {
                  chosenDelta = refDelta;
                  FEEDBACK_LOG(fs->feedbackLogsEnabled,
                               "[sender_calc] row=%d anchor_col=%d(unchanged) target_col=%d "
                               "ref_row=%d ref_shift=%d delta=%d (sat-guided, cur_shift=%d->%d)\n",
                               row, anchorCol, targetCol, refRow, refShift, chosenDelta,
                               curShift, (curShift + chosenDelta) % Z);
                } else {
                  /* Fallback: local max->min within targetCol */
                  int bitStart = targetCol * Z;
                  int bitEnd   = bitStart + Z;
                  int bit, blockMaxE, blockMinE, blockMaxPos = 0, step;
                  if (bitEnd > fs->xLength) bitEnd = fs->xLength;
                  blockMaxE = blockMinE = fs->bitEnergy[bitStart];
                  for (bit = bitStart + 1; bit < bitEnd; bit++) {
                    if (fs->bitEnergy[bit] > blockMaxE) blockMaxE = fs->bitEnergy[bit];
                    if (fs->bitEnergy[bit] < blockMinE) blockMinE = fs->bitEnergy[bit];
                  }
                  for (bit = bitStart; bit < bitEnd; bit++) {
                    if (fs->bitEnergy[bit] == blockMaxE) {
                      blockMaxPos = bit - bitStart; break;
                    }
                  }
                  chosenDelta = 1;
                  for (step = 1; step <= Z; step++) {
                    int cPos = (blockMaxPos + step) % Z;
                    if (bitStart + cPos < bitEnd &&
                        fs->bitEnergy[bitStart + cPos] == blockMinE) {
                      chosenDelta = step; break;
                    }
                  }
                  FEEDBACK_LOG(fs->feedbackLogsEnabled,
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

              fs->feedbackRowLastCol1[row]  = targetCol;
              fs->feedbackRowLastCol2[row]  = -1;
              fs->feedbackRowNextShift[row] = chosenDelta;
              fs->feedbackShiftDeltas[row * fs->base->ColBlockCount + targetCol] =
                chosenDelta + 1;
            }
          } /* ANCHOR-MAX strategy */

          /* --- Step 5: Downlink log -------------------------------- */
          FEEDBACK_LOG(fs->feedbackLogsEnabled,
                       "\n[sender->receiver][downlink] LAYER MASKS rows=%d\n",
                       proposedCount);
          for (i = 0; i < proposedCount; i++) {
            if (proposedCol3[i] >= 0) {
              FEEDBACK_LOG(fs->feedbackLogsEnabled,
                           "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d col2=%d delta2=%+d col3=%d delta3=%+d\n",
                           proposedRows[i], proposedCol1[i], proposedDelta1[i],
                           proposedCol2[i], proposedDelta2[i],
                           proposedCol3[i], proposedDelta3[i]);
            } else if (proposedCol2[i] >= 0) {
              FEEDBACK_LOG(fs->feedbackLogsEnabled,
                           "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d col2=%d delta2=%+d\n",
                           proposedRows[i], proposedCol1[i], proposedDelta1[i],
                           proposedCol2[i], proposedDelta2[i]);
            } else {
              FEEDBACK_LOG(fs->feedbackLogsEnabled,
                           "[sender->receiver][downlink]   row=%d col1=%d delta1=%+d\n",
                           proposedRows[i], proposedCol1[i], proposedDelta1[i]);
            }
          }

          /* --- Step 6: Commit masks to auxMask arrays ------------- */
          {
            int newMasksAdded  = 0;
            fs->auxMaskCount = 0;

            for (i = 0; i < proposedCount; i++) {
              int row       = proposedRows[i];
              int sc1       = proposedCol1[i];
              int sc2       = proposedCol2[i];
              int sc3       = proposedCol3[i];
              int shiftCount = 1 + (sc2 >= 0 ? 1 : 0) + (sc3 >= 0 ? 1 : 0);

              if (fs->auxMaskCount < 128) {
                fs->auxMaskRows[fs->auxMaskCount]   = row;
                fs->auxMaskCol1[fs->auxMaskCount]   = sc1;
                fs->auxMaskDelta1[fs->auxMaskCount] = proposedDelta1[i];
                fs->auxMaskCol2[fs->auxMaskCount]   = sc2;
                fs->auxMaskDelta2[fs->auxMaskCount] = proposedDelta2[i];
                fs->auxMaskCol3[fs->auxMaskCount]   = sc3;
                fs->auxMaskDelta3[fs->auxMaskCount] = proposedDelta3[i];
                fs->auxMaskCount++;
                newMasksAdded++;
                totalShifts += shiftCount;
              }

              if (sc3 >= 0) {
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) col2=%d delta2=%+d (%d->%d) col3=%d delta3=%+d (%d->%d) score=%d\n",
                             row,
                             sc1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                             sc2, proposedDelta2[i], proposedOldShift2[i], proposedNewShift2[i],
                             sc3, proposedDelta3[i], proposedOldShift3[i], proposedNewShift3[i],
                             proposedScore[i]);
              } else if (sc2 >= 0) {
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) col2=%d delta2=%+d (%d->%d) score=%d\n",
                             row,
                             sc1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                             sc2, proposedDelta2[i], proposedOldShift2[i], proposedNewShift2[i],
                             proposedScore[i]);
              } else {
                FEEDBACK_LOG(fs->feedbackLogsEnabled,
                             "[receiver_calc] AUX row=%d col1=%d delta1=%+d (%d->%d) score=%d\n",
                             row,
                             sc1, proposedDelta1[i], proposedOldShift1[i], proposedNewShift1[i],
                             proposedScore[i]);
              }
            }

            fs->auxRoundsRemaining = fs->feedbackIntervalIters;
            if (newMasksAdded > 0) fs->frameShiftMatrixGenerations++;
            fs->frameAddedAuxEquations += newMasksAdded;

            FEEDBACK_LOG(fs->feedbackLogsEnabled,
                         "[receiver_calc] generation=%d aux_equation_set_size=%d active_for_next_%d_iterations\n\n",
                         fs->frameShiftMatrixGenerations, fs->auxMaskCount,
                         fs->auxRoundsRemaining);
          } /* mask commit */

        } /* layer row range */
      } /* layer selection */

      FEEDBACK_LOG(fs->feedbackLogsEnabled,
                   "[receiver_calc] total shifts applied: %d across %d row(s)\n",
                   totalShifts, targetRowCount);
      FEEDBACK_LOG(fs->feedbackLogsEnabled,
                   "[receiver_calc] continuing GDBF with original H + auxiliary equations...\n\n");

    } /* sender/receiver block */

    fs->feedbackRounds++;
    fs->lastFeedbackIter = fs->iter;
    FEEDBACK_LOG(fs->feedbackLogsEnabled,
                 "[sender->receiver][round] round=%d mask_sent, receiver_will_run_next_%d_iterations\n",
                 fs->feedbackRounds, fs->feedbackIntervalIters);

    StagnationStateReset(&fs->stagnationState);
    return FEEDBACK_CONTINUE_ITER;

  } /* violatedCount > 0 && originalShiftMatrix != NULL */

  return FEEDBACK_NONE;
}
