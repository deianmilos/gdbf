#ifndef STATS_H
#define STATS_H

#include <stdio.h>

typedef struct
{
  int nbtestedframes;
  int NiterMoy;
  int NiterMax;
  int Dmin;
  int NbTotalErrors;
  int NbBitError;
  int NbUnDetectedErrors;
  
  /* Successful frame iteration statistics */
  int minSuccessIter;
  int maxSuccessIter;
  long long sumSuccessIter;
  int successFrameCount;

  /* Failed frame bit error statistics */
  int minFailedBitErrors;
  int maxFailedBitErrors;
  long long sumFailedBitErrors;  /* Use long long to avoid overflow */
  int failedFrameCount;
  int failedZeroBitErrorAnomalyCount;

  /* FEEDBACK_SHIFT auxiliary-equation statistics (per frame) */
  int minAddedAuxEquations;
  int maxAddedAuxEquations;
  long long sumAddedAuxEquations;

  /* FEEDBACK_SHIFT sender/receiver unsuccessful rounds before syndrome-0 */
  int minUnsuccessfulRoundsToSyndrome0;
  int maxUnsuccessfulRoundsToSyndrome0;
  long long sumUnsuccessfulRoundsToSyndrome0;
  int minUnsuccessfulRoundsToSyndrome0Count;
  int maxUnsuccessfulRoundsToSyndrome0Count;
  int syndrome0FrameCount;

  /* FEEDBACK_SHIFT count of max-energy bits before each feedback request */
  int minMaxEnergyBitsBeforeFeedback;
  int maxMaxEnergyBitsBeforeFeedback;
  long long sumMaxEnergyBitsBeforeFeedback;
  long long countMaxEnergyBitsBeforeFeedback;

  /* ML inference calls per successful frame */
  int minMLInferencesPerFrame;
  int maxMLInferencesPerFrame;
  long long sumMLInferencesPerFrame;
  int mlInferenceFrameCount;
} SimulationStats;

void ResetSimulationStats(SimulationStats *stats);
void UpdateSimulationStats(
  SimulationStats *stats,
  int frameBitErrors,
  int isCodeword,
  int usedIterations,
  int maxDecoderIterations,
  int addedAuxEquations,
  int unsuccessfulRoundsToSyndrome0,
  int maxEnergyBitsBeforeFeedbackMin,
  int maxEnergyBitsBeforeFeedbackMax,
  long long maxEnergyBitsBeforeFeedbackSum,
  int maxEnergyBitsBeforeFeedbackCount,
  int mlInferencesThisFrame);
void PrintStatsHeader(FILE *fout, int showAuxEquationStats, int circulantSize);
void PrintStatsLine(
  float alpha,
  const SimulationStats *stats,
  int codeLength,
  FILE *fout,
  int showAuxEquationStats,
  int circulantSize);

#endif
