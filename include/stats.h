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
} SimulationStats;

void ResetSimulationStats(SimulationStats *stats);
void UpdateSimulationStats(
  SimulationStats *stats,
  int frameBitErrors,
  int isCodeword,
  int usedIterations,
  int maxDecoderIterations);
void PrintStatsHeader(FILE *fout);
void PrintStatsLine(float alpha, const SimulationStats *stats, int codeLength, FILE *fout);

#endif
