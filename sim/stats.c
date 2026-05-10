#include "stats.h"

#include "common.h"

void ResetSimulationStats(SimulationStats *stats)
{
  if (stats == NULL) {
    return;
  }

  stats->nbtestedframes = 0;
  stats->NiterMoy = 0;
  stats->NiterMax = 0;
  stats->Dmin = 100000;
  stats->NbTotalErrors = 0;
  stats->NbBitError = 0;
  stats->NbUnDetectedErrors = 0;
}

void UpdateSimulationStats(
  SimulationStats *stats,
  int frameBitErrors,
  int isCodeword,
  int usedIterations,
  int maxDecoderIterations)
{
  if (stats == NULL) {
    return;
  }

  stats->nbtestedframes++;
  stats->NbBitError += frameBitErrors;

  if (!isCodeword) {
    stats->NiterMoy += maxDecoderIterations;
    stats->NbTotalErrors++;
  } else if (frameBitErrors == 0) {
    stats->NiterMax = max(stats->NiterMax, usedIterations);
    stats->NiterMoy += usedIterations;
  } else {
    stats->NiterMax = max(stats->NiterMax, usedIterations);
    stats->NiterMoy += usedIterations;
    stats->NbTotalErrors++;
    stats->NbUnDetectedErrors++;
    stats->Dmin = min(stats->Dmin, frameBitErrors);
  }
}

void PrintStatsHeader(FILE *fout)
{
  printf("alpha\t\t\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");
  if (fout != NULL) {
    fprintf(fout, "alpha\t\t\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");
  }
}

void PrintStatsLine(float alpha, const SimulationStats *stats, int codeLength, FILE *fout)
{
  if (stats == NULL || stats->nbtestedframes <= 0 || codeLength <= 0) {
    return;
  }

  printf("%1.5f\t\t", alpha);
  printf("%10d (%1.15f)\t\t", stats->NbBitError, (float)stats->NbBitError / codeLength / stats->nbtestedframes);
  printf("%4d (%1.15f)\t\t", stats->NbTotalErrors, (float)stats->NbTotalErrors / stats->nbtestedframes);
  printf("%10d\t", stats->nbtestedframes);
  printf("%1.2f(%d)\t\t", (float)stats->NiterMoy / stats->nbtestedframes, stats->NiterMax);
  printf("%d(%d)\n", stats->NbUnDetectedErrors, stats->Dmin);
  fflush(stdout);

  if (fout != NULL) {
    fprintf(fout, "%1.5f\t\t", alpha);
    fprintf(fout, "%10d (%1.15f)\t\t", stats->NbBitError, (float)stats->NbBitError / codeLength / stats->nbtestedframes);
    fprintf(fout, "%4d (%1.15f)\t\t", stats->NbTotalErrors, (float)stats->NbTotalErrors / stats->nbtestedframes);
    fprintf(fout, "%10d\t", stats->nbtestedframes);
    fprintf(fout, "%1.2f(%d)\t\t", (float)stats->NiterMoy / stats->nbtestedframes, stats->NiterMax);
    fprintf(fout, "%d(%d)\n", stats->NbUnDetectedErrors, stats->Dmin);
    fflush(fout);
  }
}
