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
  
  stats->minSuccessIter = 100000;
  stats->maxSuccessIter = 0;
  stats->sumSuccessIter = 0;
  stats->successFrameCount = 0;

  stats->minFailedBitErrors = 100000;
  stats->maxFailedBitErrors = 0;
  stats->sumFailedBitErrors = 0;
  stats->failedFrameCount = 0;

  stats->minAddedAuxEquations = 100000;
  stats->maxAddedAuxEquations = 0;
  stats->sumAddedAuxEquations = 0;

  stats->minUnsuccessfulRoundsToSyndrome0 = 100000;
  stats->maxUnsuccessfulRoundsToSyndrome0 = 0;
  stats->sumUnsuccessfulRoundsToSyndrome0 = 0;
  stats->syndrome0FrameCount = 0;

  stats->minMaxEnergyBitsBeforeFeedback = 100000;
  stats->maxMaxEnergyBitsBeforeFeedback = 0;
  stats->sumMaxEnergyBitsBeforeFeedback = 0;
  stats->countMaxEnergyBitsBeforeFeedback = 0;

  stats->minMLInferencesPerFrame = 100000;
  stats->maxMLInferencesPerFrame = 0;
  stats->sumMLInferencesPerFrame = 0;
  stats->mlInferenceFrameCount = 0;
}

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
  int mlInferencesThisFrame)
{
  if (stats == NULL) {
    return;
  }

  if (maxEnergyBitsBeforeFeedbackCount > 0) {
    stats->countMaxEnergyBitsBeforeFeedback += maxEnergyBitsBeforeFeedbackCount;
    stats->sumMaxEnergyBitsBeforeFeedback += maxEnergyBitsBeforeFeedbackSum;
    if (maxEnergyBitsBeforeFeedbackMin < stats->minMaxEnergyBitsBeforeFeedback) {
      stats->minMaxEnergyBitsBeforeFeedback = maxEnergyBitsBeforeFeedbackMin;
    }
    if (maxEnergyBitsBeforeFeedbackMax > stats->maxMaxEnergyBitsBeforeFeedback) {
      stats->maxMaxEnergyBitsBeforeFeedback = maxEnergyBitsBeforeFeedbackMax;
    }
  }

  stats->nbtestedframes++;
  stats->NbBitError += frameBitErrors;

  if (isCodeword && frameBitErrors == 0 && unsuccessfulRoundsToSyndrome0 >= 0) {
    if (addedAuxEquations < 0) {
      addedAuxEquations = 0;
    }
    stats->syndrome0FrameCount++;
    stats->sumAddedAuxEquations += addedAuxEquations;
    if (addedAuxEquations < stats->minAddedAuxEquations) {
      stats->minAddedAuxEquations = addedAuxEquations;
    }
    if (addedAuxEquations > stats->maxAddedAuxEquations) {
      stats->maxAddedAuxEquations = addedAuxEquations;
    }

    stats->sumUnsuccessfulRoundsToSyndrome0 += unsuccessfulRoundsToSyndrome0;
    if (unsuccessfulRoundsToSyndrome0 < stats->minUnsuccessfulRoundsToSyndrome0) {
      stats->minUnsuccessfulRoundsToSyndrome0 = unsuccessfulRoundsToSyndrome0;
    }
    if (unsuccessfulRoundsToSyndrome0 > stats->maxUnsuccessfulRoundsToSyndrome0) {
      stats->maxUnsuccessfulRoundsToSyndrome0 = unsuccessfulRoundsToSyndrome0;
    }
  }

  if (!isCodeword) {
    stats->NiterMoy += maxDecoderIterations;
    stats->NbTotalErrors++;
    
    /* Track failed frame bit error statistics */
    stats->failedFrameCount++;
    stats->sumFailedBitErrors += frameBitErrors;
    if (frameBitErrors < stats->minFailedBitErrors) {
      stats->minFailedBitErrors = frameBitErrors;
    }
    if (frameBitErrors > stats->maxFailedBitErrors) {
      stats->maxFailedBitErrors = frameBitErrors;
    }
  } else if (frameBitErrors == 0) {
    stats->NiterMax = max(stats->NiterMax, usedIterations);
    stats->NiterMoy += usedIterations;
    /* Track per-success-frame iteration stats */
    stats->successFrameCount++;
    stats->sumSuccessIter += usedIterations;
    if (usedIterations < stats->minSuccessIter) stats->minSuccessIter = usedIterations;

    /* Track ML inferences only for frames where ML was actually invoked. */
    if (mlInferencesThisFrame > 0) {
      stats->mlInferenceFrameCount++;
      stats->sumMLInferencesPerFrame += mlInferencesThisFrame;
      if (mlInferencesThisFrame < stats->minMLInferencesPerFrame) {
        stats->minMLInferencesPerFrame = mlInferencesThisFrame;
      }
      if (mlInferencesThisFrame > stats->maxMLInferencesPerFrame) {
        stats->maxMLInferencesPerFrame = mlInferencesThisFrame;
      }
    }
    if (usedIterations > stats->maxSuccessIter) stats->maxSuccessIter = usedIterations;
  } else {
    stats->NiterMax = max(stats->NiterMax, usedIterations);
    stats->NiterMoy += usedIterations;
    stats->NbTotalErrors++;
    stats->NbUnDetectedErrors++;
    stats->Dmin = min(stats->Dmin, frameBitErrors);
  }
}

void PrintStatsHeader(FILE *fout, int showAuxEquationStats, int circulantSize)
{
  (void)circulantSize;

  printf("alpha\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tSuccessIter(min/avg/max)\tFailedBits(min/avg/max)\tMLInferences(min/avg/max)");
  if (showAuxEquationStats) {
    printf("\tAddedAuxEq(min/avg/max)");
    printf("\tUnsuccSRRoundsToS0(min/avg/max)");
    printf("\tMaxEnergyBitsBeforeFB(min/avg/max)");
  }
  printf("\n");

  if (fout != NULL) {
    fprintf(fout, "alpha\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tSuccessIter(min/avg/max)\tFailedBits(min/avg/max)\tMLInferences(min/avg/max)");
    if (showAuxEquationStats) {
      fprintf(fout, "\tAddedAuxEq(min/avg/max)");
      fprintf(fout, "\tUnsuccSRRoundsToS0(min/avg/max)");
      fprintf(fout, "\tMaxEnergyBitsBeforeFB(min/avg/max)");
    }
    fprintf(fout, "\n");
  }
}

void PrintStatsLine(
  float alpha,
  const SimulationStats *stats,
  int codeLength,
  FILE *fout,
  int showAuxEquationStats,
  int circulantSize)
{
  (void)circulantSize;

  if (stats == NULL || stats->nbtestedframes <= 0 || codeLength <= 0) {
    return;
  }

  printf("%1.5f\t", alpha);
  printf("%10d (%1.15f)\t\t", stats->NbBitError, (float)stats->NbBitError / codeLength / stats->nbtestedframes);
  printf("%4d (%1.15f)\t\t", stats->NbTotalErrors, (float)stats->NbTotalErrors / stats->nbtestedframes);
  printf("%10d\t", stats->nbtestedframes);
  printf("%1.2f(%d)\t\t", (float)stats->NiterMoy / stats->nbtestedframes, stats->NiterMax);

  /* Print success frame iteration statistics */
  if (stats->successFrameCount > 0) {
    float avgSuccessIter = (float)stats->sumSuccessIter / stats->successFrameCount;
    printf("%d/%.1f/%d\t\t", stats->minSuccessIter, avgSuccessIter, stats->maxSuccessIter);
  } else {
    printf("-/-/-\t\t");
  }

  /* Print failed frame bit error statistics */
  if (stats->failedFrameCount > 0) {
    float avgFailedBits = (float)stats->sumFailedBitErrors / stats->failedFrameCount;
    printf("%d/%.1f/%d", stats->minFailedBitErrors, avgFailedBits, stats->maxFailedBitErrors);
  } else {
    printf("-/-/-");
  }

  /* Print ML inferences per frame statistics */
  printf("\t");
  if (stats->mlInferenceFrameCount > 0) {
    float avgMLInferences = (float)stats->sumMLInferencesPerFrame / stats->mlInferenceFrameCount;
    printf("%d/%.1f/%d", stats->minMLInferencesPerFrame, avgMLInferences, stats->maxMLInferencesPerFrame);
  } else {
    printf("-/-/-");
  }

  if (showAuxEquationStats) {
    if (stats->syndrome0FrameCount > 0) {
      float avgAuxEq = (float)stats->sumAddedAuxEquations / stats->syndrome0FrameCount;
      float avgUnsuccessfulRounds =
        (float)stats->sumUnsuccessfulRoundsToSyndrome0 / stats->syndrome0FrameCount;
      printf("\t%d/%.2f/%d", stats->minAddedAuxEquations, avgAuxEq, stats->maxAddedAuxEquations);
      printf("\t%d/%.2f/%d",
             stats->minUnsuccessfulRoundsToSyndrome0,
             avgUnsuccessfulRounds,
             stats->maxUnsuccessfulRoundsToSyndrome0);
    } else {
      printf("\t-/-/-");
      printf("\t-/-/-");
    }

    if (stats->countMaxEnergyBitsBeforeFeedback > 0) {
      float avgMaxEnergyBitsBeforeFeedback =
        (float)stats->sumMaxEnergyBitsBeforeFeedback / (float)stats->countMaxEnergyBitsBeforeFeedback;
      printf("\t%d/%.2f/%d",
             stats->minMaxEnergyBitsBeforeFeedback,
             avgMaxEnergyBitsBeforeFeedback,
             stats->maxMaxEnergyBitsBeforeFeedback);
    } else {
      printf("\t-/-/-");
    }
  }
  printf("\n");
  fflush(stdout);

  if (fout != NULL) {
    fprintf(fout, "%1.5f\t", alpha);
    fprintf(fout, "%10d (%1.15f)\t\t", stats->NbBitError, (float)stats->NbBitError / codeLength / stats->nbtestedframes);
    fprintf(fout, "%4d (%1.15f)\t\t", stats->NbTotalErrors, (float)stats->NbTotalErrors / stats->nbtestedframes);
    fprintf(fout, "%10d\t", stats->nbtestedframes);
    fprintf(fout, "%1.2f(%d)\t\t", (float)stats->NiterMoy / stats->nbtestedframes, stats->NiterMax);

    if (stats->successFrameCount > 0) {
      float avgSuccessIter = (float)stats->sumSuccessIter / stats->successFrameCount;
      fprintf(fout, "%d/%.1f/%d\t\t", stats->minSuccessIter, avgSuccessIter, stats->maxSuccessIter);
    } else {
      fprintf(fout, "-/-/-\t\t");
    }

    if (stats->failedFrameCount > 0) {
      float avgFailedBits = (float)stats->sumFailedBitErrors / stats->failedFrameCount;
      fprintf(fout, "%d/%.1f/%d", stats->minFailedBitErrors, avgFailedBits, stats->maxFailedBitErrors);
    } else {
      fprintf(fout, "-/-/-");
    }

    if (showAuxEquationStats) {
      if (stats->syndrome0FrameCount > 0) {
        float avgAuxEq = (float)stats->sumAddedAuxEquations / stats->syndrome0FrameCount;
        float avgUnsuccessfulRounds =
          (float)stats->sumUnsuccessfulRoundsToSyndrome0 / stats->syndrome0FrameCount;
        fprintf(fout, "\t%d/%.2f/%d", stats->minAddedAuxEquations, avgAuxEq, stats->maxAddedAuxEquations);
        fprintf(fout, "\t%d/%.2f/%d",
               stats->minUnsuccessfulRoundsToSyndrome0,
               avgUnsuccessfulRounds,
               stats->maxUnsuccessfulRoundsToSyndrome0);
      } else {
        fprintf(fout, "\t-/-/-");
        fprintf(fout, "\t-/-/-");
      }

      if (stats->countMaxEnergyBitsBeforeFeedback > 0) {
        float avgMaxEnergyBitsBeforeFeedback =
          (float)stats->sumMaxEnergyBitsBeforeFeedback / (float)stats->countMaxEnergyBitsBeforeFeedback;
        fprintf(fout, "\t%d/%.2f/%d",
               stats->minMaxEnergyBitsBeforeFeedback,
               avgMaxEnergyBitsBeforeFeedback,
               stats->maxMaxEnergyBitsBeforeFeedback);
      } else {
        fprintf(fout, "\t-/-/-");
      }
    }
    fprintf(fout, "\n");
  }
}
