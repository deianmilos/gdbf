#ifndef STAGNATION_DETECTION_H
#define STAGNATION_DETECTION_H

#include "common.h"

typedef struct {
  int energyHistoryLen;
  int stagnationTrigger;
  int oscillationTrigger;
} StagnationConfig;

typedef struct {
  int energyHistoryLen;
  int previousMaxEnergy;
  int stagnationCounter;
  int oscillationCounter;
  int historyCount;
  int *history;
} StagnationState;

int StagnationStateInit(StagnationState *state, int historyLen);
void StagnationStateReset(StagnationState *state);
void StagnationStateFree(StagnationState *state);

/* Returns 1 when stagnation/oscillation trigger is reached, else 0. */
int StagnationStateUpdate(
  StagnationState *state,
  const StagnationConfig *config,
  int maxEnergy);

#endif
