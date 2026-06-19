#include "stagnation_detection.h"

#define ENERGY_UNSET (-2147483647)

int StagnationStateInit(StagnationState *state, int historyLen)
{
  if (state == NULL || historyLen <= 0) {
    return 1;
  }

  state->history = (int *)malloc((size_t)historyLen * sizeof(int));
  if (state->history == NULL) {
    return 1;
  }

  state->energyHistoryLen = historyLen;
  StagnationStateReset(state);
  return 0;
}

void StagnationStateReset(StagnationState *state)
{
  int i;
  if (state == NULL || state->history == NULL) {
    return;
  }

  state->previousMaxEnergy = ENERGY_UNSET;
  state->stagnationCounter = 0;
  state->oscillationCounter = 0;
  state->historyCount = 0;

  for (i = 0; i < state->energyHistoryLen; i++) {
    state->history[i] = ENERGY_UNSET;
  }
}

void StagnationStateFree(StagnationState *state)
{
  if (state == NULL) {
    return;
  }

  free(state->history);
  state->history = NULL;
  state->energyHistoryLen = 0;
}

int StagnationStateUpdate(
  StagnationState *state,
  const StagnationConfig *config,
  int maxEnergy)
{
  int i;
  int inHistory = 0;

  if (state == NULL || config == NULL || state->history == NULL) {
    return 0;
  }

  if (maxEnergy == state->previousMaxEnergy) {
    state->stagnationCounter++;
  } else {
    state->stagnationCounter = 0;
  }
  state->previousMaxEnergy = maxEnergy;

  for (i = 0; i < state->historyCount; i++) {
    if (state->history[i] == maxEnergy) {
      inHistory = 1;
      break;
    }
  }

  /*
   * Treat oscillation as repeated revisits inside the recent max-energy
   * window.  The counter is a leaky accumulator:
   *   - A revisit (value already in history) adds one.
   *   - A miss (new value not seen recently) subtracts one, but never below
   *     zero.  This lets noisy cycles like 2,3,1,5,2,3,1 accumulate evidence
   *     across the interrupting values without a full reset, while truly novel
   *     sequences drain the counter back to zero.
   */
  if (inHistory) {
    state->oscillationCounter++;
  } else if (state->oscillationCounter > 0) {
    state->oscillationCounter--;
  }

  if (state->historyCount < state->energyHistoryLen) {
    state->history[state->historyCount++] = maxEnergy;
  } else {
    for (i = 0; i < state->energyHistoryLen - 1; i++) {
      state->history[i] = state->history[i + 1];
    }
    state->history[state->energyHistoryLen - 1] = maxEnergy;
  }

  return (state->stagnationCounter >= config->stagnationTrigger) ||
         (state->oscillationCounter >= config->oscillationTrigger);
}
