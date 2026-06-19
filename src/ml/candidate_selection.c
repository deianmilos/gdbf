#include "candidate_selection.h"

static void FillTopByEnergy(
  const BaseMatrixData *base,
  const int *decodedBits,
  const int *bitEnergy,
  const int *unsatCounts,
  int xLength,
  const CandidateSelectionConfig *config,
  int *candidateIdx,
  int *candidateCountOut)
{
  int i;
  int j;
  int K;
  (void)base;
  (void)decodedBits;
  (void)unsatCounts;

  if (candidateIdx == NULL || candidateCountOut == NULL || bitEnergy == NULL || config == NULL) {
    return;
  }

  K = config->candidateCount;
  if (K > xLength) K = xLength;
  if (K < 0) K = 0;

  for (i = 0; i < K; i++) {
    candidateIdx[i] = i;
  }

  for (i = 0; i < K; i++) {
    for (j = i + 1; j < K; j++) {
      if (bitEnergy[candidateIdx[j]] > bitEnergy[candidateIdx[i]]) {
        int t = candidateIdx[i];
        candidateIdx[i] = candidateIdx[j];
        candidateIdx[j] = t;
      }
    }
  }

  for (i = K; i < xLength; i++) {
    if (K == 0) break;
    if (bitEnergy[i] > bitEnergy[candidateIdx[K - 1]]) {
      candidateIdx[K - 1] = i;
      for (j = K - 1; j > 0; j--) {
        if (bitEnergy[candidateIdx[j]] > bitEnergy[candidateIdx[j - 1]]) {
          int t = candidateIdx[j];
          candidateIdx[j] = candidateIdx[j - 1];
          candidateIdx[j - 1] = t;
        }
      }
    }
  }

  *candidateCountOut = K;
}

static void InsertIfNew(int idx, int *pool, int *poolCount, int poolCap, int *seen)
{
  if (idx < 0 || idx >= poolCap) {
    return;
  }
  if (seen[idx]) {
    return;
  }
  seen[idx] = 1;
  pool[*poolCount] = idx;
  (*poolCount)++;
}

static void ExpandSeedNeighborhood(
  const BaseMatrixData *base,
  int seedIdx,
  int xLength,
  int *pool,
  int *poolCount,
  int *seen)
{
  int Z = base->CirculantSize;
  int seedBlock = seedIdx / Z;
  int seedLocal = seedIdx % Z;
  int layer;

  for (layer = 0; layer < base->RowBlockCount; layer++) {
    int shiftSeed = base->ShiftMatrix[layer][seedBlock];
    int checkPos;
    int block;

    if (shiftSeed == -1) {
      continue;
    }

    checkPos = (seedLocal - shiftSeed + Z) % Z;

    for (block = 0; block < base->ColBlockCount; block++) {
      int shift = base->ShiftMatrix[layer][block];
      int local;
      int idx;
      if (shift == -1) {
        continue;
      }
      local = (checkPos + shift) % Z;
      idx = block * Z + local;
      if (idx < xLength) {
        InsertIfNew(idx, pool, poolCount, xLength, seen);
      }
    }
  }
}

static void SortByEnergyThenUnsat(const int *bitEnergy, const int *unsatCounts, int *arr, int n)
{
  int i;
  int j;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      int ai = arr[i];
      int aj = arr[j];
      if (bitEnergy[aj] > bitEnergy[ai] ||
          (bitEnergy[aj] == bitEnergy[ai] && unsatCounts[aj] > unsatCounts[ai])) {
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }
}

int CollectMaxEnergySeedIndices(
  const int *bitEnergy,
  int xLength,
  int *seedIdxOut,
  int maxSeedCount)
{
  int i;
  int maxEnergy = -2147483647;
  int seedCount = 0;

  if (bitEnergy == NULL || seedIdxOut == NULL || xLength <= 0 || maxSeedCount <= 0) {
    return 0;
  }

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] > maxEnergy) {
      maxEnergy = bitEnergy[i];
    }
  }

  for (i = 0; i < xLength && seedCount < maxSeedCount; i++) {
    if (bitEnergy[i] == maxEnergy) {
      seedIdxOut[seedCount++] = i;
    }
  }

  return seedCount;
}

int BuildMaxEnergyChecksCandidatesForSeed(
  const BaseMatrixData *base,
  const int *bitEnergy,
  const int *unsatCounts,
  int xLength,
  const CandidateSelectionConfig *config,
  int seedIdx,
  int *candidateIdx)
{
  int i;
  int j;
  int K;
  int *pool;
  int poolCount = 0;
  int *seen;

  if (candidateIdx == NULL || bitEnergy == NULL || unsatCounts == NULL ||
      base == NULL || config == NULL || xLength <= 0) {
    return 0;
  }

  K = config->candidateCount;
  if (K > xLength) K = xLength;
  if (K < 0) K = 0;
  if (K == 0) {
    return 0;
  }

  pool = (int *)malloc((size_t)xLength * sizeof(int));
  seen = (int *)calloc((size_t)xLength, sizeof(int));
  if (pool == NULL || seen == NULL) {
    free(pool);
    free(seen);
    return 0;
  }

  InsertIfNew(seedIdx, pool, &poolCount, xLength, seen);
  ExpandSeedNeighborhood(base, seedIdx, xLength, pool, &poolCount, seen);

  if (poolCount == 0) {
    free(pool);
    free(seen);
    return 0;
  }

  if (poolCount > 1) {
    SortByEnergyThenUnsat(bitEnergy, unsatCounts, pool + 1, poolCount - 1);
  }

  if (poolCount < K) {
    for (i = 0; i < xLength && poolCount < K; i++) {
      int best = -1;
      for (j = 0; j < xLength; j++) {
        if (seen[j]) {
          continue;
        }
        if (best == -1 || bitEnergy[j] > bitEnergy[best] ||
            (bitEnergy[j] == bitEnergy[best] && unsatCounts[j] > unsatCounts[best])) {
          best = j;
        }
      }
      if (best == -1) {
        break;
      }
      InsertIfNew(best, pool, &poolCount, xLength, seen);
    }
  }

  for (i = 0; i < K && i < poolCount; i++) {
    candidateIdx[i] = pool[i];
  }

  free(pool);
  free(seen);
  return (K < poolCount) ? K : poolCount;
}

static void SelectMaxEnergyChecks(
  const BaseMatrixData *base,
  const int *decodedBits,
  const int *bitEnergy,
  const int *unsatCounts,
  int xLength,
  const CandidateSelectionConfig *config,
  int *candidateIdx,
  int *candidateCountOut)
{
  int i;
  int maxEnergy = -2147483647;
  int seedIdx = 0;
  int selectedCount;
  (void)decodedBits;

  if (candidateIdx == NULL || candidateCountOut == NULL || bitEnergy == NULL ||
      unsatCounts == NULL || base == NULL || config == NULL) {
    return;
  }

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] > maxEnergy) {
      maxEnergy = bitEnergy[i];
      seedIdx = i;
    }
  }

  selectedCount = BuildMaxEnergyChecksCandidatesForSeed(
    base, bitEnergy, unsatCounts, xLength, config, seedIdx, candidateIdx);

  if (selectedCount <= 0) {
    FillTopByEnergy(base, decodedBits, bitEnergy, unsatCounts, xLength, config, candidateIdx, candidateCountOut);
    return;
  }

  *candidateCountOut = selectedCount;
}

static const CandidateSelectionStrategy kMaxEnergyChecksStrategy = {
  "max_energy_checks",
  SelectMaxEnergyChecks
};

const CandidateSelectionStrategy *GetCandidateSelectionStrategy(CandidateSelectionType type)
{
  (void)type;
  return &kMaxEnergyChecksStrategy;
}
