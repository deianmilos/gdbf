#include "candidate_selection.h"

static void SelectTopK(
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

static void SelectGraph(
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
  int *seeds;
  int seedCount = 0;
  int *pool;
  int poolCount = 0;
  int *seen;
  int K;
  (void)decodedBits;

  if (candidateIdx == NULL || candidateCountOut == NULL || bitEnergy == NULL ||
      unsatCounts == NULL || base == NULL || config == NULL) {
    return;
  }

  K = config->candidateCount;
  if (K > xLength) K = xLength;
  if (K < 0) K = 0;

  seeds = (int *)malloc((size_t)xLength * sizeof(int));
  pool = (int *)malloc((size_t)xLength * sizeof(int));
  seen = (int *)calloc((size_t)xLength, sizeof(int));
  if (seeds == NULL || pool == NULL || seen == NULL) {
    free(seeds);
    free(pool);
    free(seen);
    SelectTopK(base, decodedBits, bitEnergy, unsatCounts, xLength, config, candidateIdx, candidateCountOut);
    return;
  }

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] > maxEnergy) {
      maxEnergy = bitEnergy[i];
    }
  }

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] == maxEnergy) {
      seeds[seedCount++] = i;
    }
  }

  for (i = 0; i < seedCount; i++) {
    InsertIfNew(seeds[i], pool, &poolCount, xLength, seen);
    ExpandSeedNeighborhood(base, seeds[i], xLength, pool, &poolCount, seen);
  }

  if (poolCount == 0) {
    SelectTopK(base, decodedBits, bitEnergy, unsatCounts, xLength, config, candidateIdx, candidateCountOut);
    free(seeds);
    free(pool);
    free(seen);
    return;
  }

  SortByEnergyThenUnsat(bitEnergy, unsatCounts, pool, poolCount);

  for (i = 0; i < K && i < poolCount; i++) {
    candidateIdx[i] = pool[i];
  }
  *candidateCountOut = (K < poolCount) ? K : poolCount;

  free(seeds);
  free(pool);
  free(seen);
}

static const CandidateSelectionStrategy kTopKStrategy = {
  "topk",
  SelectTopK
};

static const CandidateSelectionStrategy kGraphStrategy = {
  "graph",
  SelectGraph
};

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
  int j;
  int K;
  int maxEnergy = -2147483647;
  int seedIdx = 0;
  int *pool;
  int poolCount = 0;
  int *seen;
  (void)decodedBits;
  (void)unsatCounts;

  if (candidateIdx == NULL || candidateCountOut == NULL || bitEnergy == NULL ||
      base == NULL || config == NULL) {
    return;
  }

  K = config->candidateCount;
  if (K > xLength) K = xLength;
  if (K < 0) K = 0;

  pool = (int *)malloc((size_t)xLength * sizeof(int));
  seen = (int *)calloc((size_t)xLength, sizeof(int));
  if (pool == NULL || seen == NULL) {
    free(pool);
    free(seen);
    SelectGraph(base, decodedBits, bitEnergy, unsatCounts, xLength, config, candidateIdx, candidateCountOut);
    return;
  }

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] > maxEnergy) {
      maxEnergy = bitEnergy[i];
      seedIdx = i;
    }
  }

  /* Use one max-energy bit as anchor, then collect VN neighborhood from its checks. */
  InsertIfNew(seedIdx, pool, &poolCount, xLength, seen);
  ExpandSeedNeighborhood(base, seedIdx, xLength, pool, &poolCount, seen);

  if (poolCount == 0) {
    SelectTopK(base, decodedBits, bitEnergy, unsatCounts, xLength, config, candidateIdx, candidateCountOut);
    free(pool);
    free(seen);
    return;
  }

  /* Keep anchor bit in slot 0 and sort the remaining neighborhood by energy/unsat. */
  if (poolCount > 1) {
    SortByEnergyThenUnsat(bitEnergy, unsatCounts, pool + 1, poolCount - 1);
  }

  /* Guarantee fixed-width candidate set when caller requests K bits. */
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
  *candidateCountOut = (K < poolCount) ? K : poolCount;

  free(pool);
  free(seen);
}

static const CandidateSelectionStrategy kMaxEnergyChecksStrategy = {
  "max_energy_checks",
  SelectMaxEnergyChecks
};

const CandidateSelectionStrategy *GetCandidateSelectionStrategy(CandidateSelectionType type)
{
  switch (type) {
    case CANDIDATE_SELECTION_MAX_ENERGY_CHECKS:
      return &kMaxEnergyChecksStrategy;
    case CANDIDATE_SELECTION_GRAPH:
      return &kGraphStrategy;
    case CANDIDATE_SELECTION_TOPK:
    default:
      return &kTopKStrategy;
  }
}
