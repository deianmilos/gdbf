#ifndef CANDIDATE_SELECTION_H
#define CANDIDATE_SELECTION_H

#include "common.h"

typedef enum {
  CANDIDATE_SELECTION_TOPK = 0,
  CANDIDATE_SELECTION_GRAPH = 1,
  CANDIDATE_SELECTION_MAX_ENERGY_CHECKS = 2
} CandidateSelectionType;

typedef struct {
  CandidateSelectionType type;
  int candidateCount;
} CandidateSelectionConfig;

typedef struct {
  const char *name;
  void (*select)(
    const BaseMatrixData *base,
    const int *decodedBits,
    const int *bitEnergy,
    const int *unsatCounts,
    int xLength,
    const CandidateSelectionConfig *config,
    int *candidateIdx,
    int *candidateCountOut);
} CandidateSelectionStrategy;

const CandidateSelectionStrategy *GetCandidateSelectionStrategy(CandidateSelectionType type);

#endif
