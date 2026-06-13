#ifndef DECODER_PERTURB_H
#define DECODER_PERTURB_H

#include "decoder_framework.h"

int FindMaxEnergy(const int *bitEnergy, int xLength);

void FlipAtMaxEnergy(
  int *decodedBits,
  const int *bitEnergy,
  int xLength,
  DecoderType decoderType,
  double pgdbfFlipProbability);

void FlipAtMaxEnergyTrack(
  int *decodedBits,
  const int *bitEnergy,
  int xLength,
  DecoderType decoderType,
  double pgdbfFlipProbability,
  int *flipCounts);

#endif
