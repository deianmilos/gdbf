#include "decoder_perturb.h"

int FindMaxEnergy(const int *bitEnergy, int xLength)
{
  int i;
  int maxEnergy = -2147483647;
  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] > maxEnergy) {
      maxEnergy = bitEnergy[i];
    }
  }
  return maxEnergy;
}

void FlipAtMaxEnergy(
  int *decodedBits,
  const int *bitEnergy,
  int xLength,
  DecoderType decoderType,
  double pgdbfFlipProbability)
{
  FlipAtMaxEnergyTrack(
    decodedBits,
    bitEnergy,
    xLength,
    decoderType,
    pgdbfFlipProbability,
    NULL);
}

void FlipAtMaxEnergyTrack(
  int *decodedBits,
  const int *bitEnergy,
  int xLength,
  DecoderType decoderType,
  double pgdbfFlipProbability,
  int *flipCounts)
{
  int i;
  int maxEnergy = FindMaxEnergy(bitEnergy, xLength);

  for (i = 0; i < xLength; i++) {
    if (bitEnergy[i] == maxEnergy) {
      if (decoderType == DECODER_TYPE_PGDBF) {
        double p = (double)rand() / (double)RAND_MAX;
        if (p <= pgdbfFlipProbability) {
          decodedBits[i] ^= 1;
          if (flipCounts != NULL) {
            flipCounts[i]++;
          }
        }
      } else {
        decodedBits[i] ^= 1;
        if (flipCounts != NULL) {
          flipCounts[i]++;
        }
      }
    }
  }
}
