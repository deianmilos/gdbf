#include <stdlib.h>

#include "channel.h"

void AddBscNoise(const int *codeword, int *receivedword, int length, float alpha)
{
  int i;
  for (i = 0; i < length; i++) {
    if (((float)rand() / (float)RAND_MAX) < alpha) {
      receivedword[i] = 1 - codeword[i];
    } else {
      receivedword[i] = codeword[i];
    }
  }
}

void AddBscNoiseSplit(
    const int *codeword,
    int *receivedword,
    int n,   // length of x
    int m,   // length of p
    float alpha)
{
  int i;

  // apply noise only on x (quantum channel)
  for (i = 0; i < n; i++) {
    if (((float)rand() / (float)RAND_MAX) < alpha) {
      receivedword[i] = 1 - codeword[i];
    } else {
      receivedword[i] = codeword[i];
    }
  }

  // copy parity bits perfectly (classical channel)
  for (i = 0; i < m; i++) {
    receivedword[n + i] = codeword[n + i];
  }
}
