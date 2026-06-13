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

void AddBscNoiseFromIndexes(
    const int *codeword,
    int *receivedword,
    int length,
    const int *errorIndexes,
    int numErrors)
{
  int i, j;
  
  // Copy codeword to received word
  for (i = 0; i < length; i++) {
    receivedword[i] = codeword[i];
  }
  
  // Apply errors at specified indexes
  for (j = 0; j < numErrors; j++) {
    int idx = errorIndexes[j];
    if (idx >= 0 && idx < length) {
      receivedword[idx] = 1 - receivedword[idx];
    }
  }
}

void AddBscNoiseFromIndexesSplit(
    const int *codeword,
    int *receivedword,
    int n,   // length of x
    int m,   // length of p
    const int *errorIndexes,
    int numErrors)
{
  int i, j;
  
  // Copy x with errors applied
  for (i = 0; i < n; i++) {
    receivedword[i] = codeword[i];
  }
  
  // Apply errors only to x part (quantum channel)
  for (j = 0; j < numErrors; j++) {
    int idx = errorIndexes[j];
    if (idx >= 0 && idx < n) {
      receivedword[idx] = 1 - receivedword[idx];
    }
  }
  
  // copy parity bits perfectly (classical channel)
  for (i = 0; i < m; i++) {
    receivedword[n + i] = codeword[n + i];
  }
}
