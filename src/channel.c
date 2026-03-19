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
