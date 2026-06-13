#ifndef CHANNEL_H
#define CHANNEL_H

void AddBscNoise(const int *codeword, int *receivedword, int length, float alpha);

void AddBscNoiseSplit(
    const int *codeword,
    int *receivedword,
    int n,
    int m,
    float alpha);

void AddBscNoiseFromIndexes(
    const int *codeword,
    int *receivedword,
    int length,
    const int *errorIndexes,
    int numErrors);

void AddBscNoiseFromIndexesSplit(
    const int *codeword,
    int *receivedword,
    int n,
    int m,
    const int *errorIndexes,
    int numErrors);

    
#endif
