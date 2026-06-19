#ifndef ARGS_AND_CONFIG_H
#define ARGS_AND_CONFIG_H

/*
 * args_and_config.h — Command-line argument and config file parsing.
 */

int ParseNamedArgs(
  int argc,
  char *argv[],
  int *nbMonteCarlo,
  int *maxDecoderIterations,
  char *codeName,
  int codeNameLen,
  float *alpha,
  int *nbFrames,
  float *alphaMax,
  float *alphaMin,
  float *alphaStep,
  char *decoderConfigPath,
  int decoderConfigPathLen,
  int *hasDecoderConfigPath);

int LoadCodeNameFromConfig(const char *filePath, char *codeName, int codeNameLen);

#endif /* ARGS_AND_CONFIG_H */
