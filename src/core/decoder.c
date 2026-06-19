#include "decoder.h"
#include "decoder_framework.h"

#include <stdio.h>

extern FILE *datasetFile;

/* Kept for backward-compatible diagnostics. */
long long dbg_stagnation_events = 0;
long long dbg_as_matched = 0;
long long dbg_ml_fired = 0;
long long dbg_ml_escaped = 0;

int DecodeFrameGdbf(
  const BaseMatrixData *base,
  const int *receivedword,
  const int *codeword,
  int codeLength,
  int maxDecoderIterations,
  int *decodedBits,
  int *bitEnergy,
  int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int *shiftedLayerVariableBuffer,
  int *isCodeword,
  int *usedIterations,
  int *frameBitErrors)
{
  DecoderConfig config;
  DecoderRuntimeStats stats;
  int rc;

  DecoderConfigInitDefaults(&config);
  DecoderConfigApplyEnv(&config);

  stats.stagnationEvents = 0;
  stats.modelInferenceCalls = 0;
  stats.mlEscapes = 0;
  stats.datasetRows = 0;

  rc = DecodeFrameWithConfig(
    base,
    receivedword,
    codeword,
    codeLength,
    maxDecoderIterations,
    decodedBits,
    bitEnergy,
    checkNodeSyndrome,
    layerVariableBuffer,
    shiftedLayerVariableBuffer,
    isCodeword,
    usedIterations,
    frameBitErrors,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    &config,
    &stats,
    datasetFile,
    -1,
    0.0f);

  dbg_stagnation_events += stats.stagnationEvents;
  dbg_as_matched += stats.stagnationEvents;
  dbg_ml_fired += stats.modelInferenceCalls;
  dbg_ml_escaped += stats.mlEscapes;

  return rc;
}
