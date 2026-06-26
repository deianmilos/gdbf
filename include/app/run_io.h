#ifndef RUN_IO_H
#define RUN_IO_H

#include "decoder_framework.h"
#include <stdio.h>

typedef struct {
  char resultDir[512];
  char resultFilePath[512];
  char feedbackShiftSummaryPath[512];
  char feedbackShiftAlphaSummaryCsvPath[512];
  char feedbackShiftGenHistCsvPath[512];
  char usedDecoderConfigCopyPath[512];
  char mlSummaryCsvPath[512];
  char mlDiagnosticsCsvPath[512];
  char mlFailureCsvPath[512];
} RunOutputPaths;

void EnsureDirRecursive(const char *path);

int CopyFileBinary(const char *srcPath, const char *dstPath);

const char *DecoderTypeToName(DecoderType type);

const char *DecoderTypeToBanner(DecoderType type);

int SetupRunOutputPaths(
  const char *codeName,
  const char *decoderConfigPath,
  const DecoderConfig *decoderConfig,
  RunOutputPaths *outPaths,
  FILE **outFile);

#endif /* RUN_IO_H */
