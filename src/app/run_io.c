#include "app/run_io.h"

#include <direct.h>
#include <stdio.h>
#include <string.h>

void EnsureDirRecursive(const char *path)
{
  char tmp[512];
  char *p;

  if (path == NULL || path[0] == '\0') {
    return;
  }

  strncpy(tmp, path, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = '\0';

  for (p = tmp + 1; *p; p++) {
    if (*p == '/' || *p == '\\') {
      char c = *p;
      *p = '\0';
      _mkdir(tmp);
      *p = c;
    }
  }
  _mkdir(tmp);
}

int CopyFileBinary(const char *srcPath, const char *dstPath)
{
  FILE *src;
  FILE *dst;
  unsigned char buffer[8192];
  size_t n;

  if (srcPath == NULL || dstPath == NULL || *srcPath == '\0' || *dstPath == '\0') {
    return 1;
  }

  src = fopen(srcPath, "rb");
  if (src == NULL) {
    return 1;
  }

  dst = fopen(dstPath, "wb");
  if (dst == NULL) {
    fclose(src);
    return 1;
  }

  while ((n = fread(buffer, 1, sizeof(buffer), src)) > 0) {
    if (fwrite(buffer, 1, n, dst) != n) {
      fclose(src);
      fclose(dst);
      return 1;
    }
  }

  fclose(src);
  fclose(dst);
  return 0;
}

const char *DecoderTypeToName(DecoderType type)
{
  switch (type) {
    case DECODER_TYPE_FEEDBACK_SHIFT:
      return "feedback_shift";
    case DECODER_TYPE_ML_FEEDBACK:
      return "ml_feedback";
    case DECODER_TYPE_PGDBF:
      return "pgdbf";
    case DECODER_TYPE_ML:
      return "ml";
    case DECODER_TYPE_GDBF:
    default:
      return "baseline";
  }
}

const char *DecoderTypeToBanner(DecoderType type)
{
  switch (type) {
    case DECODER_TYPE_PGDBF:
      return "-------------------------La-P-GDBF--------------------------------------------------\n";
    case DECODER_TYPE_ML:
      return "-------------------------La-ML-GDBF-------------------------------------------------\n";
    case DECODER_TYPE_FEEDBACK_SHIFT:
      return "-------------------------La-FEEDBACK-SHIFT-------------------------------------------\n";
    case DECODER_TYPE_ML_FEEDBACK:
      return "-------------------------La-ML-FEEDBACK----------------------------------------------\n";
    case DECODER_TYPE_GDBF:
    default:
      return "-------------------------La-GDBF--------------------------------------------------\n";
  }
}

int SetupRunOutputPaths(
  const char *codeName,
  const char *decoderConfigPath,
  const DecoderConfig *decoderConfig,
  RunOutputPaths *outPaths,
  FILE **outFile)
{
  char resultBaseDir[512];
  const char *runType;

  if (codeName == NULL || codeName[0] == '\0' ||
      decoderConfigPath == NULL || decoderConfig == NULL ||
      outPaths == NULL || outFile == NULL) {
    return 1;
  }

  memset(outPaths, 0, sizeof(*outPaths));
  *outFile = NULL;

  if (decoderConfig->enableDatasetCollection) {
    runType = "collect";
  } else {
    runType = DecoderTypeToName(decoderConfig->decoderType);
  }

  snprintf(resultBaseDir, sizeof(resultBaseDir), "results/%s/%s", codeName, runType);
  if (decoderConfig->resultsExperimentName[0] != '\0') {
    snprintf(outPaths->resultDir, sizeof(outPaths->resultDir), "%s/%s", resultBaseDir, decoderConfig->resultsExperimentName);
  } else {
    strncpy(outPaths->resultDir, resultBaseDir, sizeof(outPaths->resultDir) - 1);
    outPaths->resultDir[sizeof(outPaths->resultDir) - 1] = '\0';
  }

  EnsureDirRecursive(outPaths->resultDir);

  snprintf(
    outPaths->usedDecoderConfigCopyPath,
    sizeof(outPaths->usedDecoderConfigCopyPath),
    "%s/used_decoder_config.cfg",
    outPaths->resultDir);
  if (CopyFileBinary(decoderConfigPath, outPaths->usedDecoderConfigCopyPath) != 0) {
    fprintf(
      stderr,
      "WARNING: could not copy decoder config '%s' to '%s'\n",
      decoderConfigPath,
      outPaths->usedDecoderConfigCopyPath);
  }

  snprintf(outPaths->resultFilePath, sizeof(outPaths->resultFilePath), "%s/simulation.res", outPaths->resultDir);
  snprintf(
    outPaths->feedbackShiftSummaryPath,
    sizeof(outPaths->feedbackShiftSummaryPath),
    "%s/feedback_shift_summary.res",
    outPaths->resultDir);
  snprintf(
    outPaths->feedbackShiftAlphaSummaryCsvPath,
    sizeof(outPaths->feedbackShiftAlphaSummaryCsvPath),
    "%s/feedback_shift_alpha_summary.csv",
    outPaths->resultDir);
  snprintf(
    outPaths->feedbackShiftGenHistCsvPath,
    sizeof(outPaths->feedbackShiftGenHistCsvPath),
    "%s/feedback_shift_gen_hist.csv",
    outPaths->resultDir);
  snprintf(outPaths->mlSummaryCsvPath, sizeof(outPaths->mlSummaryCsvPath), "%s/ml_outcome_summary.csv", outPaths->resultDir);
  snprintf(outPaths->mlDiagnosticsCsvPath, sizeof(outPaths->mlDiagnosticsCsvPath), "%s/ml_diagnostics.csv", outPaths->resultDir);
  snprintf(outPaths->mlFailureCsvPath, sizeof(outPaths->mlFailureCsvPath), "%s/ml_failure_cases.csv", outPaths->resultDir);

  *outFile = fopen(outPaths->resultFilePath, "a+");
  if (*outFile == NULL) {
    fprintf(stderr, "Output file %s error!.. Abort\n", outPaths->resultFilePath);
    return 1;
  }

  /* Keep the results stream in sync with run progress. */
  setvbuf(*outFile, NULL, _IOLBF, 0);
  return 0;
}
