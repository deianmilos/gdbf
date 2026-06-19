#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>

/*
 * logging.h — Output formatting and logging functions.
 */

void PrintUsage(const char *program);

void PrintFeedbackShiftDistributionSummary(
  FILE *fout,
  float alpha,
  long long framesTested,
  long long framesNoFeedbackSuccess,
  long long framesFeedbackSuccess,
  long long framesNotManagedDecode,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow);

void BuildErrorVnIndexList(
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  char *out,
  int outSize);

void BuildIndexListFromBuffer(
  const int *indices,
  int indexCount,
  char *out,
  int outSize);

void BuildMlProposedErrorOverlapList(
  const int *mlProposedIndices,
  int mlProposedCount,
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  char *out,
  int outSize);

void BuildErrorVnEnergyHistoryList(
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  char *out,
  int outSize);

void BuildGlobalMaxEnergyHistoryList(
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  int variableNodeCount,
  char *out,
  int outSize);

void BuildGlobalMaxVnIndexHistoryList(
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  int variableNodeCount,
  char *out,
  int outSize);

void AppendMlOutcomeSummaryCsvPerAlpha(
  const char *csvPath,
  const char *resultFile,
  int nbMonteCarlo,
  int maxDecoderIterations,
  int nbFramesStop,
  float alphaValue,
  float alphaMax,
  float alphaMin,
  float alphaStep,
  long long framesTested,
  long long framesDecodedClean,
  long long framesBaselineOnlyDecoded,
  long long framesMlNeeded,
  long long framesMlEscaped,
  long long framesMlNoEscape,
  long long framesMlInvokedDecodedClean,
  long long stagnationEvents,
  long long mlCalls,
  long long mlEscapes);

#endif /* LOGGING_H */
