#include "app/logging.h"
#include "decoder_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AUX_EQ_HIST_MAX 256

static void WriteFeedbackShiftDistributionSummary(
  FILE *stream,
  float alpha,
  long long framesNoFeedbackSuccess,
  long long framesFeedbackSuccess,
  long long framesNotManagedDecode,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow,
  double pNoFeedbackSuccess,
  double pFeedbackSuccess,
  double pNotManagedDecode)
{
  int eq;
  int printedAny = 0;

  if (stream == NULL) {
    return;
  }

  fprintf(stream,
          "[feedback_shift][alpha=%.5f] Frame outcome (%% of tested): no_feedback_success=%.2f%%, feedback_success=%.2f%%, not_managed_decode=%.2f%%\n",
          alpha,
          pNoFeedbackSuccess,
          pFeedbackSuccess,
          pNotManagedDecode);

  if (framesFeedbackSuccess > 0) {
    int lineCount = 0;
    fprintf(stream,
            "[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations (%% among feedback_success frames):\n",
            alpha);
    for (eq = 0; eq < AUX_EQ_HIST_MAX; eq++) {
      if (feedbackSuccessAuxEqHist[eq] > 0) {
        double pEq = (100.0 * (double)feedbackSuccessAuxEqHist[eq]) / (double)framesFeedbackSuccess;
        fprintf(stream,
                "  gen=%d: %.2f%% (%lld/%lld)",
                eq,
                pEq,
                feedbackSuccessAuxEqHist[eq],
                framesFeedbackSuccess);
        lineCount++;
        if (lineCount % 5 == 0) {
          fprintf(stream, "\n");
        } else {
          fprintf(stream, " | ");
        }
        printedAny = 1;
      }
    }
    if (printedAny && lineCount % 5 != 0) {
      fprintf(stream, "\n");
    }
    if (feedbackSuccessAuxEqOverflow > 0) {
      double pOverflow = (100.0 * (double)feedbackSuccessAuxEqOverflow) / (double)framesFeedbackSuccess;
      fprintf(stream,
              "  gen>=%d: %.2f%% (%lld/%lld)\n",
              AUX_EQ_HIST_MAX,
              pOverflow,
              feedbackSuccessAuxEqOverflow,
              framesFeedbackSuccess);
      printedAny = 1;
    }
    if (!printedAny) {
      fprintf(stream, "  none\n");
    }
  } else {
    fprintf(stream,
            "[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations: none (no successful feedback-assisted frames)\n",
            alpha);
  }

  fflush(stream);
}

static int FeedbackQuantileFromHist(
  const long long *hist,
  long long overflow,
  int p,
  long long total)
{
  long long threshold;
  long long cumulative = 0;
  int gen;

  if (hist == NULL || total <= 0) {
    return -1;
  }

  if (p < 0) {
    p = 0;
  }
  if (p > 100) {
    p = 100;
  }

  threshold = (total * (long long)p + 99LL) / 100LL;
  if (threshold <= 0) {
    threshold = 1;
  }

  for (gen = 0; gen < AUX_EQ_HIST_MAX; gen++) {
    cumulative += hist[gen];
    if (cumulative >= threshold) {
      return gen;
    }
  }

  cumulative += overflow;
  if (cumulative >= threshold && overflow > 0) {
    return AUX_EQ_HIST_MAX;
  }

  return -1;
}

void AppendFeedbackShiftAlphaSummaryCsv(
  const char *csvPath,
  float alpha,
  long long framesTested,
  long long framesNoFeedbackRound,
  long long framesEnteredFeedbackRound,
  long long framesNoFeedbackSuccess,
  long long framesFeedbackSuccess,
  long long framesNotManagedDecode,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow)
{
  FILE *f;
  int csvExists;
  int recreateCsv = 0;
  double noFeedbackPct;
  double feedbackPct;
  double notManagedPct;
  double genMean = -1.0;
  int genP50 = -1;
  int genP90 = -1;
  int genP95 = -1;
  int genMax = -1;
  long long i;
  long long feedbackTotal;
  long long weightedGenSum = 0;

  if (csvPath == NULL || feedbackSuccessAuxEqHist == NULL || framesTested <= 0) {
    return;
  }

  csvExists = (access(csvPath, 0) == 0);
  if (csvExists) {
    FILE *hdr = fopen(csvPath, "r");
    if (hdr != NULL) {
      char headerLine[1024];
      if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
        if (strstr(headerLine, "frames_no_feedback_round_count") == NULL ||
            strstr(headerLine, "frames_entered_feedback_round_count") == NULL ||
            strstr(headerLine, "pct_of_tested_no_feedback_success") == NULL ||
            strstr(headerLine, "pct_of_tested_feedback_success") == NULL ||
            strstr(headerLine, "pct_of_tested_not_managed_decode") == NULL ||
            strstr(headerLine, "gen_p95") == NULL ||
            strstr(headerLine, "gen_mean") == NULL ||
            strstr(headerLine, "gen_max") == NULL) {
          recreateCsv = 1;
        }
        if (strstr(headerLine, "no_feedback_success_count") != NULL) {
          recreateCsv = 1;
        }
      }
      fclose(hdr);
    }
  }

  f = fopen(csvPath, recreateCsv ? "w" : "a");
  if (f == NULL) {
    fprintf(stderr, "WARNING: cannot open feedback summary CSV: %s\n", csvPath);
    return;
  }

  if (!csvExists || recreateCsv) {
    fprintf(
      f,
      "alpha,frames_tested,frames_no_feedback_round_count,frames_entered_feedback_round_count,feedback_success_count,not_managed_decode_count,pct_of_tested_no_feedback_success,pct_of_tested_feedback_success,pct_of_tested_not_managed_decode,gen_p50,gen_p90,gen_p95,gen_mean,gen_max\n");
  }

  noFeedbackPct = (100.0 * (double)framesNoFeedbackSuccess) / (double)framesTested;
  feedbackPct = (100.0 * (double)framesFeedbackSuccess) / (double)framesTested;
  notManagedPct = (100.0 * (double)framesNotManagedDecode) / (double)framesTested;

  feedbackTotal = framesFeedbackSuccess;
  if (feedbackTotal > 0) {
    genP50 = FeedbackQuantileFromHist(feedbackSuccessAuxEqHist, feedbackSuccessAuxEqOverflow, 50, feedbackTotal);
    genP90 = FeedbackQuantileFromHist(feedbackSuccessAuxEqHist, feedbackSuccessAuxEqOverflow, 90, feedbackTotal);
    genP95 = FeedbackQuantileFromHist(feedbackSuccessAuxEqHist, feedbackSuccessAuxEqOverflow, 95, feedbackTotal);

    for (i = 0; i < AUX_EQ_HIST_MAX; i++) {
      long long c = feedbackSuccessAuxEqHist[i];
      if (c > 0) {
        weightedGenSum += c * i;
        genMax = (int)i;
      }
    }
    if (feedbackSuccessAuxEqOverflow > 0) {
      weightedGenSum += feedbackSuccessAuxEqOverflow * AUX_EQ_HIST_MAX;
      genMax = AUX_EQ_HIST_MAX;
    }

    genMean = (double)weightedGenSum / (double)feedbackTotal;
  }

  fprintf(
    f,
    "%.6f,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%d\n",
    alpha,
    framesTested,
    framesNoFeedbackRound,
    framesEnteredFeedbackRound,
    framesFeedbackSuccess,
    framesNotManagedDecode,
    noFeedbackPct,
    feedbackPct,
    notManagedPct,
    genP50,
    genP90,
    genP95,
    genMean,
    genMax);

  fclose(f);
}

void AppendFeedbackShiftGenHistogramCsv(
  const char *csvPath,
  float alpha,
  long long framesTested,
  long long framesFeedbackSuccess,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow)
{
  FILE *f;
  int csvExists;
  int recreateCsv = 0;
  int gen;
  long long cumulative = 0;

  if (csvPath == NULL || feedbackSuccessAuxEqHist == NULL || framesTested <= 0) {
    return;
  }

  csvExists = (access(csvPath, 0) == 0);
  if (csvExists) {
    FILE *hdr = fopen(csvPath, "r");
    if (hdr != NULL) {
      char headerLine[1024];
      if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
        if (strstr(headerLine, "pct_among_feedback_success") == NULL ||
            strstr(headerLine, "is_overflow") == NULL) {
          recreateCsv = 1;
        }
      }
      fclose(hdr);
    }
  }

  f = fopen(csvPath, recreateCsv ? "w" : "a");
  if (f == NULL) {
    fprintf(stderr, "WARNING: cannot open feedback histogram CSV: %s\n", csvPath);
    return;
  }

  if (!csvExists || recreateCsv) {
    fprintf(
      f,
      "alpha,frames_tested,frames_feedback_success,gen,count,pct_among_feedback_success,cumulative_pct,is_overflow\n");
  }

  for (gen = 0; gen < AUX_EQ_HIST_MAX; gen++) {
    long long c = feedbackSuccessAuxEqHist[gen];
    if (c <= 0) {
      continue;
    }

    cumulative += c;
    fprintf(
      f,
      "%.6f,%lld,%lld,%d,%lld,%.6f,%.6f,0\n",
      alpha,
      framesTested,
      framesFeedbackSuccess,
      gen,
      c,
      (framesFeedbackSuccess > 0) ? (100.0 * (double)c / (double)framesFeedbackSuccess) : 0.0,
      (framesFeedbackSuccess > 0) ? (100.0 * (double)cumulative / (double)framesFeedbackSuccess) : 0.0);
  }

  if (feedbackSuccessAuxEqOverflow > 0) {
    cumulative += feedbackSuccessAuxEqOverflow;
    fprintf(
      f,
      "%.6f,%lld,%lld,%d,%lld,%.6f,%.6f,1\n",
      alpha,
      framesTested,
      framesFeedbackSuccess,
      AUX_EQ_HIST_MAX,
      feedbackSuccessAuxEqOverflow,
      (framesFeedbackSuccess > 0) ? (100.0 * (double)feedbackSuccessAuxEqOverflow / (double)framesFeedbackSuccess) : 0.0,
      (framesFeedbackSuccess > 0) ? (100.0 * (double)cumulative / (double)framesFeedbackSuccess) : 0.0);
  }

  fclose(f);
}

void PrintUsage(const char *program)
{
  fprintf(stderr,
    "Usage (named): %s --frames <N> --max-iter <N> [--code <CodeName>] --alpha <A> "
    "[--nb-frames <N>] [--alpha-max <A>] [--alpha-min <A>] [--alpha-step <A>] [--decoder-config <path>]\n",
    program);
  fprintf(stderr,
    "Usage (positional): %s <NbMonteCarlo> <NbIter> <CodeName> <alpha> "
    "[NBframes [alpha_max [alpha_min [alpha_step]]]]\n",
    program);
}

void PrintFeedbackShiftDistributionSummary(
  FILE *fout,
  FILE *feedbackSummaryFile,
  float alpha,
  long long framesTested,
  long long framesNoFeedbackSuccess,
  long long framesFeedbackSuccess,
  long long framesNotManagedDecode,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow)
{
  int eq;
  double pNoFeedbackSuccess;
  double pFeedbackSuccess;
  double pNotManagedDecode;

  if (framesTested <= 0 || feedbackSuccessAuxEqHist == NULL) {
    return;
  }

  pNoFeedbackSuccess = (100.0 * (double)framesNoFeedbackSuccess) / (double)framesTested;
  pFeedbackSuccess = (100.0 * (double)framesFeedbackSuccess) / (double)framesTested;
  pNotManagedDecode = (100.0 * (double)framesNotManagedDecode) / (double)framesTested;

  if (fout != NULL) {
    WriteFeedbackShiftDistributionSummary(
      fout,
      alpha,
      framesNoFeedbackSuccess,
      framesFeedbackSuccess,
      framesNotManagedDecode,
      feedbackSuccessAuxEqHist,
      feedbackSuccessAuxEqOverflow,
      pNoFeedbackSuccess,
      pFeedbackSuccess,
      pNotManagedDecode);
  }

  if (feedbackSummaryFile != NULL && feedbackSummaryFile != fout) {
    WriteFeedbackShiftDistributionSummary(
      feedbackSummaryFile,
      alpha,
      framesNoFeedbackSuccess,
      framesFeedbackSuccess,
      framesNotManagedDecode,
      feedbackSuccessAuxEqHist,
      feedbackSuccessAuxEqOverflow,
      pNoFeedbackSuccess,
      pFeedbackSuccess,
      pNotManagedDecode);
  }
}

void BuildErrorVnIndexList(
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  char *out,
  int outSize)
{
  int i;
  int pos = 0;
  int wroteAny = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  for (i = 0; i < codeLength; i++) {
    if (decodedBits[i] != codeword[i]) {
      int n;
      if (wroteAny) {
        n = snprintf(out + pos, (size_t)(outSize - pos), ";%d", i);
      } else {
        n = snprintf(out + pos, (size_t)(outSize - pos), "%d", i);
      }

      if (n < 0 || n >= (outSize - pos)) {
        break;
      }

      pos += n;
      wroteAny = 1;
    }
  }
}

void BuildIndexListFromBuffer(
  const int *indices,
  int indexCount,
  char *out,
  int outSize)
{
  int i;
  int pos = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  if (indices == NULL || indexCount <= 0) {
    return;
  }

  for (i = 0; i < indexCount; i++) {
    int n;
    if (i > 0) {
      n = snprintf(out + pos, (size_t)(outSize - pos), ";%d", indices[i]);
    } else {
      n = snprintf(out + pos, (size_t)(outSize - pos), "%d", indices[i]);
    }

    if (n < 0 || n >= (outSize - pos)) {
      return;
    }

    pos += n;
  }
}

void BuildMlProposedErrorOverlapList(
  const int *mlProposedIndices,
  int mlProposedCount,
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  char *out,
  int outSize)
{
  int i;
  int pos = 0;
  int wroteAny = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  if (mlProposedIndices == NULL || mlProposedCount <= 0 || decodedBits == NULL || codeword == NULL) {
    return;
  }

  for (i = 0; i < mlProposedCount; i++) {
    int idx = mlProposedIndices[i];
    if (idx >= 0 && idx < codeLength && decodedBits[idx] != codeword[idx]) {
      int n;
      if (wroteAny) {
        n = snprintf(out + pos, (size_t)(outSize - pos), ";%d", idx);
      } else {
        n = snprintf(out + pos, (size_t)(outSize - pos), "%d", idx);
      }

      if (n < 0 || n >= (outSize - pos)) {
        return;
      }

      pos += n;
      wroteAny = 1;
    }
  }
}

void BuildErrorVnEnergyHistoryList(
  const int *decodedBits,
  const int *codeword,
  int codeLength,
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  char *out,
  int outSize)
{
  int vn;
  int pos = 0;
  int wroteAnyVn = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  if (lastBitEnergyHistory == NULL || lastBitEnergyHistoryCount <= 0) {
    return;
  }

  for (vn = 0; vn < codeLength; vn++) {
    if (decodedBits[vn] != codeword[vn]) {
      int h;
      int n;

      if (wroteAnyVn) {
        n = snprintf(out + pos, (size_t)(outSize - pos), ";%d:", vn);
      } else {
        n = snprintf(out + pos, (size_t)(outSize - pos), "%d:", vn);
      }
      if (n < 0 || n >= (outSize - pos)) {
        break;
      }
      pos += n;

      for (h = 0; h < lastBitEnergyHistoryCount; h++) {
        int e = lastBitEnergyHistory[h * codeLength + vn];
        if (h > 0) {
          n = snprintf(out + pos, (size_t)(outSize - pos), "|%d", e);
        } else {
          n = snprintf(out + pos, (size_t)(outSize - pos), "%d", e);
        }
        if (n < 0 || n >= (outSize - pos)) {
          return;
        }
        pos += n;
      }

      wroteAnyVn = 1;
    }
  }
}

void BuildGlobalMaxEnergyHistoryList(
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  int variableNodeCount,
  char *out,
  int outSize)
{
  int h;
  int pos = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  if (lastBitEnergyHistory == NULL || lastBitEnergyHistoryCount <= 0 || variableNodeCount <= 0) {
    return;
  }

  for (h = 0; h < lastBitEnergyHistoryCount; h++) {
    int i;
    int maxE = -2147483647;
    int n;

    for (i = 0; i < variableNodeCount; i++) {
      int e = lastBitEnergyHistory[h * variableNodeCount + i];
      if (e > maxE) {
        maxE = e;
      }
    }

    if (h > 0) {
      n = snprintf(out + pos, (size_t)(outSize - pos), "|%d", maxE);
    } else {
      n = snprintf(out + pos, (size_t)(outSize - pos), "%d", maxE);
    }

    if (n < 0 || n >= (outSize - pos)) {
      return;
    }
    pos += n;
  }
}

void BuildGlobalMaxVnIndexHistoryList(
  const int *lastBitEnergyHistory,
  int lastBitEnergyHistoryCount,
  int variableNodeCount,
  char *out,
  int outSize)
{
  int h;
  int pos = 0;

  if (out == NULL || outSize <= 0) {
    return;
  }

  out[0] = '\0';

  if (lastBitEnergyHistory == NULL || lastBitEnergyHistoryCount <= 0 || variableNodeCount <= 0) {
    return;
  }

  for (h = 0; h < lastBitEnergyHistoryCount; h++) {
    int i;
    int maxE = -2147483647;
    int wroteAny = 0;
    int n;

    for (i = 0; i < variableNodeCount; i++) {
      int e = lastBitEnergyHistory[h * variableNodeCount + i];
      if (e > maxE) {
        maxE = e;
      }
    }

    if (h > 0) {
      n = snprintf(out + pos, (size_t)(outSize - pos), "|");
      if (n < 0 || n >= (outSize - pos)) {
        return;
      }
      pos += n;
    }

    for (i = 0; i < variableNodeCount; i++) {
      int e = lastBitEnergyHistory[h * variableNodeCount + i];
      if (e == maxE) {
        if (wroteAny) {
          n = snprintf(out + pos, (size_t)(outSize - pos), "/%d", i);
        } else {
          n = snprintf(out + pos, (size_t)(outSize - pos), "%d", i);
        }

        if (n < 0 || n >= (outSize - pos)) {
          return;
        }
        pos += n;
        wroteAny = 1;
      }
    }
  }
}

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
  long long mlEscapes)
{
  FILE *f;
  int csvExists;
  int recreateCsv = 0;
  double effectivenessPct;

  if (csvPath == NULL || resultFile == NULL) {
    return;
  }

  csvExists = (access(csvPath, 0) == 0);
  if (csvExists) {
    FILE *hdr = fopen(csvPath, "r");
    if (hdr != NULL) {
      char headerLine[1024];
      if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
        if (strstr(headerLine, "frames_with_ml_invocation_decoded_clean") == NULL ||
            strstr(headerLine, "applied_ml_escape_frame_rate_pct") == NULL) {
          recreateCsv = 1;
        }
      }
      fclose(hdr);
    }
  }

  f = fopen(csvPath, recreateCsv ? "w" : "a");
  if (f == NULL) {
    fprintf(stderr, "WARNING: cannot open ML summary CSV: %s\n", csvPath);
    return;
  }

  if (!csvExists || recreateCsv) {
    fprintf(f,
      "result_file,nb_monte_carlo,max_iterations,nbframes_stop,"
      "alpha,alpha_max,alpha_min,alpha_step,"
      "frames_tested,frames_decoded_clean,frames_decoded_clean_without_ml_invocation,"
      "frames_with_ml_invocation,frames_with_applied_ml_escape,frames_with_ml_invocation_no_applied_escape,frames_with_ml_invocation_decoded_clean,"
      "applied_ml_escape_frame_rate_pct,"
      "total_stagnation_events,total_ml_predict_calls,total_applied_ml_escape_actions\n");
  }

  effectivenessPct = (framesMlNeeded > 0)
    ? (100.0 * (double)framesMlEscaped / (double)framesMlNeeded)
    : 0.0;

  fprintf(f,
    "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%lld,%lld,%lld\n",
    resultFile,
    nbMonteCarlo,
    maxDecoderIterations,
    nbFramesStop,
    alphaValue,
    alphaMax,
    alphaMin,
    alphaStep,
    framesTested,
    framesDecodedClean,
    framesBaselineOnlyDecoded,
    framesMlNeeded,
    framesMlEscaped,
    framesMlNoEscape,
    framesMlInvokedDecodedClean,
    effectivenessPct,
    stagnationEvents,
    mlCalls,
    mlEscapes);

  fclose(f);
}

void AppendMlDiagnosticsCsvOverall(
  const char *csvPath,
  const char *resultFile,
  int nbMonteCarlo,
  int maxDecoderIterations,
  int nbFramesStop,
  float alphaValue,
  float alphaMax,
  float alphaMin,
  float alphaStep,
  long long stagnationEvents,
  long long mlCounterfactualSkips,
  long long mlPredictCalls,
  long long mlEscapesApplied,
  long long bitsCorrectedByMl,
  long long framesTested,
  long long framesDecodedClean,
  long long framesBaselineOnlyDecoded,
  long long framesMlInvoked,
  long long appliedMlEscapeActionsInCleanFrames,
  long long failedFramesMlInvoked,
  long long failedFramesNoMlInvocation,
  long long framesWithSuccessfulMl,
  long long minMlCallsPerInvokedFrame,
  long long maxMlCallsPerInvokedFrame,
  long long sumMlCallsPerInvokedFrame,
  long long minCorrectedBitsPerSuccessfulMlFrame,
  long long maxCorrectedBitsPerSuccessfulMlFrame,
  long long sumCorrectedBitsPerSuccessfulMlFrame)
{
  FILE *f;
  int csvExists;
  int recreateCsv = 0;
  double appliedMlEscapeActionsInCleanFramesPct;
  double avgMlCallsPerInvokedFrame;
  double avgCorrectedBitsPerSuccessfulMlFrame;

  if (csvPath == NULL || resultFile == NULL) {
    return;
  }

  csvExists = (access(csvPath, 0) == 0);
  if (csvExists) {
    FILE *hdr = fopen(csvPath, "r");
    if (hdr != NULL) {
      char headerLine[2048];
      if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
        if (strstr(headerLine, "total_ml_counterfactual_skips") == NULL ||
            strstr(headerLine, "applied_ml_escape_actions_in_clean_frames_pct") == NULL) {
          recreateCsv = 1;
        }
      }
      fclose(hdr);
    }
  }

  f = fopen(csvPath, recreateCsv ? "w" : "a");
  if (f == NULL) {
    fprintf(stderr, "WARNING: cannot open ML diagnostics CSV: %s\n", csvPath);
    return;
  }

  if (!csvExists || recreateCsv) {
    fprintf(f,
      "result_file,nb_monte_carlo,max_iterations,nbframes_stop,"
      "alpha,alpha_max,alpha_min,alpha_step,"
      "total_stagnation_events,total_ml_counterfactual_skips,total_ml_predict_calls,total_applied_ml_escape_actions,total_bits_corrected_by_ml,"
      "frames_tested,frames_decoded_clean,frames_decoded_clean_without_ml_invocation,frames_with_ml_invocation,"
      "failed_frames_with_ml_invocation,failed_frames_without_ml_invocation,frames_with_applied_ml_escape,"
      "applied_ml_escape_actions_in_clean_frames,applied_ml_escape_actions_in_clean_frames_pct,"
      "min_ml_calls_per_ml_invoked_frame,max_ml_calls_per_ml_invoked_frame,avg_ml_calls_per_ml_invoked_frame,"
      "min_corrected_bits_per_frame_with_applied_ml_escape,max_corrected_bits_per_frame_with_applied_ml_escape,avg_corrected_bits_per_frame_with_applied_ml_escape\n");
  }

  appliedMlEscapeActionsInCleanFramesPct = (mlEscapesApplied > 0)
    ? (100.0 * (double)appliedMlEscapeActionsInCleanFrames / (double)mlEscapesApplied)
    : 0.0;

  avgMlCallsPerInvokedFrame = (framesMlInvoked > 0)
    ? ((double)sumMlCallsPerInvokedFrame / (double)framesMlInvoked)
    : -1.0;

  avgCorrectedBitsPerSuccessfulMlFrame = (framesWithSuccessfulMl > 0)
    ? ((double)sumCorrectedBitsPerSuccessfulMlFrame / (double)framesWithSuccessfulMl)
    : -1.0;

  fprintf(f,
    "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%lld,%lld,%.6f,%lld,%lld,%.6f\n",
    resultFile,
    nbMonteCarlo,
    maxDecoderIterations,
    nbFramesStop,
    alphaValue,
    alphaMax,
    alphaMin,
    alphaStep,
    stagnationEvents,
    mlCounterfactualSkips,
    mlPredictCalls,
    mlEscapesApplied,
    bitsCorrectedByMl,
    framesTested,
    framesDecodedClean,
    framesBaselineOnlyDecoded,
    framesMlInvoked,
    appliedMlEscapeActionsInCleanFrames,
    failedFramesMlInvoked,
    failedFramesNoMlInvocation,
    framesWithSuccessfulMl,
    appliedMlEscapeActionsInCleanFramesPct,
    (framesMlInvoked > 0) ? minMlCallsPerInvokedFrame : -1,
    (framesMlInvoked > 0) ? maxMlCallsPerInvokedFrame : -1,
    avgMlCallsPerInvokedFrame,
    (framesWithSuccessfulMl > 0) ? minCorrectedBitsPerSuccessfulMlFrame : -1,
    (framesWithSuccessfulMl > 0) ? maxCorrectedBitsPerSuccessfulMlFrame : -1,
    avgCorrectedBitsPerSuccessfulMlFrame);

  fclose(f);
}
