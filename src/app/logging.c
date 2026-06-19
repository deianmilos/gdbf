#include "app/logging.h"
#include "decoder_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AUX_EQ_HIST_MAX 256

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
  float alpha,
  long long framesTested,
  long long framesNoFeedbackSuccess,
  long long framesFeedbackSuccess,
  long long framesNotManagedDecode,
  const long long *feedbackSuccessAuxEqHist,
  long long feedbackSuccessAuxEqOverflow)
{
  int eq;
  int printedAny = 0;
  double pNoFeedbackSuccess;
  double pFeedbackSuccess;
  double pNotManagedDecode;

  if (framesTested <= 0 || feedbackSuccessAuxEqHist == NULL) {
    return;
  }

  pNoFeedbackSuccess = (100.0 * (double)framesNoFeedbackSuccess) / (double)framesTested;
  pFeedbackSuccess = (100.0 * (double)framesFeedbackSuccess) / (double)framesTested;
  pNotManagedDecode = (100.0 * (double)framesNotManagedDecode) / (double)framesTested;

  printf("[feedback_shift][alpha=%.5f] Frame outcome (%% of tested): no_feedback_success=%.2f%%, feedback_success=%.2f%%, not_managed_decode=%.2f%%\n",
         alpha,
         pNoFeedbackSuccess,
         pFeedbackSuccess,
         pNotManagedDecode);

  if (framesFeedbackSuccess > 0) {
    int lineCount = 0;
    printf("[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations (%% among feedback_success frames):\n",
           alpha);
    for (eq = 0; eq < AUX_EQ_HIST_MAX; eq++) {
      if (feedbackSuccessAuxEqHist[eq] > 0) {
        double pEq = (100.0 * (double)feedbackSuccessAuxEqHist[eq]) / (double)framesFeedbackSuccess;
        printf("  gen=%d: %.2f%% (%lld/%lld)",
               eq,
               pEq,
               feedbackSuccessAuxEqHist[eq],
               framesFeedbackSuccess);
        lineCount++;
        if (lineCount % 5 == 0) {
          printf("\n");
        } else {
          printf(" | ");
        }
        printedAny = 1;
      }
    }
    if (printedAny && lineCount % 5 != 0) {
      printf("\n");
    }
    if (feedbackSuccessAuxEqOverflow > 0) {
      double pOverflow = (100.0 * (double)feedbackSuccessAuxEqOverflow) / (double)framesFeedbackSuccess;
      printf("  gen>=%d: %.2f%% (%lld/%lld)\n",
             AUX_EQ_HIST_MAX,
             pOverflow,
             feedbackSuccessAuxEqOverflow,
             framesFeedbackSuccess);
      printedAny = 1;
    }
    if (!printedAny) {
      printf("  none\n");
    }
  } else {
    printf("[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations: none (no successful feedback-assisted frames)\n",
           alpha);
  }

  if (fout != NULL) {
    printedAny = 0;
    fprintf(fout,
            "[feedback_shift][alpha=%.5f] Frame outcome (%% of tested): no_feedback_success=%.2f%%, feedback_success=%.2f%%, not_managed_decode=%.2f%%\n",
            alpha,
            pNoFeedbackSuccess,
            pFeedbackSuccess,
            pNotManagedDecode);

    if (framesFeedbackSuccess > 0) {
      int lineCountFile = 0;
      fprintf(fout,
              "[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations (%% among feedback_success frames):\n",
              alpha);
      for (eq = 0; eq < AUX_EQ_HIST_MAX; eq++) {
        if (feedbackSuccessAuxEqHist[eq] > 0) {
          double pEq = (100.0 * (double)feedbackSuccessAuxEqHist[eq]) / (double)framesFeedbackSuccess;
          fprintf(fout,
                  "  gen=%d: %.2f%% (%lld/%lld)",
                  eq,
                  pEq,
                  feedbackSuccessAuxEqHist[eq],
                  framesFeedbackSuccess);
          lineCountFile++;
          if (lineCountFile % 5 == 0) {
            fprintf(fout, "\n");
          } else {
            fprintf(fout, " | ");
          }
          printedAny = 1;
        }
      }
      if (printedAny && lineCountFile % 5 != 0) {
        fprintf(fout, "\n");
      }
      if (feedbackSuccessAuxEqOverflow > 0) {
        double pOverflow = (100.0 * (double)feedbackSuccessAuxEqOverflow) / (double)framesFeedbackSuccess;
        fprintf(fout,
                "  gen>=%d: %.2f%% (%lld/%lld)\n",
                AUX_EQ_HIST_MAX,
                pOverflow,
                feedbackSuccessAuxEqOverflow,
                framesFeedbackSuccess);
        printedAny = 1;
      }
      if (!printedAny) {
        fprintf(fout, "  none\n");
      }
    } else {
      fprintf(fout,
              "[feedback_shift][alpha=%.5f] Feedback-success breakdown by ShiftMatrixGenerations: none (no successful feedback-assisted frames)\n",
              alpha);
    }
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
        if (strstr(headerLine, "frames_ml_invoked_decoded_clean") == NULL ||
            strstr(headerLine, "ml_escape_effectiveness_pct") == NULL) {
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
      "frames_tested,frames_decoded_clean,frames_baseline_only_decoded,"
      "frames_ml_needed,frames_ml_escaped,frames_ml_no_escape,frames_ml_invoked_decoded_clean,"
      "ml_escape_effectiveness_pct,"
      "stagnation_events,ml_calls,ml_escapes\n");
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
