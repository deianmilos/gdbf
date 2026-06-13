#include "common.h"
#include "matrix_io.h"
#include "encoding.h"
#include "channel.h"
#include "decoder.h"
#include "decoder_framework.h"
#include "stats.h"
#include <io.h>
#include <direct.h>

FILE *datasetFile = NULL;

#define AUX_EQ_HIST_MAX 256

static void PrintFeedbackShiftDistributionSummary(
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

static void PrintUsage(const char *program)
{
  fprintf(stderr,
    "Usage (named): %s --frames <N> --max-iter <N> [--code <CodeName>] --alpha <A> "
    "[--nb-frames <N>] [--alpha-max <A>] [--alpha-min <A>] [--alpha-step <A>] [--decoder-config <path>] [--error-indexes <path>]\n"
    "  Note: --alpha is optional when --error-indexes is used (deterministic mode).\n",
    program);
  fprintf(stderr,
    "Usage (positional): %s <NbMonteCarlo> <NbIter> <CodeName> <alpha> "
    "[NBframes [alpha_max [alpha_min [alpha_step]]]]\n",
    program);
}

// Load error indexes from file (one index per line)
static int LoadErrorIndexesFromFile(const char *filepath, int **outIndexes, int *outCount)
{
  FILE *f = fopen(filepath, "r");
  if (f == NULL) {
    fprintf(stderr, "Error: Could not open error indexes file: %s\n", filepath);
    return 1;
  }

  // First pass: count lines
  int count = 0;
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '\n') {
      count++;
    }
  }
  // Handle case where last line doesn't have newline
  fseek(f, -1, SEEK_END);
  if ((c = fgetc(f)) != '\n' && ftell(f) > 0) {
    count++;
  }

  if (count == 0) {
    fprintf(stderr, "Warning: Error indexes file is empty: %s\n", filepath);
    fclose(f);
    *outIndexes = NULL;
    *outCount = 0;
    return 0;
  }

  // Allocate array
  *outIndexes = (int *)malloc(count * sizeof(int));
  if (*outIndexes == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for error indexes\n");
    fclose(f);
    return 1;
  }

  // Second pass: read indexes
  rewind(f);
  int idx = 0;
  int lineNum = 0;
  char line[64];
  while (fgets(line, sizeof(line), f) != NULL && idx < count) {
    lineNum++;
    int index = atoi(line);
    if (index < 0) {
      fprintf(stderr, "Warning: Negative index at line %d: %d (skipping)\n", lineNum, index);
      continue;
    }
    (*outIndexes)[idx++] = index;
  }

  fclose(f);
  *outCount = idx;
  return 0;
}

static int ParseNamedArgs(
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
  int *hasDecoderConfigPath,
  char *errorIndexesPath,
  int errorIndexesPathLen,
  int *hasErrorIndexesPath)
{
  int i;
  int hasFrames = 0;
  int hasMaxIter = 0;
  int hasCode = 0;
  int hasAlpha = 0;
  int hasAlphaMax = 0;
  int hasAlphaMin = 0;
  int hasAlphaStep = 0;

  *nbFrames = 0;
  *hasDecoderConfigPath = 0;
  *hasErrorIndexesPath = 0;

  for (i = 1; i < argc; i++) {
    const char *arg = argv[i];

    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      return 1;
    }

    if (strncmp(arg, "--", 2) != 0) {
      fprintf(stderr, "Unknown positional token in named mode: %s\n", arg);
      return 1;
    }

    if (i + 1 >= argc) {
      fprintf(stderr, "Missing value for option: %s\n", arg);
      return 1;
    }

    i++;
    if (strcmp(arg, "--frames") == 0) {
      *nbMonteCarlo = atoi(argv[i]);
      hasFrames = 1;
    } else if (strcmp(arg, "--max-iter") == 0) {
      *maxDecoderIterations = atoi(argv[i]);
      hasMaxIter = 1;
    } else if (strcmp(arg, "--code") == 0) {
      strncpy(codeName, argv[i], codeNameLen - 1);
      codeName[codeNameLen - 1] = '\0';
      hasCode = 1;
    } else if (strcmp(arg, "--alpha") == 0) {
      *alpha = (float)atof(argv[i]);
      hasAlpha = 1;
    } else if (strcmp(arg, "--nb-frames") == 0) {
      *nbFrames = atoi(argv[i]);
    } else if (strcmp(arg, "--alpha-max") == 0) {
      *alphaMax = (float)atof(argv[i]);
      hasAlphaMax = 1;
    } else if (strcmp(arg, "--alpha-min") == 0) {
      *alphaMin = (float)atof(argv[i]);
      hasAlphaMin = 1;
    } else if (strcmp(arg, "--alpha-step") == 0) {
      *alphaStep = (float)atof(argv[i]);
      hasAlphaStep = 1;
    } else if (strcmp(arg, "--decoder-config") == 0) {
      strncpy(decoderConfigPath, argv[i], decoderConfigPathLen - 1);
      decoderConfigPath[decoderConfigPathLen - 1] = '\0';
      *hasDecoderConfigPath = 1;
    } else if (strcmp(arg, "--error-indexes") == 0) {
      strncpy(errorIndexesPath, argv[i], errorIndexesPathLen - 1);
      errorIndexesPath[errorIndexesPathLen - 1] = '\0';
      *hasErrorIndexesPath = 1;
    } else {
      fprintf(stderr, "Unknown option: %s\n", arg);
      return 1;
    }
  }

  if (!hasFrames || !hasMaxIter) {
    fprintf(stderr, "Missing required options in named mode.\n");
    PrintUsage(argv[0]);
    return 1;
  }

  // --alpha is required only when error indexes are NOT provided via CLI
  // (if provided via config file, that is checked later in main)
  if (!hasAlpha && !(*hasErrorIndexesPath)) {
    fprintf(stderr, "Missing required option: --alpha (only optional when --error-indexes is used)\n");
    PrintUsage(argv[0]);
    return 1;
  }

  if (!hasAlpha) {
    *alpha = 0.0f;
  }

  if (!hasCode) {
    codeName[0] = '\0';
  }

  if (!hasAlphaMax) *alphaMax = *alpha;
  if (!hasAlphaMin) *alphaMin = *alpha - 1.0f;
  if (!hasAlphaStep) *alphaStep = 1.0f;

  return 0;
}

static char *TrimLeftLocal(char *s)
{
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
    s++;
  }
  return s;
}

static void TrimRightLocal(char *s)
{
  int n = (int)strlen(s);
  while (n > 0) {
    char c = s[n - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      s[n - 1] = '\0';
      n--;
    } else {
      break;
    }
  }
}

static int LoadCodeNameFromConfig(const char *filePath, char *codeName, int codeNameLen)
{
  FILE *f;
  char line[1024];

  if (filePath == NULL || codeName == NULL || codeNameLen <= 1) {
    return 1;
  }

  f = fopen(filePath, "r");
  if (f == NULL) {
    return 1;
  }

  while (fgets(line, sizeof(line), f) != NULL) {
    char *p = TrimLeftLocal(line);
    char *eq;
    char *k;
    char *v;

    TrimRightLocal(p);
    if (*p == '\0' || *p == '#' || *p == ';') {
      continue;
    }

    eq = strchr(p, '=');
    if (eq == NULL) {
      continue;
    }

    *eq = '\0';
    k = TrimLeftLocal(p);
    TrimRightLocal(k);
    v = TrimLeftLocal(eq + 1);
    TrimRightLocal(v);

    if (strcmp(k, "code") == 0 && *v != '\0') {
      strncpy(codeName, v, codeNameLen - 1);
      codeName[codeNameLen - 1] = '\0';
      fclose(f);
      return 0;
    }
  }

  fclose(f);
  return 1;
}

/* Create all intermediate directories in path (POSIX-style or Windows). */
static void EnsureDir(const char *path)
{
  char tmp[512];
  char *p;

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

static const char *DecoderTypeToName(DecoderType type)
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

static void BuildErrorVnIndexList(
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

static void BuildIndexListFromBuffer(
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

static void BuildMlProposedErrorOverlapList(
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

static void BuildErrorVnEnergyHistoryList(
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

static void BuildGlobalMaxEnergyHistoryList(
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

static void BuildGlobalMaxVnIndexHistoryList(
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

static void AppendMlOutcomeSummaryCsvPerAlpha(
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

int main(int argc, char *argv[])
{
  SparseMatrixData matrix;
  BaseMatrixData base;
  EncodingTransformData encoding;

  FILE *fout = NULL;
  char codeName[256];
  char matrixFilePath[512];
  char baseMatrixPrefix[512];
  char resultDir[512];
  char resultFileName[512];
  char mlSummaryCsvPath[512];
  char mlFailureCsvPath[512];
  FILE *mlFailureFile = NULL;
  const char *runType;
  DecoderConfig decoderConfig;
  DecoderRuntimeStats decoderRuntimeStats;
  char decoderConfigPath[512] = "configs/decoder/default.cfg";
  int hasDecoderConfigPath = 0;
  char errorIndexesPath[512] = "";
  int hasErrorIndexesPath = 0;

  int NbMonteCarlo;
  int maxDecoderIterations;
  int NBframes;
  int nb;

  float alpha;
  float alpha_max;
  float alpha_min;
  float alpha_step;

  int *codeword = NULL;
  int *receivedword = NULL;
  int *workVector = NULL;
  int *decodedBits = NULL;
  int *bitEnergy = NULL;
  int *lastBitEnergyHistory = NULL;
  int *mlProposedIndices = NULL;
  int *checkNodeSyndrome = NULL;
  int *layerVariableBuffer = NULL;
  int *shiftedLayerVariableBuffer = NULL;

  long long overallFramesTested = 0;
  long long overallFramesDecodedClean = 0;
  long long overallFramesBaselineOnlyDecoded = 0;
  long long overallFramesMlNeeded = 0;
  long long overallFramesMlEscaped = 0;
  long long overallFramesMlNoEscape = 0;
  long long overallFramesMlInvokedDecodedClean = 0;
  long long overallFramesWithAppliedEscape = 0;
  long long overallFramesWithAppliedEscapeDecodedClean = 0;
  long long overallEscapeActionsInCleanFrames = 0;
  long long overallFailedFramesMlInvoked = 0;
  long long overallFailedFramesNoMlInvocation = 0;
  long long overallFramesWithMlInvocation = 0;
  long long overallMlCallsPerInvokedFrameSum = 0;
  long long overallMlCallsPerInvokedFrameMin = -1;
  long long overallMlCallsPerInvokedFrameMax = 0;
  long long overallFramesWithSuccessfulMl = 0;
  long long overallCorrectedBitsPerSuccessfulFrameSum = 0;
  long long overallCorrectedBitsPerSuccessfulFrameMin = -1;
  long long overallCorrectedBitsPerSuccessfulFrameMax = 0;

  if (argc > 1 && strncmp(argv[1], "--", 2) == 0) {
    if (ParseNamedArgs(
          argc,
          argv,
          &NbMonteCarlo,
          &maxDecoderIterations,
          codeName,
          (int)sizeof(codeName),
          &alpha,
          &NBframes,
          &alpha_max,
          &alpha_min,
          &alpha_step,
          decoderConfigPath,
          (int)sizeof(decoderConfigPath),
          &hasDecoderConfigPath,
          errorIndexesPath,
          (int)sizeof(errorIndexesPath),
          &hasErrorIndexesPath) != 0) {
      return 1;
    }
  } else {
    if (argc < 5) {
      PrintUsage(argv[0]);
      return 1;
    }

    NbMonteCarlo = atoi(argv[1]);
    maxDecoderIterations = atoi(argv[2]);
    strncpy(codeName, argv[3], sizeof(codeName) - 1);
    codeName[sizeof(codeName) - 1] = '\0';
    alpha = (float)atof(argv[4]);
    NBframes = (argc > 5) ? atoi(argv[5]) : 0;
    alpha_max = (argc > 6) ? (float)atof(argv[6]) : alpha;
    alpha_min = (argc > 7) ? (float)atof(argv[7]) : (alpha - 1.0f);
    alpha_step = (argc > 8) ? (float)atof(argv[8]) : 1.0f;
  }

  DecoderConfigInitDefaults(&decoderConfig);
  if (access(decoderConfigPath, 0) == 0) {
    if (DecoderConfigLoadFromFile(&decoderConfig, decoderConfigPath) != 0) {
      fprintf(stderr, "Failed to parse decoder config file: %s\n", decoderConfigPath);
      return 1;
    }
  } else if (hasDecoderConfigPath) {
    fprintf(stderr, "Decoder config file not found: %s\n", decoderConfigPath);
    return 1;
  }

  if (codeName[0] == '\0') {
    if (LoadCodeNameFromConfig(decoderConfigPath, codeName, (int)sizeof(codeName)) != 0) {
      fprintf(stderr, "Missing code name. Pass --code or set code=<CodeName> in %s\n", decoderConfigPath);
      return 1;
    }
  }

  DecoderConfigApplyEnv(&decoderConfig);
  memset(&decoderRuntimeStats, 0, sizeof(decoderRuntimeStats));

  // Use error indexes path from config if not provided via command-line
  if (!hasErrorIndexesPath && decoderConfig.errorIndexesPath[0] != '\0') {
    strncpy(errorIndexesPath, decoderConfig.errorIndexesPath, sizeof(errorIndexesPath) - 1);
    errorIndexesPath[sizeof(errorIndexesPath) - 1] = '\0';
    hasErrorIndexesPath = 1;
  }

  snprintf(matrixFilePath, sizeof(matrixFilePath), "codes/%s/%s_Dform", codeName, codeName);
  snprintf(baseMatrixPrefix, sizeof(baseMatrixPrefix), "codes/%s/%s_Base", codeName, codeName);

  if (NbMonteCarlo <= 0) {
    fprintf(stderr, "NbMonteCarlo must be > 0\n");
    return 1;
  }
  if (maxDecoderIterations <= 0) {
    fprintf(stderr, "NbIter must be > 0\n");
    return 1;
  }
  if (alpha < 0.0f || alpha > 1.0f) {
    // Only validate alpha when not using deterministic error indexes
    if (!hasErrorIndexesPath) {
      fprintf(stderr, "alpha must be in [0, 1]\n");
      return 1;
    }
  }
  if (alpha_step <= 0.0f) {
    fprintf(stderr, "alpha_step must be > 0\n");
    return 1;
  }

  if (LoadSparseMatrixData(matrixFilePath, &matrix) != 0) {
    return 1;
  }

  if (LoadBaseMatrixData(baseMatrixPrefix, &base) != 0) {
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  if (BuildEncodingTransformFromSparseMatrix(&matrix, &encoding) != 0) {
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  printf("Matrix Loaded\n");
  printf("Graph Built\n");
  printf("Base Matrix Loaded\n");
  printf("Encoding Transform Built\n");
  printf("BaseRows=%d, BaseCols=%d, Circulant=%d\n", base.RowBlockCount, base.ColBlockCount, base.CirculantSize);
  printf("EncodingRank=%d\n", encoding.Rank);
  printf("DecoderConfig=%s\n", decoderConfigPath);
  printf("\n");
  printf("Base Matrix:\n");
  {
    int m;
    int k;
    for (m = 0; m < base.RowBlockCount; m++) {
      for (k = 0; k < base.ColBlockCount; k++) {
        printf("%d  ", base.ShiftMatrix[m][k]);
      }
      printf("\n");
    }
  }

#if GDBF
  printf("-------------------------La-GDBF--------------------------------------------------\n");
#endif

#if PGDBF
  printf("-------------------------La-P-GDBF--------------------------------------------------\n");
#endif

  if (decoderConfig.enableDatasetCollection) {
    runType = "collect";
  } else {
    runType = DecoderTypeToName(decoderConfig.decoderType);
  }
  snprintf(resultDir, sizeof(resultDir), "results/%s/%s", codeName, runType);
  EnsureDir(resultDir);
  snprintf(resultFileName, sizeof(resultFileName), "%s/simulation.res", resultDir);
  snprintf(mlSummaryCsvPath, sizeof(mlSummaryCsvPath), "%s/ml_outcome_summary.csv", resultDir);
  snprintf(mlFailureCsvPath, sizeof(mlFailureCsvPath), "%s/ml_failure_cases.csv", resultDir);

  fout = fopen(resultFileName, "a+");
  if (fout == NULL) {
    fprintf(stderr, "Output file %s error!.. Abort\n", resultFileName);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  PrintStatsHeader(
    fout,
    decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
    base.CirculantSize);

  if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
    int csvExists = (access(mlFailureCsvPath, 0) == 0);
    int recreateCsv = 0;

    if (csvExists) {
      FILE *hdr = fopen(mlFailureCsvPath, "r");
      if (hdr != NULL) {
        char headerLine[1024];
        if (fgets(headerLine, sizeof(headerLine), hdr) != NULL) {
          if (strstr(headerLine, "error_vn_indices") == NULL ||
              strstr(headerLine, "error_vn_last5_energies") == NULL ||
              strstr(headerLine, "global_max_last5_energies") == NULL ||
              strstr(headerLine, "global_max_vn_indices_last5") == NULL ||
              strstr(headerLine, "ml_proposed_vn_indices") == NULL ||
              strstr(headerLine, "ml_proposed_error_overlap") == NULL) {
            recreateCsv = 1;
          }
        }
        fclose(hdr);
      }
    }

    mlFailureFile = fopen(mlFailureCsvPath, recreateCsv ? "w" : "a");
    if (mlFailureFile == NULL) {
      fprintf(stderr, "WARNING: cannot open ML failure CSV: %s\n", mlFailureCsvPath);
    } else if (!csvExists || recreateCsv) {
      fprintf(mlFailureFile,
        "alpha,frame_index_alpha,frame_index_global,used_iterations,frame_bit_errors,is_codeword,ml_invoked,ml_calls_in_frame,error_vn_indices,error_vn_last5_energies,global_max_last5_energies,global_max_vn_indices_last5,ml_proposed_vn_indices,ml_proposed_error_overlap\n");
    }
  }

  workVector = (int *)calloc(matrix.N, sizeof(int));
  codeword = (int *)calloc(matrix.N, sizeof(int));
  receivedword = (int *)calloc(matrix.N, sizeof(int));
  decodedBits = (int *)calloc(matrix.N, sizeof(int));
  bitEnergy = (int *)calloc(matrix.N, sizeof(int));
  lastBitEnergyHistory = (int *)calloc((size_t)matrix.N * 5u, sizeof(int));
  mlProposedIndices = (int *)calloc(matrix.N, sizeof(int));
  checkNodeSyndrome = (int *)calloc(base.CirculantSize, sizeof(int));
  layerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));
  shiftedLayerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));

  if (decoderConfig.enableDatasetCollection) {
  {
    char datasetDir[512];
    char datasetPath[512];
    int datasetExists;
    int recreateDataset = 0;
    int featureDim = DecoderFeatureDimension(&decoderConfig);
    int featureCount = decoderConfig.candidateCount * featureDim;
    int labelCount = decoderConfig.candidateCount;
    snprintf(datasetDir, sizeof(datasetDir), "datasets/%s", codeName);
    EnsureDir(datasetDir);
    snprintf(datasetPath, sizeof(datasetPath), "%s/dataset.csv", datasetDir);
    datasetExists = (access(datasetPath, 0) == 0);

    if (datasetExists) {
      FILE *headerFile = fopen(datasetPath, "r");
      if (headerFile) {
        char headerLine[8192];
        if (fgets(headerLine, sizeof(headerLine), headerFile)) {
          int colCount = 1;
          int ci;
          int expectedCols = featureCount + labelCount;
          for (ci = 0; headerLine[ci] != '\0'; ci++) {
            if (headerLine[ci] == ',') {
              colCount++;
            }
          }
          if (colCount != expectedCols || strncmp(headerLine, "C0_", 3) != 0) {
            recreateDataset = 1;
          }
        }
        fclose(headerFile);
      }
    }

    datasetFile = fopen(datasetPath, recreateDataset ? "w" : "a");
    if (!datasetFile) {
        printf("ERROR opening dataset file\n");
        return 1;
    }
    if (!datasetExists || recreateDataset)
    {
      int fi;
      int li;
      for (fi = 0; fi < decoderConfig.candidateCount; fi++) {
        int fj;
        for (fj = 0; fj < featureDim; fj++) {
          const char *fname = DecoderFeatureName(&decoderConfig, fj);
          fprintf(datasetFile, "C%d_%s,", fi, fname);
        }
      }
      for (li = 0; li < labelCount; li++) {
        fprintf(datasetFile, "Y%d%s", li, (li < labelCount - 1) ? "," : "\n");
      }
    }
  }
  }

    if (workVector == NULL || codeword == NULL || receivedword == NULL || decodedBits == NULL ||
      lastBitEnergyHistory == NULL ||
      mlProposedIndices == NULL ||
      bitEnergy == NULL || checkNodeSyndrome == NULL || layerVariableBuffer == NULL || shiftedLayerVariableBuffer == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding/decoder buffers.\n");
    free(workVector);
    free(codeword);
    free(receivedword);
    free(decodedBits);
    free(lastBitEnergyHistory);
    free(mlProposedIndices);
    free(bitEnergy);
    free(checkNodeSyndrome);
    free(layerVariableBuffer);
    free(shiftedLayerVariableBuffer);
    fclose(fout);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  srand(RAND_SEED);

  // Load error indexes if provided
  int *errorIndexes = NULL;
  int errorIndexCount = 0;
  if (hasErrorIndexesPath && errorIndexesPath[0] != '\0') {
    if (LoadErrorIndexesFromFile(errorIndexesPath, &errorIndexes, &errorIndexCount) != 0) {
      fprintf(stderr, "Failed to load error indexes from file: %s\n", errorIndexesPath);
      free(workVector);
      free(codeword);
      free(receivedword);
      free(decodedBits);
      free(lastBitEnergyHistory);
      free(mlProposedIndices);
      free(bitEnergy);
      free(checkNodeSyndrome);
      free(layerVariableBuffer);
      free(shiftedLayerVariableBuffer);
      fclose(fout);
      FreeEncodingTransformData(&encoding);
      FreeBaseMatrixData(&base);
      FreeSparseMatrixData(&matrix);
      return 1;
    }
    printf("Loaded %d error indexes from file: %s\n", errorIndexCount, errorIndexesPath);
  }

  for (alpha = alpha_max; alpha > alpha_min; alpha -= alpha_step) {
    SimulationStats stats;
    long long alphaFeedbackSuccessAuxEqHist[AUX_EQ_HIST_MAX];
    long long alphaFeedbackSuccessAuxEqOverflow = 0;
    long long alphaNoFeedbackSuccess = 0;
    long long alphaFeedbackSuccess = 0;
    long long alphaNotManagedDecode = 0;
    long long alphaFramesTested = 0;
    long long alphaFramesDecodedClean = 0;
    long long alphaFramesBaselineOnlyDecoded = 0;
    long long alphaFramesMlNeeded = 0;
    long long alphaFramesMlEscaped = 0;
    long long alphaFramesMlNoEscape = 0;
    long long alphaFramesMlInvokedDecodedClean = 0;
    long long alphaFailedFramesMlInvoked = 0;
    long long alphaFailedFramesNoMlInvocation = 0;
    long long alphaStagnationBefore = decoderRuntimeStats.stagnationEvents;
    long long alphaMlCallsBeforeAll = decoderRuntimeStats.modelInferenceCalls;
    long long alphaMlEscapesBeforeAll = decoderRuntimeStats.mlEscapes;
    memset(alphaFeedbackSuccessAuxEqHist, 0, sizeof(alphaFeedbackSuccessAuxEqHist));
    ResetSimulationStats(&stats);

    for (nb = 0; nb < NbMonteCarlo; nb++) {
      int isCodeword;
      int cleanDecoded;
      int usedIterations;
      int frameBitErrors;
      int addedAuxEquations = 0;
      int shiftMatrixGenerations = 0;  /* Number of layer shift matrix proposals */
      int maxEnergyBitsBeforeFeedbackMin = 0;
      int maxEnergyBitsBeforeFeedbackMax = 0;
      long long maxEnergyBitsBeforeFeedbackSum = 0;
      int maxEnergyBitsBeforeFeedbackCount = 0;
      int unsuccessfulRoundsToSyndrome0 = -1;
      int lastBitEnergyHistoryCount = 0;
      int mlProposedCount = 0;
      long long mlCallsBefore = decoderRuntimeStats.modelInferenceCalls;
      long long mlEscapesBefore = decoderRuntimeStats.mlEscapes;
      long long mlCorrectedBitsBefore = decoderRuntimeStats.mlCorrectedBits;

      // EncodeRandomCodeword(
      //   encoding.Rank,
      //   matrix.N,
      //   encoding.SystematicMatrix,
      //   encoding.ColumnPermutation,
      //   workVector,
      //   codeword);

      if (decoderConfig.quantumOnlySyndrome) {
        // Quantum-only mode: generate a proper full codeword satisfying H·c = 0,
        // apply noise to all N bits, no classical channel involved
        EncodeRandomCodeword(
          encoding.Rank,
          matrix.N,
          encoding.SystematicMatrix,
          encoding.ColumnPermutation,
          workVector,
          codeword);

        if (errorIndexes != NULL && errorIndexCount > 0) {
          AddBscNoiseFromIndexes(codeword, receivedword, matrix.N, errorIndexes, errorIndexCount);
        } else {
          AddBscNoise(codeword, receivedword, matrix.N, alpha);
        }
      } else {
        // Standard split mode: x over quantum channel, parity over classical channel
        int *x = workVector;
        for (int i = 0; i < matrix.N - matrix.M; i++) {
          x[i] = rand() & 1;
        }
        EncodeSplitCodeword(&matrix, x, codeword);

        if (errorIndexes != NULL && errorIndexCount > 0) {
          AddBscNoiseFromIndexesSplit(codeword, receivedword, matrix.N - matrix.M, matrix.M, errorIndexes, errorIndexCount);
        } else {
          AddBscNoiseSplit(codeword, receivedword, matrix.N - matrix.M, matrix.M, alpha);
        }
      }

      if (DecodeFrameWithConfig(
            &base,
            receivedword,
            codeword,
            matrix.N,
            maxDecoderIterations,
            decodedBits,
            bitEnergy,
            checkNodeSyndrome,
            layerVariableBuffer,
            shiftedLayerVariableBuffer,
            &isCodeword,
            &usedIterations,
            &frameBitErrors,
            &addedAuxEquations,
            &shiftMatrixGenerations,
            &maxEnergyBitsBeforeFeedbackMin,
            &maxEnergyBitsBeforeFeedbackMax,
            &maxEnergyBitsBeforeFeedbackSum,
            &maxEnergyBitsBeforeFeedbackCount,
            &unsuccessfulRoundsToSyndrome0,
            lastBitEnergyHistory,
            &lastBitEnergyHistoryCount,
            mlProposedIndices,
            &mlProposedCount,
            matrix.N,
            &decoderConfig,
            &decoderRuntimeStats,
            datasetFile,
            errorIndexes,
            errorIndexCount,
            nb,
            alpha) != 0) {
        fprintf(stderr, "Decoder error.\n");
        free(workVector);
        free(codeword);
        free(receivedword);
        free(decodedBits);
        free(lastBitEnergyHistory);
        free(mlProposedIndices);
        free(bitEnergy);
        free(checkNodeSyndrome);
        free(layerVariableBuffer);
        free(shiftedLayerVariableBuffer);
        fclose(fout);
        FreeEncodingTransformData(&encoding);
        FreeBaseMatrixData(&base);
        FreeSparseMatrixData(&matrix);
        return 1;
      }

      {
        cleanDecoded = (isCodeword && frameBitErrors == 0);
        overallFramesTested++;
        alphaFramesTested++;
        if (cleanDecoded) {
          overallFramesDecodedClean++;
          alphaFramesDecodedClean++;
        }

        {
          long long frameMlCalls = decoderRuntimeStats.modelInferenceCalls - mlCallsBefore;
          long long frameMlEscapes = decoderRuntimeStats.mlEscapes - mlEscapesBefore;
          long long frameMlCorrectedBits = decoderRuntimeStats.mlCorrectedBits - mlCorrectedBitsBefore;

          if (frameMlEscapes > 0) {
            overallFramesWithAppliedEscape++;
            if (cleanDecoded) {
              overallFramesWithAppliedEscapeDecodedClean++;
              overallEscapeActionsInCleanFrames += frameMlEscapes;
            }
          }

          if (frameMlCalls > 0) {
            overallFramesMlNeeded++;
            alphaFramesMlNeeded++;
            overallFramesWithMlInvocation++;
            overallMlCallsPerInvokedFrameSum += frameMlCalls;
            if (overallMlCallsPerInvokedFrameMin < 0 || frameMlCalls < overallMlCallsPerInvokedFrameMin) {
              overallMlCallsPerInvokedFrameMin = frameMlCalls;
            }
            if (frameMlCalls > overallMlCallsPerInvokedFrameMax) {
              overallMlCallsPerInvokedFrameMax = frameMlCalls;
            }

            /*
             * Escape-based effectiveness definition:
             * - effective: ML was invoked and at least one ML mask was accepted
             *   (i.e., produced a non-rolled-back escape action) in this frame.
             * - not effective: ML was invoked but produced no accepted escape action.
             */
            if (frameMlEscapes > 0) {
              overallFramesMlEscaped++;
              alphaFramesMlEscaped++;
              overallFramesWithSuccessfulMl++;
              overallCorrectedBitsPerSuccessfulFrameSum += frameMlCorrectedBits;
              if (overallCorrectedBitsPerSuccessfulFrameMin < 0 || frameMlCorrectedBits < overallCorrectedBitsPerSuccessfulFrameMin) {
                overallCorrectedBitsPerSuccessfulFrameMin = frameMlCorrectedBits;
              }
              if (frameMlCorrectedBits > overallCorrectedBitsPerSuccessfulFrameMax) {
                overallCorrectedBitsPerSuccessfulFrameMax = frameMlCorrectedBits;
              }
            } else {
              overallFramesMlNoEscape++;
              alphaFramesMlNoEscape++;
            }

            if (cleanDecoded) {
              overallFramesMlInvokedDecodedClean++;
              alphaFramesMlInvokedDecodedClean++;
            } else {
              overallFailedFramesMlInvoked++;
              alphaFailedFramesMlInvoked++;
            }
          } else {
            if (cleanDecoded) {
              overallFramesBaselineOnlyDecoded++;
              alphaFramesBaselineOnlyDecoded++;
            } else {
              overallFailedFramesNoMlInvocation++;
              alphaFailedFramesNoMlInvocation++;
            }
          }

          if (!cleanDecoded && mlFailureFile != NULL) {
            char errorVnIndices[16384];
            char errorVnEnergies[65536];
            char globalMaxEnergies[1024];
            char globalMaxVnIndices[65536];
            char mlProposedVnIndices[16384];
            char mlProposedErrorOverlap[16384];
            BuildErrorVnIndexList(decodedBits, codeword, matrix.N, errorVnIndices, (int)sizeof(errorVnIndices));
            BuildErrorVnEnergyHistoryList(
              decodedBits,
              codeword,
              matrix.N,
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              errorVnEnergies,
              (int)sizeof(errorVnEnergies));
            BuildGlobalMaxEnergyHistoryList(
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              matrix.N - matrix.M,
              globalMaxEnergies,
              (int)sizeof(globalMaxEnergies));
            BuildGlobalMaxVnIndexHistoryList(
              lastBitEnergyHistory,
              lastBitEnergyHistoryCount,
              matrix.N - matrix.M,
              globalMaxVnIndices,
              (int)sizeof(globalMaxVnIndices));
            BuildIndexListFromBuffer(
              mlProposedIndices,
              mlProposedCount,
              mlProposedVnIndices,
              (int)sizeof(mlProposedVnIndices));
            BuildMlProposedErrorOverlapList(
              mlProposedIndices,
              mlProposedCount,
              decodedBits,
              codeword,
              matrix.N,
              mlProposedErrorOverlap,
              (int)sizeof(mlProposedErrorOverlap));
            fprintf(
              mlFailureFile,
              "%.6f,%d,%lld,%d,%d,%d,%d,%lld,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
              alpha,
              nb + 1,
              overallFramesTested,
              usedIterations,
              frameBitErrors,
              isCodeword,
              (frameMlCalls > 0) ? 1 : 0,
              frameMlCalls,
              errorVnIndices,
              errorVnEnergies,
              globalMaxEnergies,
              globalMaxVnIndices,
              mlProposedVnIndices,
              mlProposedErrorOverlap);
          }
        }
      }

      if (addedAuxEquations < 0) {
        addedAuxEquations = 0;
      }

      if (cleanDecoded) {
        if (unsuccessfulRoundsToSyndrome0 >= 0) {
          alphaFeedbackSuccess++;
          if (shiftMatrixGenerations < AUX_EQ_HIST_MAX) {
            alphaFeedbackSuccessAuxEqHist[shiftMatrixGenerations]++;
          } else {
            alphaFeedbackSuccessAuxEqOverflow++;
          }
        } else {
          alphaNoFeedbackSuccess++;
        }
      } else {
        alphaNotManagedDecode++;
      }

      UpdateSimulationStats(
        &stats,
        frameBitErrors,
        isCodeword,
        usedIterations,
        maxDecoderIterations,
        addedAuxEquations,
        unsuccessfulRoundsToSyndrome0,
        maxEnergyBitsBeforeFeedbackMin,
        maxEnergyBitsBeforeFeedbackMax,
        maxEnergyBitsBeforeFeedbackSum,
        maxEnergyBitsBeforeFeedbackCount);

      if (stats.nbtestedframes % 1000000 == 0) {
        PrintStatsLine(
          alpha,
          &stats,
          matrix.N,
          fout,
          decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
          base.CirculantSize);
      }

      if (NBframes > 0 && stats.NbTotalErrors == NBframes) {
        break;
      }
    }

    if ((stats.nbtestedframes % 1000000) != 0) {
      PrintStatsLine(
        alpha,
        &stats,
        matrix.N,
        fout,
        decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK,
        base.CirculantSize);
    }

    if (decoderConfig.decoderType == DECODER_TYPE_FEEDBACK_SHIFT || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
      PrintFeedbackShiftDistributionSummary(
        fout,
        alpha,
        alphaFramesTested,
        alphaNoFeedbackSuccess,
        alphaFeedbackSuccess,
        alphaNotManagedDecode,
        alphaFeedbackSuccessAuxEqHist,
        alphaFeedbackSuccessAuxEqOverflow);
    }
    
    if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
      AppendMlOutcomeSummaryCsvPerAlpha(
        mlSummaryCsvPath,
        resultFileName,
        NbMonteCarlo,
        maxDecoderIterations,
        NBframes,
        alpha,
        alpha_max,
        alpha_min,
        alpha_step,
        alphaFramesTested,
        alphaFramesDecodedClean,
        alphaFramesBaselineOnlyDecoded,
        alphaFramesMlNeeded,
        alphaFramesMlEscaped,
        alphaFramesMlNoEscape,
        alphaFramesMlInvokedDecodedClean,
        decoderRuntimeStats.stagnationEvents - alphaStagnationBefore,
        decoderRuntimeStats.modelInferenceCalls - alphaMlCallsBeforeAll,
        decoderRuntimeStats.mlEscapes - alphaMlEscapesBeforeAll);

      (void)alphaFailedFramesMlInvoked;
      (void)alphaFailedFramesNoMlInvocation;
    }
  }

  free(workVector);
  free(codeword);
  free(receivedword);
  free(decodedBits);
  free(lastBitEnergyHistory);
  free(mlProposedIndices);
  free(bitEnergy);
  free(checkNodeSyndrome);
  free(layerVariableBuffer);
  free(shiftedLayerVariableBuffer);
  free(errorIndexes);

  fclose(fout);
  FreeEncodingTransformData(&encoding);
  FreeBaseMatrixData(&base);
  FreeSparseMatrixData(&matrix);

  if (datasetFile) fclose(datasetFile);
  if (mlFailureFile) fclose(mlFailureFile);

  if (decoderConfig.decoderType == DECODER_TYPE_ML || decoderConfig.decoderType == DECODER_TYPE_ML_FEEDBACK) {
    printf("\n=== ML Escape Diagnostics ===\n");
    printf("  Stagnation events detected : %lld\n", decoderRuntimeStats.stagnationEvents);
    printf("  ML counterfactual skips    : %lld\n", decoderRuntimeStats.mlCounterfactualSkips);
    printf("  ML predict() calls         : %lld\n", decoderRuntimeStats.modelInferenceCalls);
    printf("  ML escapes applied         : %lld\n", decoderRuntimeStats.mlEscapes);
    printf("  Bits corrected by ML       : %lld\n", decoderRuntimeStats.mlCorrectedBits);

    if (overallFramesTested > 0) {
      printf("\n=== Frame-Level Outcome Summary ===\n");
      printf("  Frames tested                          : %lld\n", overallFramesTested);
      printf("  Frames decoded clean                   : %lld\n", overallFramesDecodedClean);
      printf("  Decoded by baseline GDBF only          : %lld\n", overallFramesBaselineOnlyDecoded);
      printf("  Stuck-triggered frames (ML invoked)    : %lld\n", overallFramesMlNeeded);
      printf("  ML escape success (applied escapes)    : %.2f%%\n",
             (decoderRuntimeStats.mlEscapes > 0)
               ? (100.0 * (double)overallEscapeActionsInCleanFrames / (double)decoderRuntimeStats.mlEscapes)
               : 0.0);

      printf("\n=== Failure Breakdown ===\n");
      printf("  Failed frames with ML invoked          : %lld\n", overallFailedFramesMlInvoked);
      printf("  Failed frames with no ML invocation    : %lld\n", overallFailedFramesNoMlInvocation);

      printf("\n=== ML Calls Per Invoked Frame ===\n");
      if (overallFramesWithMlInvocation > 0) {
        printf("  Min ML calls/frame                     : %lld\n", overallMlCallsPerInvokedFrameMin);
        printf("  Max ML calls/frame                     : %lld\n", overallMlCallsPerInvokedFrameMax);
        printf("  Avg ML calls/frame                     : %.2f\n",
               (double)overallMlCallsPerInvokedFrameSum / (double)overallFramesWithMlInvocation);
      } else {
        printf("  Min ML calls/frame                     : n/a\n");
        printf("  Max ML calls/frame                     : n/a\n");
        printf("  Avg ML calls/frame                     : n/a\n");
      }

      printf("\n=== Corrected Bits Per Successful ML Frame ===\n");
      if (overallFramesWithSuccessfulMl > 0) {
        printf("  Min corrected bits/frame               : %lld\n", overallCorrectedBitsPerSuccessfulFrameMin);
        printf("  Max corrected bits/frame               : %lld\n", overallCorrectedBitsPerSuccessfulFrameMax);
        printf("  Avg corrected bits/frame               : %.2f\n",
               (double)overallCorrectedBitsPerSuccessfulFrameSum / (double)overallFramesWithSuccessfulMl);
      } else {
        printf("  Min corrected bits/frame               : n/a\n");
        printf("  Max corrected bits/frame               : n/a\n");
        printf("  Avg corrected bits/frame               : n/a\n");
      }
    }

    printf("  CSV summary written to               : %s\n", mlSummaryCsvPath);
    if (mlFailureFile != NULL) {
      printf("  Failure cases CSV written to         : %s\n", mlFailureCsvPath);
    }
  }

  if (decoderConfig.enableDatasetCollection) {
    printf("\n=== Dataset Collection ===\n");
    printf("  Rows written                           : %lld\n", decoderRuntimeStats.datasetRows);
  }

  return 0;
}
