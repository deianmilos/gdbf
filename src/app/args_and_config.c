#include "app/args_and_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void SkipBOM(FILE *f)
{
  /* Skip UTF-8 BOM if present (0xEF, 0xBB, 0xBF) */
  unsigned char byte1, byte2, byte3;
  long pos = ftell(f);
  if (fread(&byte1, 1, 1, f) == 1 &&
      fread(&byte2, 1, 1, f) == 1 &&
      fread(&byte3, 1, 1, f) == 1) {
    if (byte1 == 0xEF && byte2 == 0xBB && byte3 == 0xBF) {
      /* BOM detected, position is already after it */
      return;
    }
  }
  /* No BOM, reset to start */
  fseek(f, pos, SEEK_SET);
}

/* Forward declaration for PrintUsage */
void PrintUsage(const char *program);

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
  int *hasDecoderConfigPath)
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

  if (!hasAlpha) {
    fprintf(stderr, "Missing required option: --alpha\n");
    PrintUsage(argv[0]);
    return 1;
  }

  if (!hasCode) {
    codeName[0] = '\0';
  }

  if (!hasAlphaMax) *alphaMax = *alpha;
  if (!hasAlphaMin) *alphaMin = *alpha - 1.0f;
  if (!hasAlphaStep) *alphaStep = 1.0f;

  return 0;
}

int LoadCodeNameFromConfig(const char *filePath, char *codeName, int codeNameLen)
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

  SkipBOM(f);

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
