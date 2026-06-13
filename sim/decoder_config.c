#include "decoder_config.h"

static int StrEq(const char *a, const char *b)
{
  if (a == NULL || b == NULL) {
    return 0;
  }
  return strcmp(a, b) == 0;
}

static char *TrimLeft(char *s)
{
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
    s++;
  }
  return s;
}

static void TrimRight(char *s)
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

static int ParseBoolean(const char *v, int fallback)
{
  if (StrEq(v, "1") || StrEq(v, "true") || StrEq(v, "yes") || StrEq(v, "on")) return 1;
  if (StrEq(v, "0") || StrEq(v, "false") || StrEq(v, "no") || StrEq(v, "off")) return 0;
  return fallback;
}

static int ParseFeatureListMask(const char *v)
{
  char buf[256];
  char *token;
  int mask = 0;

  if (v == NULL || *v == '\0') {
    return 0;
  }

  strncpy(buf, v, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  token = strtok(buf, ",");
  while (token != NULL) {
    char *t = TrimLeft(token);
    TrimRight(t);

    if (StrEq(t, "energy")) mask |= FEATURE_FLAG_ENERGY;
    else if (StrEq(t, "mismatch") || StrEq(t, "channel_disagreement")) mask |= FEATURE_FLAG_MISMATCH;
    else if (StrEq(t, "unsat") || StrEq(t, "unsatisfied") || StrEq(t, "unsatisfied_checks")) mask |= FEATURE_FLAG_UNSAT;
    else if (StrEq(t, "flip_impact")) mask |= FEATURE_FLAG_FLIP_IMPACT;
    else if (StrEq(t, "flip_count")) mask |= FEATURE_FLAG_FLIP_COUNT;

    token = strtok(NULL, ",");
  }

  return mask;
}

static int ReadEnvInt(const char *name, int fallback)
{
  const char *v = getenv(name);
  if (v == NULL || *v == '\0') {
    return fallback;
  }
  return atoi(v);
}

static double ReadEnvDouble(const char *name, double fallback)
{
  const char *v = getenv(name);
  if (v == NULL || *v == '\0') {
    return fallback;
  }
  return atof(v);
}

void DecoderConfigInitDefaults(DecoderConfig *config)
{
  if (config == NULL) {
    return;
  }

#if AS_ML_MODE
  config->decoderType = DECODER_TYPE_ML;
#elif PGDBF
  config->decoderType = DECODER_TYPE_PGDBF;
#else
  config->decoderType = DECODER_TYPE_GDBF;
#endif

  config->enableDatasetCollection = 0;
  config->labelingStrategy = LABELING_GROUND_TRUTH;

  config->candidateSelection = CANDIDATE_SELECTION_TOPK;
  config->candidateCount = 19;
  config->featureFlags = 0;
  config->featureSelectionExplicit = 0;
  config->rolloutIters = 3;
  config->rolloutTargetFlipCount = 0;
  config->pgdbfFlipProbability = 0.7;
  config->allowedWorsening = 4;
  config->feedbackTriggerIter = 15;
  config->feedbackMaskWindowIters = 10;
  config->feedbackDeltaMax = 3;
  config->feedbackTargetRows = 6;
  config->feedbackPrioritySeverityW = 4;
  config->feedbackPriorityPersistenceW = 3;
  config->feedbackPriorityFailW = 2;
  config->feedbackShiftColumnsPerRow = 2;
  config->feedbackSenderStrictObservability = 0;
  config->feedbackLogsEnabled = 1;
  config->feedbackRowSelectionMode = 0;  /* 0 = max severity, 1 = min non-zero severity */
  config->feedbackShiftSourceMode = 0;   /* 0 = fixed (default), 1 = random, 2 = fixed-number */
  config->feedbackShiftFixedDelta = 1;
  config->mlInvokeOnlyIfBaselineFails = 1;
  config->mlPeriodicInterval = 0;

  config->stagnation.energyHistoryLen = 8;
  config->stagnation.stagnationTrigger = 2;
  config->stagnation.oscillationTrigger = 5;
  config->errorIndexesLoggingEnabled = 0;
  config->quantumOnlySyndrome = 0;
  config->errorIndexesPath[0] = '\0';
}

int DecoderConfigLoadFromFile(DecoderConfig *config, const char *filePath)
{
  FILE *f;
  char line[1024];

  if (config == NULL || filePath == NULL || *filePath == '\0') {
    return 1;
  }

  f = fopen(filePath, "r");
  if (f == NULL) {
    return 1;
  }

  while (fgets(line, sizeof(line), f) != NULL) {
    char *p = TrimLeft(line);
    char *eq;
    char *k;
    char *v;

    TrimRight(p);
    if (*p == '\0' || *p == '#' || *p == ';') {
      continue;
    }

    eq = strchr(p, '=');
    if (eq == NULL) {
      continue;
    }

    *eq = '\0';
    k = TrimLeft(p);
    TrimRight(k);
    v = TrimLeft(eq + 1);
    TrimRight(v);

    if (StrEq(k, "decoder_type")) {
      if (StrEq(v, "gdbf")) config->decoderType = DECODER_TYPE_GDBF;
      else if (StrEq(v, "pgdbf")) config->decoderType = DECODER_TYPE_PGDBF;
      else if (StrEq(v, "ml")) config->decoderType = DECODER_TYPE_ML;
      else if (StrEq(v, "feedback_shift")) config->decoderType = DECODER_TYPE_FEEDBACK_SHIFT;
      else if (StrEq(v, "ml_feedback")) config->decoderType = DECODER_TYPE_ML_FEEDBACK;
    } else if (StrEq(k, "candidate_selection")) {
      if (StrEq(v, "topk")) config->candidateSelection = CANDIDATE_SELECTION_TOPK;
      else if (StrEq(v, "graph")) config->candidateSelection = CANDIDATE_SELECTION_GRAPH;
      else if (StrEq(v, "max_energy_checks")) config->candidateSelection = CANDIDATE_SELECTION_MAX_ENERGY_CHECKS;
    } else if (StrEq(k, "labeling_strategy")) {
      if (StrEq(v, "ground_truth")) config->labelingStrategy = LABELING_GROUND_TRUTH;
      else if (StrEq(v, "rollout")) config->labelingStrategy = LABELING_ROLLOUT;
      else if (StrEq(v, "corrective_mask")) config->labelingStrategy = LABELING_CORRECTIVE_MASK;
    } else if (StrEq(k, "feature_set")) {
      if (StrEq(v, "basic")) config->featureFlags = 0;
      else if (StrEq(v, "basic_plus_impact")) config->featureFlags = FEATURE_FLAG_FLIP_IMPACT;
      config->featureSelectionExplicit = 0;
    } else if (StrEq(k, "feature_list")) {
      config->featureFlags = ParseFeatureListMask(v);
      config->featureSelectionExplicit = 1;
    } else if (StrEq(k, "candidate_k")) {
      config->candidateCount = atoi(v);
    } else if (StrEq(k, "collect")) {
      config->enableDatasetCollection = ParseBoolean(v, config->enableDatasetCollection);
    } else if (StrEq(k, "rollout_iters")) {
      config->rolloutIters = atoi(v);
    } else if (StrEq(k, "rollout_target_flip_count")) {
      config->rolloutTargetFlipCount = atoi(v);
    } else if (StrEq(k, "pgdbf_flip_probability")) {
      config->pgdbfFlipProbability = atof(v);
    } else if (StrEq(k, "allowed_worsening")) {
      config->allowedWorsening = atoi(v);
    } else if (StrEq(k, "feedback_trigger_iter")) {
      config->feedbackTriggerIter = atoi(v);
    } else if (StrEq(k, "feedback_mask_window_iters")) {
      config->feedbackMaskWindowIters = atoi(v);
    } else if (StrEq(k, "feedback_delta_max")) {
      config->feedbackDeltaMax = atoi(v);
    } else if (StrEq(k, "feedback_target_rows")) {
      config->feedbackTargetRows = atoi(v);
    } else if (StrEq(k, "feedback_priority_severity_w")) {
      config->feedbackPrioritySeverityW = atoi(v);
    } else if (StrEq(k, "feedback_priority_persistence_w")) {
      config->feedbackPriorityPersistenceW = atoi(v);
    } else if (StrEq(k, "feedback_priority_fail_w")) {
      config->feedbackPriorityFailW = atoi(v);
    } else if (StrEq(k, "feedback_shift_columns_per_row")) {
      config->feedbackShiftColumnsPerRow = atoi(v);
    } else if (StrEq(k, "feedback_sender_strict_observability")) {
      config->feedbackSenderStrictObservability = ParseBoolean(v, config->feedbackSenderStrictObservability);
    } else if (StrEq(k, "feedback_logs_enabled")) {
      config->feedbackLogsEnabled = ParseBoolean(v, config->feedbackLogsEnabled);
    } else if (StrEq(k, "feedback_row_selection_mode")) {
      if (StrEq(v, "min")) config->feedbackRowSelectionMode = 1;
      else config->feedbackRowSelectionMode = 0;  /* default: max */
    } else if (StrEq(k, "feedback_shift_source_mode")) {
      if (StrEq(v, "random")) config->feedbackShiftSourceMode = 1;
      else if (StrEq(v, "fixed_number")) {
        config->feedbackShiftSourceMode = 2;
      }
      else config->feedbackShiftSourceMode = 0;  /* default: fixed */
    } else if (StrEq(k, "feedback_shift_fixed_delta")) {
      config->feedbackShiftFixedDelta = atoi(v);
    } else if (StrEq(k, "ml_invoke_only_if_baseline_fails")) {
      config->mlInvokeOnlyIfBaselineFails = ParseBoolean(v, config->mlInvokeOnlyIfBaselineFails);
    } else if (StrEq(k, "ml_periodic_interval")) {
      config->mlPeriodicInterval = atoi(v);
    } else if (StrEq(k, "energy_hist_len")) {
      config->stagnation.energyHistoryLen = atoi(v);
    } else if (StrEq(k, "stagnation_trigger")) {
      config->stagnation.stagnationTrigger = atoi(v);
    } else if (StrEq(k, "oscillation_trigger")) {
      config->stagnation.oscillationTrigger = atoi(v);
    } else if (StrEq(k, "error_indexes_path")) {
      strncpy(config->errorIndexesPath, v, sizeof(config->errorIndexesPath) - 1);
      config->errorIndexesPath[sizeof(config->errorIndexesPath) - 1] = '\0';
    } else if (StrEq(k, "error_indexes_logging_enabled")) {
      config->errorIndexesLoggingEnabled = ParseBoolean(v, config->errorIndexesLoggingEnabled);
    } else if (StrEq(k, "quantum_only_syndrome")) {
      config->quantumOnlySyndrome = ParseBoolean(v, config->quantumOnlySyndrome);
    }
  }

  fclose(f);

  if (config->candidateCount < 1) config->candidateCount = 1;
  if (config->rolloutIters < 1) config->rolloutIters = 1;
  if (config->rolloutTargetFlipCount < 0) config->rolloutTargetFlipCount = 0;
  if (config->feedbackMaskWindowIters < 1) config->feedbackMaskWindowIters = 1;
  if (config->feedbackDeltaMax < 1) config->feedbackDeltaMax = 1;
  if (config->feedbackDeltaMax > 64) config->feedbackDeltaMax = 64;
  if (config->feedbackTargetRows < 1) config->feedbackTargetRows = 1;
  if (config->feedbackTargetRows > 128) config->feedbackTargetRows = 128;
  if (config->feedbackPrioritySeverityW < 0) config->feedbackPrioritySeverityW = 0;
  if (config->feedbackPriorityPersistenceW < 0) config->feedbackPriorityPersistenceW = 0;
  if (config->feedbackPriorityFailW < 0) config->feedbackPriorityFailW = 0;
  if (config->feedbackShiftColumnsPerRow < 1) config->feedbackShiftColumnsPerRow = 1;
  if (config->feedbackShiftColumnsPerRow > 3) config->feedbackShiftColumnsPerRow = 3;
  if (config->feedbackShiftSourceMode < 0) config->feedbackShiftSourceMode = 0;
  if (config->feedbackShiftSourceMode > 2) config->feedbackShiftSourceMode = 2;
  if (config->feedbackShiftFixedDelta < 1) config->feedbackShiftFixedDelta = 1;
  if (config->feedbackShiftFixedDelta > 64) config->feedbackShiftFixedDelta = 64;
  if (config->mlPeriodicInterval < 0) config->mlPeriodicInterval = 0;
  if (config->stagnation.energyHistoryLen < 2) config->stagnation.energyHistoryLen = 2;
  if (config->pgdbfFlipProbability < 0.0) config->pgdbfFlipProbability = 0.0;
  if (config->pgdbfFlipProbability > 1.0) config->pgdbfFlipProbability = 1.0;

  return 0;
}

void DecoderConfigApplyEnv(DecoderConfig *config)
{
  const char *decoderType;
  const char *candidateSelection;
  const char *labeling;
  const char *featureSet;
  const char *featureList;

  if (config == NULL) {
    return;
  }

  decoderType = getenv("GDBF_DECODER_TYPE");
  candidateSelection = getenv("GDBF_CANDIDATE_SELECTION");
  labeling = getenv("GDBF_LABELING_STRATEGY");
  featureSet = getenv("GDBF_FEATURE_SET");
  featureList = getenv("GDBF_FEATURE_LIST");

  if (StrEq(decoderType, "gdbf")) config->decoderType = DECODER_TYPE_GDBF;
  if (StrEq(decoderType, "pgdbf")) config->decoderType = DECODER_TYPE_PGDBF;
  if (StrEq(decoderType, "ml")) config->decoderType = DECODER_TYPE_ML;
  if (StrEq(decoderType, "feedback_shift")) config->decoderType = DECODER_TYPE_FEEDBACK_SHIFT;
  if (StrEq(decoderType, "ml_feedback")) config->decoderType = DECODER_TYPE_ML_FEEDBACK;

  if (StrEq(candidateSelection, "topk")) config->candidateSelection = CANDIDATE_SELECTION_TOPK;
  if (StrEq(candidateSelection, "graph")) config->candidateSelection = CANDIDATE_SELECTION_GRAPH;
  if (StrEq(candidateSelection, "max_energy_checks")) config->candidateSelection = CANDIDATE_SELECTION_MAX_ENERGY_CHECKS;

  if (StrEq(labeling, "ground_truth")) config->labelingStrategy = LABELING_GROUND_TRUTH;
  if (StrEq(labeling, "rollout")) config->labelingStrategy = LABELING_ROLLOUT;
  if (StrEq(labeling, "corrective_mask")) config->labelingStrategy = LABELING_CORRECTIVE_MASK;

  if (StrEq(featureSet, "basic")) {
    config->featureFlags = 0;
    config->featureSelectionExplicit = 0;
  } else if (StrEq(featureSet, "basic_plus_impact")) {
    config->featureFlags = FEATURE_FLAG_FLIP_IMPACT;
    config->featureSelectionExplicit = 0;
  }

  if (featureList != NULL && *featureList != '\0') {
    config->featureFlags = ParseFeatureListMask(featureList);
    config->featureSelectionExplicit = 1;
  }

  config->candidateCount = ReadEnvInt("GDBF_CANDIDATE_K", config->candidateCount);
  config->enableDatasetCollection = ReadEnvInt("GDBF_COLLECT", config->enableDatasetCollection);
  config->rolloutIters = ReadEnvInt("GDBF_ROLLOUT_ITERS", config->rolloutIters);
  config->rolloutTargetFlipCount = ReadEnvInt("GDBF_ROLLOUT_TARGET_FLIP_COUNT", config->rolloutTargetFlipCount);
  config->allowedWorsening = ReadEnvInt("GDBF_ALLOWED_WORSENING", config->allowedWorsening);
  config->feedbackTriggerIter = ReadEnvInt("GDBF_FEEDBACK_TRIGGER_ITER", config->feedbackTriggerIter);
  config->feedbackMaskWindowIters = ReadEnvInt("GDBF_FEEDBACK_MASK_WINDOW_ITERS", config->feedbackMaskWindowIters);
  config->feedbackDeltaMax = ReadEnvInt("GDBF_FEEDBACK_DELTA_MAX", config->feedbackDeltaMax);
  config->feedbackTargetRows = ReadEnvInt("GDBF_FEEDBACK_TARGET_ROWS", config->feedbackTargetRows);
  config->feedbackPrioritySeverityW = ReadEnvInt("GDBF_FEEDBACK_PRIORITY_SEVERITY_W", config->feedbackPrioritySeverityW);
  config->feedbackPriorityPersistenceW = ReadEnvInt("GDBF_FEEDBACK_PRIORITY_PERSISTENCE_W", config->feedbackPriorityPersistenceW);
  config->feedbackPriorityFailW = ReadEnvInt("GDBF_FEEDBACK_PRIORITY_FAIL_W", config->feedbackPriorityFailW);
  config->feedbackShiftColumnsPerRow = ReadEnvInt("GDBF_FEEDBACK_SHIFT_COLUMNS_PER_ROW", config->feedbackShiftColumnsPerRow);
  config->feedbackSenderStrictObservability = ReadEnvInt("GDBF_FEEDBACK_SENDER_STRICT_OBSERVABILITY", config->feedbackSenderStrictObservability);
  config->feedbackLogsEnabled = ReadEnvInt("GDBF_FEEDBACK_LOGS", config->feedbackLogsEnabled);
  config->feedbackRowSelectionMode = ReadEnvInt("GDBF_FEEDBACK_ROW_SELECTION_MODE", config->feedbackRowSelectionMode);
  config->feedbackShiftFixedDelta = ReadEnvInt("GDBF_FEEDBACK_SHIFT_FIXED_DELTA", config->feedbackShiftFixedDelta);
  {
    const char *shiftSourceMode = getenv("GDBF_FEEDBACK_SHIFT_SOURCE_MODE");
    if (StrEq(shiftSourceMode, "random")) config->feedbackShiftSourceMode = 1;
    else if (StrEq(shiftSourceMode, "fixed_number")) {
      config->feedbackShiftSourceMode = 2;
    }
    else if (StrEq(shiftSourceMode, "fixed")) config->feedbackShiftSourceMode = 0;
    else config->feedbackShiftSourceMode = ReadEnvInt("GDBF_FEEDBACK_SHIFT_SOURCE_MODE", config->feedbackShiftSourceMode);
  }
  config->pgdbfFlipProbability = ReadEnvDouble("GDBF_PGDBF_P", config->pgdbfFlipProbability);
  config->mlInvokeOnlyIfBaselineFails = ReadEnvInt("GDBF_ML_INVOKE_ONLY_IF_BASELINE_FAILS", config->mlInvokeOnlyIfBaselineFails);
  config->mlPeriodicInterval = ReadEnvInt("GDBF_ML_PERIODIC_INTERVAL", config->mlPeriodicInterval);

  config->stagnation.energyHistoryLen = ReadEnvInt("GDBF_ENERGY_HIST_LEN", config->stagnation.energyHistoryLen);
  config->stagnation.stagnationTrigger = ReadEnvInt("GDBF_STAGNATION_TRIGGER", config->stagnation.stagnationTrigger);
  config->stagnation.oscillationTrigger = ReadEnvInt("GDBF_OSCILLATION_TRIGGER", config->stagnation.oscillationTrigger);

  if (config->candidateCount < 1) {
    config->candidateCount = 1;
  }
  if (config->rolloutIters < 1) {
    config->rolloutIters = 1;
  }
  if (config->rolloutTargetFlipCount < 0) {
    config->rolloutTargetFlipCount = 0;
  }
  if (config->feedbackTriggerIter < 1) {
    config->feedbackTriggerIter = 1;
  }
  if (config->feedbackMaskWindowIters < 1) {
    config->feedbackMaskWindowIters = 1;
  }
  if (config->feedbackDeltaMax < 1) {
    config->feedbackDeltaMax = 1;
  }
  if (config->feedbackDeltaMax > 64) {
    config->feedbackDeltaMax = 64;
  }
  if (config->feedbackTargetRows < 1) {
    config->feedbackTargetRows = 1;
  }
  if (config->feedbackTargetRows > 128) {
    config->feedbackTargetRows = 128;
  }
  if (config->feedbackPrioritySeverityW < 0) {
    config->feedbackPrioritySeverityW = 0;
  }
  if (config->feedbackPriorityPersistenceW < 0) {
    config->feedbackPriorityPersistenceW = 0;
  }
  if (config->feedbackPriorityFailW < 0) {
    config->feedbackPriorityFailW = 0;
  }
  if (config->feedbackShiftColumnsPerRow < 1) {
    config->feedbackShiftColumnsPerRow = 1;
  }
  if (config->feedbackShiftColumnsPerRow > 3) {
    config->feedbackShiftColumnsPerRow = 3;
  }
  if (config->feedbackShiftSourceMode < 0) {
    config->feedbackShiftSourceMode = 0;
  }
  if (config->feedbackShiftSourceMode > 2) {
    config->feedbackShiftSourceMode = 2;
  }
  if (config->feedbackShiftFixedDelta < 1) {
    config->feedbackShiftFixedDelta = 1;
  }
  if (config->feedbackShiftFixedDelta > 64) {
    config->feedbackShiftFixedDelta = 64;
  }
  if (config->mlPeriodicInterval < 0) {
    config->mlPeriodicInterval = 0;
  }
  if (config->stagnation.energyHistoryLen < 2) {
    config->stagnation.energyHistoryLen = 2;
  }
  if (config->pgdbfFlipProbability < 0.0) {
    config->pgdbfFlipProbability = 0.0;
  }
  if (config->pgdbfFlipProbability > 1.0) {
    config->pgdbfFlipProbability = 1.0;
  }
}

int DecoderFeatureDimension(const DecoderConfig *config)
{
  FeatureExtractorConfig featureConfig;
  if (config == NULL) {
    return 0;
  }
  featureConfig.featureFlags = config->featureFlags;
  featureConfig.explicitSelection = config->featureSelectionExplicit;
  featureConfig.candidateCount = config->candidateCount;
  return GetFeatureDimension(&featureConfig);
}

const char *DecoderFeatureName(const DecoderConfig *config, int localFeatureIndex)
{
  FeatureExtractorConfig featureConfig;
  if (config == NULL) {
    return "unknown";
  }
  featureConfig.featureFlags = config->featureFlags;
  featureConfig.explicitSelection = config->featureSelectionExplicit;
  featureConfig.candidateCount = config->candidateCount;
  return GetFeatureNameByLocalIndex(&featureConfig, localFeatureIndex);
}
