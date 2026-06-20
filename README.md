# GDBF-FEEDBAKC-ML: QC-LDPC Decoder + ML Enhancement + Feedback Enhancement

This repository contains a QC-LDPC simulator with multiple decoder modes:

- Baseline GDBF
- PGDBF
- ML-assisted escape
- Feedback-shift mode

It also includes dataset collection, training, model export to C headers, and plotting scripts.

## 1) What You Need

### C toolchain

- GCC (MinGW on Windows works)

### Python

- Python 3.10+ (recommended: Conda environment for ML training)
- Core decoder compilation: no Python required
- ML training: requires PyTorch, NumPy, Pandas

**Setup Conda environment for training:**

```powershell
conda create -n gdbf-ml python=3.10 pytorch numpy pandas matplotlib -c pytorch
conda activate gdbf-ml
```

Alternatively, install packages to existing Python:

```powershell
python -m pip install torch numpy pandas matplotlib
```

## 2) Repository Layout (Important Paths)

- `src/` canonical C source code
- `include/` headers
- `configs/decoder/` runtime decoder configuration files
- `codes/<code_name>/` parity/check matrix data per code
- `datasets/<code_name>/dataset.csv` collected training data
- `training/` ML training + export scripts
- `model/` generated model headers used by C decoder
- `results/<code_name>/<run_type>/simulation.res` simulation outputs
- `visualize/` plotting scripts

## 2.1) Core Modules

- Decoder config parser and defaults: `src/config/decoder_config.c`
- Decode orchestrator: `src/framework/decoder_framework.c`
- Frame lifecycle/buffer management: `src/framework/frame_setup.c`
- ML round logic (trigger/inference/rollback): `src/ml/ml_round.c`
- Dataset row writing helpers: `src/ml/dataset_writer.c`

## 3) Build

```powershell
gcc -Iinclude -Iinclude/common -Iinclude/core -Iinclude/feedback -Iinclude/ml -Iinclude/framework -Iinclude/config -Iinclude/stats -O2 -Wno-unused-variable -o GDBF src/app/main.c src/app/args_and_config.c src/app/logging.c src/core/matrix_io.c src/core/encoding.c src/core/channel.c src/core/decoder.c src/core/stagnation_detection.c src/feedback/decoder_feedback_shift.c src/feedback/decoder_receiver.c src/feedback/feedback_round.c src/ml/decoder_ml.c src/ml/decoder_perturb.c src/ml/candidate_selection.c src/ml/feature_extractor.c src/ml/labeling_strategy.c src/ml/dataset_writer.c src/ml/ml_round.c src/config/decoder_config.c src/stats/stats.c src/framework/frame_setup.c src/framework/decoder_framework.c -lm
```


## 4) Check CLI Help

```powershell
.\GDBF.exe --help
```

Named-argument mode:

```text
--frames <N> --max-iter <N> [--code <CodeName>] --alpha <A>
[--nb-frames <N>] [--alpha-max <A>] [--alpha-min <A>] [--alpha-step <A>]
[--decoder-config <path>]
```

Notes:

- `--frames` and `--max-iter` are required.
- `--alpha` is required.
- You can still use positional mode, but named arguments are recommended.

## 5) Runtime Configuration

The decoder reads configuration from a cfg file.

Default file used in examples:

- `configs/decoder/default.cfg`

You can override config path per run:

```powershell
.\GDBF.exe ... --decoder-config configs/decoder/default.cfg
```

Common keys in cfg:

- `code` (folder under `codes/`)
- `decoder_type` = `gdbf` | `pgdbf` | `ml` | `feedback_shift` | `ml_feedback`
- `collect` = 0/1
- `candidate_selection` = `max_energy_checks` (single supported strategy)
- `candidate_k`
- `feature_set` or `feature_list`
- `ml_trigger_mode` = `none` | `periodic` | `state_based`
- `ml_periodic_interval`
- `ml_invoke_only_if_baseline_fails`
- `allowed_worsening`

Ready-to-use example configs are in `configs/decoder/`, including:

- `baseline.cfg`
- `pgdbf.cfg`
- `best_ml_escape.cfg`
- `best_ml_feedback.cfg`
- `collect_top6_corrective_mask.cfg`

## 6) First Successful Run (Baseline)

Use a known config and run one alpha point:

```powershell
.\GDBF.exe --frames 5000 --max-iter 50 --alpha 0.006 --decoder-config configs/decoder/baseline.cfg
```

Result files are written under `results/<code_name>/baseline/`.

## 7) Alpha Sweep Run

```powershell
.\GDBF.exe --frames 1000000 --max-iter 100 --alpha 0 --nb-frames 100 --alpha-max 0.010 --alpha-min 0.003 --alpha-step 0.001 --decoder-config configs/decoder/baseline.cfg
```

Tip: keep the same sweep settings when comparing baseline vs ML.

## 8) Run Modes

Switch mode by changing cfg file (or `decoder_type` inside it), then run the same command pattern.

### Baseline GDBF

- `decoder_type = gdbf`
- `collect = 0`

### PGDBF

- `decoder_type = pgdbf`
- tune `pgdbf_flip_probability`

### ML-assisted decoding

- `decoder_type = ml`
- `collect = 0`
- choose candidate + feature settings compatible with your trained model

### Feedback-shift mode

- `decoder_type = feedback_shift`
- tune `feedback_trigger_iter` and related `feedback_*` keys
- set `feedback_continue_from_current = 1|0` to choose whether post-feedback decoding continues from the current state or restarts from the received word

### ML + Feedback hybrid mode (`ml_feedback`)

- `decoder_type = ml_feedback`
- this mode enables both ML rounds and feedback-shift rounds in the same decode loop

Short behavior summary:

- Decoder runs normal layered GDBF iterations.
- On stuck states (or periodic trigger), ML proposes candidate flips and applies the mask when allowed by policy.
- Feedback-shift rounds can also trigger (by `feedback_trigger_iter` and related settings), adding auxiliary parity-check energy guidance.
- If neither ML nor feedback resolves the state, decoding continues with baseline perturbation steps.

Use `configs/decoder/best_ml_feedback.cfg` as the starting profile.

Example run:

```powershell
.\GDBF.exe --frames 200000 --max-iter 100 --alpha 0.012 --decoder-config configs/decoder/best_ml_feedback.cfg
```

## 9) Dataset Collection

Set collect mode in cfg, for example `configs/decoder/collect_top6_corrective_mask.cfg`.

Important: collection uses the ML round path in the code, so the collection cfg should use `decoder_type = ml` together with `collect = 1` and a trigger mode such as `ml_trigger_mode = state_based`.

Then run:

```powershell
.\GDBF.exe --frames 200000 --max-iter 30 --alpha 0 --nb-frames 0 --alpha-max 0.020 --alpha-min 0.012 --alpha-step 0.003 --decoder-config configs/decoder/collect_top6_corrective_mask.cfg
```

Output dataset:

- `datasets/<code_name>/dataset.csv`

## 10) Train and Export Model Headers

**Architecture:** Mask-only corrective model (direct per-bit prediction).

Default training config:

- `configs/ml/default.json`

Mask-only specialized config (higher class weights, tighter convergence):

- `configs/ml/corrective_mask_adaptive.json`

**Train with Conda environment:**

```powershell
conda run -n gdbf-ml python training/train.py --config configs/ml/corrective_mask_adaptive.json
```

**Or activate Conda and run directly:**

```powershell
conda activate gdbf-ml
python training/train.py --config configs/ml/corrective_mask_adaptive.json
```

Train for a specific code name:

```powershell
conda run -n gdbf-ml python training/train.py --code IRISC_dv4_R050_L54_N1296 --config configs/ml/default.json
```

**Generated headers:**

Training exports C header files into `model/`:

- `as_model_<code_name>_ref.h` — float reference (full precision)
- `as_model_<code_name>_quantized.h` — quantized fixed-point (int8 weights, int32 accumulators)
- `as_model_active_quantized.h` — active alias (included by decoder runtime)

**After training/export:**

Rebuild `GDBF` so the latest generated headers are compiled in:

```powershell
if (Test-Path .\GDBF.exe) { Remove-Item .\GDBF.exe }
gcc -Iinclude -Iinclude/common -Iinclude/core -Iinclude/feedback -Iinclude/ml -Iinclude/framework -Iinclude/config -Iinclude/stats -O2 -Wno-unused-variable -o GDBF src/app/main.c src/app/args_and_config.c src/app/logging.c src/core/matrix_io.c src/core/encoding.c src/core/channel.c src/core/decoder.c src/core/stagnation_detection.c src/feedback/decoder_feedback_shift.c src/feedback/decoder_receiver.c src/feedback/feedback_round.c src/ml/decoder_ml.c src/ml/decoder_perturb.c src/ml/candidate_selection.c src/ml/feature_extractor.c src/ml/labeling_strategy.c src/ml/dataset_writer.c src/ml/ml_round.c src/config/decoder_config.c src/stats/stats.c src/framework/frame_setup.c src/framework/decoder_framework.c -lm
```

## 11) Plot Current IRISC Comparison (Baseline + ML 2/3/4 it)

Use the current config-driven plotting workflow:

```powershell
python visualize/plot_from_config.py --config visualize/plot_config_irisc_ml_latest_2_3_4_vs_baseline.json
```

This generates the latest figures under:

- `visualize/plots/irisc_ml_latest_2_3_4_vs_baseline/`

Main outputs include:

- `fer_comparison_latest.png`
- `fer_ber_combined_latest.png`
- `iteration_comparison_latest.png`
- `failed_bits_histogram_latest.png`

Optional legacy quick plot (single baseline vs single ML result):

```powershell
python visualize/plot_results.py --baseline results/wifin_r_1_2/baseline/simulation.res --ml results/wifin_r_1_2/ml/simulation.res --output visualize/plots/wifin_r_1_2/fer_comparison.png --title "FER Comparison: Baseline vs ML"
```

## 12) Minimal End-to-End 

1. Build `GDBF`:
   ```powershell
   gcc -Iinclude -Iinclude/common -Iinclude/core -Iinclude/feedback -Iinclude/ml -Iinclude/framework -Iinclude/config -Iinclude/stats -O2 -Wno-unused-variable -o GDBF src/app/main.c src/app/args_and_config.c src/app/logging.c src/core/matrix_io.c src/core/encoding.c src/core/channel.c src/core/decoder.c src/core/stagnation_detection.c src/feedback/decoder_feedback_shift.c src/feedback/decoder_receiver.c src/feedback/feedback_round.c src/ml/decoder_ml.c src/ml/decoder_perturb.c src/ml/candidate_selection.c src/ml/feature_extractor.c src/ml/labeling_strategy.c src/ml/dataset_writer.c src/ml/ml_round.c src/config/decoder_config.c src/stats/stats.c src/framework/frame_setup.c src/framework/decoder_framework.c -lm
   ```
2. Run baseline sweep with `configs/decoder/baseline.cfg`.
3. Run collect mode to generate dataset with `configs/decoder/collect_top6_corrective_mask.cfg`.
4. Train and export mask-only model in Conda:
   ```powershell
   conda run -n gdbf-ml python training/train.py --config configs/ml/corrective_mask_adaptive.json
   ```
5. Rebuild `GDBF` to include newly generated headers:
   ```powershell
   if (Test-Path .\GDBF.exe) { Remove-Item .\GDBF.exe }
   gcc -Iinclude -Iinclude/common -Iinclude/core -Iinclude/feedback -Iinclude/ml -Iinclude/framework -Iinclude/config -Iinclude/stats -O2 -Wno-unused-variable -o GDBF src/app/main.c src/app/args_and_config.c src/app/logging.c src/core/matrix_io.c src/core/encoding.c src/core/channel.c src/core/decoder.c src/core/stagnation_detection.c src/feedback/decoder_feedback_shift.c src/feedback/decoder_receiver.c src/feedback/feedback_round.c src/ml/decoder_ml.c src/ml/decoder_perturb.c src/ml/candidate_selection.c src/ml/feature_extractor.c src/ml/labeling_strategy.c src/ml/dataset_writer.c src/ml/ml_round.c src/config/decoder_config.c src/stats/stats.c src/framework/frame_setup.c src/framework/decoder_framework.c -lm
   ```
6. Run ML mode with `configs/decoder/best_ml_escape.cfg`.
7. Generate the current comparison plots (baseline + ML 2/3/4 it):
   ```powershell
   python visualize/plot_from_config.py --config visualize/plot_config_irisc_ml_latest_2_3_4_vs_baseline.json
   ```

## 13) Training Configuration Parameters

Common training settings in `configs/ml/*.json`:

- `code_name` — LDPC code name (e.g., `IRISC_P_dv3_R050_L54_N1296`)
- `dataset_path` — path to collected training data CSV
- `training.epochs` — number of training epochs (default: 500)
- `training.batch_size` — batch size (default: 32)
- `training.learning_rate` — optimizer learning rate (default: 0.001)
- `training.hidden1`, `training.hidden2` — hidden layer sizes (default: 32, 16)
- `training.min_positive_count_for_output` — minimum positive samples per label bit
- `training.max_pos_weight` — max class weight ratio for positive class
- `training.mask_logit_threshold` — threshold for mask prediction (default: 0.0)
- `export.write_active_alias` — generate `as_model_active_quantized.h` (default: true)

Mask-only specific: all budget/variant-related config keys removed (simplified).
