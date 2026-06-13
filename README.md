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

- Python 3.10+
- Recommended packages:
  - numpy
  - pandas
  - torch
  - matplotlib

Install Python packages:

```powershell
python -m pip install numpy pandas torch matplotlib
```

## 2) Repository Layout (Important Paths)

- `src/` C source code used for current builds
- `include/` headers
- `configs/decoder/` runtime decoder configuration files
- `codes/<code_name>/` parity/check matrix data per code
- `datasets/<code_name>/dataset.csv` collected training data
- `training/` ML training + export scripts
- `model/` generated model headers used by C decoder
- `results/<code_name>/<run_type>/simulation.res` simulation outputs
- `visualize/` plotting scripts

## 3) Build

```powershell
gcc -Iinclude -Iinclude/common -Iinclude/core -Iinclude/feedback -Iinclude/ml -Iinclude/framework -Iinclude/config -Iinclude/stats -O2 -Wno-unused-variable -o GDBF src/app/main.c src/core/matrix_io.c src/core/encoding.c src/core/channel.c src/core/decoder.c src/core/stagnation_detection.c src/feedback/decoder_feedback_shift.c src/feedback/decoder_receiver.c src/feedback/feedback_round.c src/ml/decoder_ml.c src/ml/decoder_perturb.c src/ml/candidate_selection.c src/ml/feature_extractor.c src/ml/labeling_strategy.c src/ml/ml_round.c src/config/decoder_config.c src/stats/stats.c src/framework/frame_setup.c src/framework/decoder_framework.c -lm
```


## 4) Check CLI Help

```powershell
.\GDBF.exe --help
```

Named-argument mode:

```text
--frames <N> --max-iter <N> [--code <CodeName>] --alpha <A>
[--nb-frames <N>] [--alpha-max <A>] [--alpha-min <A>] [--alpha-step <A>]
[--decoder-config <path>] [--error-indexes <path>]
```

Notes:

- `--frames` and `--max-iter` are required.
- `--alpha` is required unless you provide deterministic errors with `--error-indexes`.
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
- `candidate_selection` = `topk` | `graph` | `max_energy_checks`
- `labeling_strategy` = `ground_truth` | `rollout` | `corrective_mask`
- `candidate_k`
- `feature_set` or `feature_list`

Ready-to-use example configs are in `configs/decoder/`, including:

- `baseline.cfg`
- `pgdbf.cfg`
- `best_ml_escape.cfg`
- `best_ml_feedback.cfg`
- `collect_top6_ground_truth.cfg`

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

Set collect mode in cfg, for example `configs/decoder/collect_top6_ground_truth.cfg`.

Then run:

```powershell
.\GDBF.exe --frames 200000 --max-iter 30 --alpha 0 --nb-frames 0 --alpha-max 0.020 --alpha-min 0.012 --alpha-step 0.003 --decoder-config configs/decoder/collect_top6_ground_truth.cfg
```

Output dataset:

- `datasets/<code_name>/dataset.csv`

## 10) Train and Export Model Headers

Default training config:

- `training/configs/default.json`

Train:

```powershell
python training/train.py --config training/configs/default.json
```

Train for a specific code name:

```powershell
python training/train.py --code IRISC_dv4_R050_L54_N1296
```

Training exports headers into `model/`, including an active alias header used by runtime integration.

After training/export, rebuild `GDBF` so the latest generated headers are compiled in.

## 11) Plot FER Comparison

Example baseline vs ML plot:

```powershell
python visualize/plot_results.py --baseline results/wifin_r_1_2/baseline/simulation.res --ml results/wifin_r_1_2/ml/simulation.res --output visualize/plots/wifin_r_1_2/fer_comparison.png --title "FER Comparison: Baseline vs ML"
```

## 12) Deterministic Error Injection (Optional)

If you want fixed bit-error positions instead of random BSC errors, provide an index file:

```powershell
.\GDBF.exe --frames 1000 --max-iter 100 --code IRISC_P_dv3_R050_L54_N1296 --error-indexes codes/IRISC_P_dv3_R050_L54_N1296/error_indexes.txt --decoder-config configs/decoder/deterministic_errors_dv3.cfg
```

## 13) Minimal End-to-End 

1. Build `GDBF`.
2. Run baseline with `configs/decoder/baseline.cfg`.
3. Run collect mode with `configs/decoder/collect_top6_ground_truth.cfg`.
4. Train with `python training/train.py --config training/configs/default.json`.
5. Rebuild `GDBF`.
6. Run ML mode with `configs/decoder/best_ml_escape.cfg`.
7. Plot baseline vs ML with `visualize/plot_results.py`.
