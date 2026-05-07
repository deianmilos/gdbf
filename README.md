# GDBF-ML: ML-Assisted LDPC Decoder

Layered GDBF decoder for QC-LDPC WiFiN rate-1/2 (N=648, M=324, Z=27), with an optional ML escape step for stuck states.

## Repository Layout

- `src/`: C simulation and decoder logic
- `include/`: C headers
- `input/`: QC-LDPC matrix files (`wifin_r_1_2_*`)
- `ml/`: training and quantized export pipeline (PyTorch)
- `data/`: generated training dataset (`dataset.csv`)
- `results/`: `.res` simulation outputs and ML summary CSV
- `visualize/`: FER plotting scripts and output figures

## Requirements

- GCC
- Python 3
- Python packages: `numpy`, `pandas`, `torch`, `matplotlib`

## Build Binaries

Run from repository root.

```powershell
gcc -Iinclude -O2 -DAS_ML_MODE=0 -DAS_TRAIN_MODE=0 -o GDBF src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c -lm
gcc -Iinclude -O2 -DAS_ML_MODE=0 -DAS_TRAIN_MODE=1 -o GDBF_TRAIN src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c -lm
gcc -Iinclude -O2 -DAS_ML_MODE=1 -o GDBF_ML src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c -lm
```

- `GDBF`: baseline only
- `GDBF_TRAIN`: dataset collection mode
- `GDBF_ML`: ML-assisted decoding

## Reproducible End-to-End Workflow

1. Activate environment

```powershell
conda activate gdbf-ml
```

2. Generate training dataset

```powershell
.\GDBF_TRAIN.exe 200000 100 input/wifin_r_1_2_Dform input/wifin_r_1_2_Base results/Res_collect 0.025 0 0
```

3. Train model and export quantized header

```powershell
conda run -n gdbf-ml python ml/train_model.py
```

4. Rebuild ML decoder after export

```powershell
gcc -Iinclude -O2 -DAS_ML_MODE=1 -o GDBF_ML src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c -lm
```

5. Run baseline and ML with the same sweep

```powershell
.\GDBF.exe    10000000 100 input/wifin_r_1_2_Dform input/wifin_r_1_2_Base results/Res_baseline_multi 0 200 0 0.010 0.003 0.001
.\GDBF_ML.exe 10000000 100 input/wifin_r_1_2_Dform input/wifin_r_1_2_Base results/Res_ml_multi       0 200 0 0.010 0.003 0.001
```

6. Plot FER comparison

```powershell
conda run -n gdbf-ml python visualize/plot_results.py --baseline results/Res_baseline_multi.res --ml results/Res_ml_multi.res --output visualize/plots/fer_comparison_multi.png --title "FER Comparison: Baseline vs ML"
```

## Decoder CLI Arguments

Binary format:

```text
<frames> <maxIter> <DformFile> <BaseFile> <outPrefix> <alpha> <NBframes> <reserved0> <alpha_max> <alpha_min> <alpha_step>
```

- `frames`: max Monte Carlo frames per alpha point
- `maxIter`: max decoding iterations per frame
- `outPrefix`: output prefix (`.res` is appended)
- `alpha`: single-point alpha (used when sweep args are omitted)
- `NBframes`: stop after this many frame errors per alpha (`0` means no early stop)
- `reserved0`: keep `0` (legacy positional argument kept for compatibility)
- `alpha_max`, `alpha_min`, `alpha_step`: sweep from high alpha down to low alpha

Examples:

- single alpha:

```powershell
.\GDBF.exe 10000000 100 input/wifin_r_1_2_Dform input/wifin_r_1_2_Base results/Res_baseline_single 0.005 200 0
```

- multi alpha:

```powershell
.\GDBF.exe 10000000 100 input/wifin_r_1_2_Dform input/wifin_r_1_2_Base results/Res_baseline_multi 0 200 0 0.010 0.003 0.001
```

## How ML Is Used During Decoding

When the decoder detects a stuck phase (stagnation or oscillation), it:

1. selects 6 candidate bits,
2. builds quantized features from energy and disagreement,
3. calls `as_quantized_predict()` from `ml/as_model_quantized.h`,
4. applies predicted flips and resumes normal GDBF iterations.

If no escape is applied, baseline max-energy flips continue.

## Outputs You Should Expect

1. FER result files:

- `results/Res_baseline_multi.res`
- `results/Res_ml_multi.res`

2. Per-alpha ML effectiveness CSV:

- `results/ml_outcome_summary_per_alpha.csv`

This CSV contains one row per alpha point, with:

- frame totals and clean decodes,
- baseline-only decoded frames,
- trap frames needing ML,
- ML-effective and ML-not-effective trap frames,
- `stagnation_events`, `ml_calls`, `ml_escapes`.


