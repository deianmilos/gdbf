# GDBF-ML: ML-Assisted LDPC Decoder

Layered GDBF decoder for QC-LDPC WiFiN rate-1/2 (N=648, M=324, Z=27), with an optional ML escape step for stuck states.

## Repository Layout

- `sim/`: C simulation and decoder logic
- `include/`: C headers
- `codes/`: QC-LDPC matrix files, one subdirectory per code (e.g. `wifin_r_1_2/`, `IRISC_dv3_R050_L54_N1296/`)
- `model/`: generated C headers embedded by the decoder (code-specific quantized + float reference)
- `training/`: Python training and export pipeline (PyTorch)
- `datasets/`: generated training datasets, scoped per code (`datasets/<code>/dataset.csv`)
- `results/`: simulation outputs, organized as `results/<code>/<run_type>/` where `<run_type>` is `baseline`, `ml`, or `collect`
- `visualize/`: FER plotting scripts and output figures

## Requirements

- GCC
- Python 3
- Python packages: `numpy`, `pandas`, `torch`, `matplotlib`

## Build Binaries

Run from repository root.

```powershell
gcc -Iinclude -O2                -o GDBF       sim/main.c sim/matrix_io.c sim/encoding.c sim/channel.c sim/decoder.c sim/stats.c -lm
gcc -Iinclude -O2 -DAS_COLLECT_MODE=1 -o GDBF_COLLECT sim/main.c sim/matrix_io.c sim/encoding.c sim/channel.c sim/decoder.c sim/stats.c -lm
gcc -Iinclude -O2 -DAS_ML_MODE=1    -o GDBF_ML    sim/main.c sim/matrix_io.c sim/encoding.c sim/channel.c sim/decoder.c sim/stats.c -lm
```

- `GDBF`: baseline only
- `GDBF_COLLECT`: dataset collection mode
- `GDBF_ML`: ML-assisted decoding

## Reproducible End-to-End Workflow

1. Activate environment

```powershell
conda activate gdbf-ml
```

2. Generate training dataset

Single alpha:
```powershell
.\\GDBF_COLLECT.exe --frames 200000 --max-iter 100 --code wifin_r_1_2 --alpha 0.025
```

Multi-alpha (appends to same dataset):
```powershell
.\\GDBF_COLLECT.exe --frames 200000 --max-iter 100 --code wifin_r_1_2 --alpha 0 --nb-frames 0 --alpha-max 0.010 --alpha-min 0.003 --alpha-step 0.001
```

Outputs to `results/wifin_r_1_2/collect/simulation.res` and `datasets/wifin_r_1_2/dataset.csv`.

3. Train model and export model headers

Default code (wifin_r_1_2):
```powershell
conda run -n gdbf-ml python training/train.py
```

Specify a different code:
```powershell
conda run -n gdbf-ml python training/train.py --code IRISC_dv4_R050_L54_N1296
```

Training exports exactly two model headers (code-specific), for example:

- `model/as_model_wifin_r_1_2_quantized.h`
- `model/as_model_wifin_r_1_2_ref.h`

The decoder includes a code-specific quantized header in `sim/decoder.c`.
If you train another code, update that include to the matching `as_model_<code>_quantized.h` file.

4. Rebuild ML decoder after export

```powershell
gcc -Iinclude -O2 -DAS_ML_MODE=1 -o GDBF_ML sim/main.c sim/matrix_io.c sim/encoding.c sim/channel.c sim/decoder.c sim/stats.c -lm
```

5. Run baseline and ML with the same sweep

```powershell
.\\GDBF.exe    --frames 10000000 --max-iter 100 --code wifin_r_1_2 --alpha 0 --nb-frames 100 --alpha-max 0.010 --alpha-min 0.003 --alpha-step 0.001
.\\GDBF_ML.exe --frames 10000000 --max-iter 100 --code wifin_r_1_2 --alpha 0 --nb-frames 100 --alpha-max 0.010 --alpha-min 0.003 --alpha-step 0.001
```
Results land in `results/wifin_r_1_2/baseline/` and `results/wifin_r_1_2/ml/` respectively.

6. Plot FER comparison

```powershell
conda run -n gdbf-ml python visualize/plot_results.py --baseline results/wifin_r_1_2/baseline/simulation.res --ml results/wifin_r_1_2/ml/simulation.res --output visualize/plots/wifin_r_1_2/fer_comparison.png --title "FER Comparison: Baseline vs ML"
```

## Decoder CLI Arguments

Binary format (recommended named arguments):

```text
--frames <N> --max-iter <N> --code <CodeName> --alpha <A> [--nb-frames <N>] [--alpha-max <A>] [--alpha-min <A>] [--alpha-step <A>]
```

- `frames`: max Monte Carlo frames per alpha point
- `maxIter`: max decoding iterations per frame
- `CodeName`: LDPC code folder name under `codes/` (e.g. `wifin_r_1_2`); the simulator auto-loads `codes/<CodeName>/<CodeName>_Dform` and `codes/<CodeName>/<CodeName>_Base`
- `alpha`: single-point crossover probability (used when sweep args are omitted)
- `NBframes`: stop after this many frame errors per alpha (`0` = no early stop)
- `alpha_max`, `alpha_min`, `alpha_step`: sweep from high alpha down to low alpha

Legacy positional mode is still supported for compatibility:

```text
<frames> <maxIter> <CodeName> <alpha> [NBframes [alpha_max [alpha_min [alpha_step]]]]
```

Result files are written automatically to `results/<code>/<run_type>/simulation.res`. No output path argument is needed.

Examples:

- single alpha:

```powershell
.\\GDBF.exe --frames 10000000 --max-iter 100 --code wifin_r_1_2 --alpha 0.005 --nb-frames 200
```

- multi alpha:

```powershell
.\\GDBF.exe --frames 10000000 --max-iter 100 --code wifin_r_1_2 --alpha 0 --nb-frames 200 --alpha-max 0.010 --alpha-min 0.003 --alpha-step 0.001
```

## How ML Is Used During Decoding

When the decoder detects a stuck phase (stagnation or oscillation), it:

1. selects 6 candidate bits,
2. builds quantized features from energy and disagreement,
3. calls `as_quantized_predict()` from the selected code-specific quantized model header,
4. applies predicted flips and resumes normal GDBF iterations.

If no escape is applied, baseline max-energy flips continue.

## Outputs You Should Expect

1. FER result file (one per run, path depends on code and run type):

- `results/<code>/baseline/simulation.res`
- `results/<code>/ml/simulation.res`
- `results/<code>/collect/simulation.res`

2. Per-alpha ML effectiveness CSV (written alongside the result file):

- `results/<code>/ml/ml_outcome_summary.csv`

This CSV contains one row per alpha point, with:

- frame totals and clean decodes,
- baseline-only decoded frames,
- trap frames needing ML,
- ML-effective and ML-not-effective trap frames,
- `stagnation_events`, `ml_calls`, `ml_escapes`.


