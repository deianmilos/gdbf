## Build

From `gdbf`:

```powershell
gcc -Iinclude -o GDBF src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c -lm
```

## Run

```powershell
./GDBF <NbMonteCarlo> <NbIter> <MatrixFile> <BaseMatrixPrefix> <ResultFile> <alpha> [other args...]
```

Example:

```powershell
./GDBF 100000000 100 ./input/wifin_r_1_2_Dform ./input/wifin_r_1_2_Base ./results/Res 0.01 100 1 0.01 0.001 0.001 
```

## Structure

- `include/common.h`: core data structures
- `include/matrix_io.h`, `src/matrix_io.c`: sparse/base matrix loading and cleanup
- `include/encoding.h`, `src/encoding.c`: Gaussian elimination and codeword encoding
- `include/channel.h`, `src/channel.c`: BSC noise model
- `include/decoder.h`, `src/decoder.c`: layered GDBF decode flow
- `include/stats.h`, `src/stats.c`: simulation statistics collection and reporting
- `src/main.c`: main
