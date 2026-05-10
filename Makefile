CC      = gcc
CFLAGS  = -Iinclude -O2 -lm
SRCS    = sim/main.c sim/matrix_io.c sim/encoding.c sim/channel.c sim/decoder.c sim/stats.c
ASRCS   = absorbing_sets/enum/as_enum_main.c absorbing_sets/enum/as_enum.c sim/matrix_io.c sim/encoding.c

all: GDBF GDBF_COLLECT GDBF_ML AS_ENUM

# Baseline GDBF decoder (no ML)
GDBF: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Data-collection build (writes dataset.csv at stagnation, no ML inference)
GDBF_COLLECT: $(SRCS)
	$(CC) $(CFLAGS) -DAS_COLLECT_MODE=1 -o $@ $^

# Backward-compatible alias
GDBF_TRAIN: GDBF_COLLECT

# ML-inference build (AS_ML_MODE=1: reads model/as_model_quantized.h, flips bits)
GDBF_ML: $(SRCS)
	$(CC) $(CFLAGS) -DAS_ML_MODE=1 -o $@ $^

# Absorbing-set enumerator
AS_ENUM: $(ASRCS)
	$(CC) $(CFLAGS) -Iabsorbing_sets/enum -o $@ $^

clean:
	del /Q GDBF.exe GDBF_COLLECT.exe GDBF_TRAIN.exe GDBF_ML.exe AS_ENUM.exe 2>NUL || true
