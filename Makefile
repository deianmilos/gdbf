CC      = gcc
CFLAGS  = -Iinclude -O2 -lm
SRCS    = src/main.c src/matrix_io.c src/encoding.c src/channel.c src/decoder.c src/stats.c
ASRCS   = src/as_enum_main.c src/as_enum.c src/matrix_io.c src/encoding.c

all: GDBF GDBF_TRAIN GDBF_ML AS_ENUM

# Baseline GDBF decoder (no ML)
GDBF: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Data-collection build (AS_ML_MODE=0: writes data/dataset.csv at stagnation)
GDBF_TRAIN: $(SRCS)
	$(CC) $(CFLAGS) -DAS_ML_MODE=0 -o $@ $^

# ML-inference build (AS_ML_MODE=1: reads ml/as_model_weights.h, flips bits)
GDBF_ML: $(SRCS)
	$(CC) $(CFLAGS) -DAS_ML_MODE=1 -o $@ $^

# Absorbing-set enumerator
AS_ENUM: $(ASRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	del /Q GDBF.exe GDBF_TRAIN.exe GDBF_ML.exe AS_ENUM.exe 2>NUL || true
