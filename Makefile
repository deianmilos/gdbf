CC      = gcc
CFLAGS  = -Iinclude \
          -Iinclude/common \
          -Iinclude/core \
          -Iinclude/feedback \
          -Iinclude/ml \
          -Iinclude/framework \
          -Iinclude/config \
          -Iinclude/stats \
          -O2 -Wno-unused-variable -lm

SRCS    = src/app/main.c \
          src/app/args_and_config.c \
          src/app/logging.c \
          src/core/matrix_io.c \
          src/core/encoding.c \
          src/core/channel.c \
          src/core/decoder.c \
          src/core/stagnation_detection.c \
          src/feedback/decoder_feedback_shift.c \
          src/feedback/decoder_receiver.c \
          src/feedback/feedback_round.c \
          src/ml/decoder_ml.c \
          src/ml/decoder_perturb.c \
          src/ml/candidate_selection.c \
          src/ml/feature_extractor.c \
          src/ml/labeling_strategy.c \
          src/ml/dataset_writer.c \
          src/ml/ml_round.c \
          src/config/decoder_config.c \
          src/stats/stats.c \
          src/framework/frame_setup.c \
          src/framework/decoder_framework.c

ASRCS   = absorbing_sets/enum/as_enum_main.c \
          absorbing_sets/enum/as_enum.c \
          src/core/matrix_io.c \
          src/core/encoding.c

all: GDBF GDBF_COLLECT AS_ENUM

# Baseline GDBF decoder (no ML)
GDBF: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Data-collection build (writes dataset.csv at stagnation, no ML inference)
GDBF_COLLECT: $(SRCS)
	$(CC) $(CFLAGS) -DAS_COLLECT_MODE=1 -o $@ $^

# Backward-compatible alias
GDBF_TRAIN: GDBF_COLLECT

# ML is runtime-selected via decoder config (`decoder_type = ml`).
# Keep this alias so old commands still work.
GDBF_ML: GDBF

# Absorbing-set enumerator
AS_ENUM: $(ASRCS)
	$(CC) $(CFLAGS) -Iabsorbing_sets/enum -o $@ $^

clean:
	del /Q GDBF.exe GDBF_COLLECT.exe GDBF_TRAIN.exe GDBF_ML.exe AS_ENUM.exe 2>NUL || true

# ============================================================================
# ML Training (Python: requires conda environment gdbf-ml)
# ============================================================================
# Train and export quantized + float reference model headers:
#   conda run -n gdbf-ml python training/train.py --config configs/ml/default.json
# or:
#   conda activate gdbf-ml && python training/train.py --config configs/ml/default.json
#
# Generated headers are placed in model/, include active alias as_model_active_quantized.h
# After training, rebuild GDBF with: make clean && make
