from pathlib import Path

# Paths
DATASET_PATH = Path("data/dataset.csv")
MODEL_OUTPUT_PATH = Path("ml/as_model_weights.h")  # legacy alias
REF_OUTPUT_PATH = Path("ml/as_model_ref.h")
QUANTIZED_OUTPUT_PATH = Path("ml/as_model_quantized.h")

# Training
TEST_SPLIT = 0.2
EPOCHS = 300
BATCH_SIZE = 32
HIDDEN1 = 32
HIDDEN2 = 16
LEARNING_RATE = 0.001
THRESHOLD = 0.65
MIN_TRAIN_SAMPLES_WARNING = 50
OUTPUT_BITS = 6
