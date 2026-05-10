from pathlib import Path

# Defaults
DEFAULT_CODE_NAME = "wifin_r_1_2"
MODEL_DIR = Path("model")


def dataset_path(code_name: str) -> Path:
	return Path(f"datasets/{code_name}/dataset.csv")


def infer_code_name(dataset_path: Path) -> str:
	code_name = dataset_path.parent.name.strip()
	return code_name if code_name else DEFAULT_CODE_NAME


def model_output_paths(code_name: str):
	return {
		"ref_code": MODEL_DIR / f"as_model_{code_name}_ref.h",
		"quantized_code": MODEL_DIR / f"as_model_{code_name}_quantized.h",
	}

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
