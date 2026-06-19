from pathlib import Path

# Legacy helpers.
# Training hyperparameters are defined in JSON config files under configs/ml.
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
