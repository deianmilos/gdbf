from config import HIDDEN1, HIDDEN2, THRESHOLD


def _header_guard(output_path):
  stem = output_path.replace("\\", "/").split("/")[-1].replace(".", "_")
  return stem.upper()


def _array_to_c_float(arr, name):
    flat = arr.flatten()
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 8):
        chunk = flat[i : i + 8]
        lines.append("  " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def export_to_c_header(model, mean, std, output_path):
    state = model.state_dict()
    w1 = state["0.weight"].numpy()
    b1 = state["0.bias"].numpy()
    w2 = state["2.weight"].numpy()
    b2 = state["2.bias"].numpy()
    w3 = state["4.weight"].numpy()
    b3 = state["4.bias"].numpy()

    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* Auto-generated AS escape model weights */\n")
        f.write(f"/* Input: {w1.shape[1]} features, Output: 6 flip probabilities */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")

        f.write(f"#define AS_MODEL_INPUT_SIZE {w1.shape[1]}\n")
        f.write(f"#define AS_MODEL_HIDDEN1 {HIDDEN1}\n")
        f.write(f"#define AS_MODEL_HIDDEN2 {HIDDEN2}\n")
        f.write("#define AS_MODEL_OUTPUT_SIZE 6\n")
        f.write(f"#define AS_MODEL_THRESHOLD {THRESHOLD:.2f}f\n\n")

        f.write(_array_to_c_float(mean, "as_model_mean") + "\n\n")
        f.write(_array_to_c_float(std, "as_model_std") + "\n\n")
        f.write(_array_to_c_float(w1, "as_model_w1") + "\n\n")
        f.write(_array_to_c_float(b1, "as_model_b1") + "\n\n")
        f.write(_array_to_c_float(w2, "as_model_w2") + "\n\n")
        f.write(_array_to_c_float(b2, "as_model_b2") + "\n\n")
        f.write(_array_to_c_float(w3, "as_model_w3") + "\n\n")
        f.write(_array_to_c_float(b3, "as_model_b3") + "\n\n")

        f.write(
            """
/* Run inference: returns 1 if any bit should be flipped */
static int as_model_predict(const float *features, int *flip_mask)
{
  float h1[AS_MODEL_HIDDEN1];
  float h2[AS_MODEL_HIDDEN2];
  float out[AS_MODEL_OUTPUT_SIZE];
  int i, j;
  int any_flip = 0;

  /* Normalize input */
  float normed[AS_MODEL_INPUT_SIZE];
  for (i = 0; i < AS_MODEL_INPUT_SIZE; i++) {
    normed[i] = (features[i] - as_model_mean[i]) / as_model_std[i];
  }

  /* Layer 1: Linear + ReLU */
  for (i = 0; i < AS_MODEL_HIDDEN1; i++) {
    float sum = as_model_b1[i];
    for (j = 0; j < AS_MODEL_INPUT_SIZE; j++) {
      sum += as_model_w1[i * AS_MODEL_INPUT_SIZE + j] * normed[j];
    }
    h1[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  /* Layer 2: Linear + ReLU */
  for (i = 0; i < AS_MODEL_HIDDEN2; i++) {
    float sum = as_model_b2[i];
    for (j = 0; j < AS_MODEL_HIDDEN1; j++) {
      sum += as_model_w2[i * AS_MODEL_HIDDEN1 + j] * h1[j];
    }
    h2[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  /* Layer 3: Linear + Sigmoid */
  for (i = 0; i < AS_MODEL_OUTPUT_SIZE; i++) {
    float sum = as_model_b3[i];
    for (j = 0; j < AS_MODEL_HIDDEN2; j++) {
      sum += as_model_w3[i * AS_MODEL_HIDDEN2 + j] * h2[j];
    }
    out[i] = 1.0f / (1.0f + expf(-sum));
    flip_mask[i] = (out[i] > AS_MODEL_THRESHOLD) ? 1 : 0;
    if (flip_mask[i]) any_flip = 1;
  }

  return any_flip;
}
"""
        )
        f.write(f"\n#endif /* {guard} */\n")

    print(f"\nExported model to {output_path}")
    print(f"  Parameters: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size}")


def export_ref_header(w1, b1, w2, b2, w3, b3, mean, std, output_path):
    input_size = w1.shape[1]
    hidden1 = w1.shape[0]
    hidden2 = w2.shape[0]
    output_size = w3.shape[0]

    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* AS escape model - float reference (full precision) */\n")
        f.write(f"/* Input: {input_size} features  Hidden: {hidden1}->{hidden2}  Output: {output_size} */\n")
        f.write("/* Use this for validation and as the default software decoder. */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define AS_MODEL_INPUT_SIZE  {input_size}\n")
        f.write(f"#define AS_MODEL_HIDDEN1     {hidden1}\n")
        f.write(f"#define AS_MODEL_HIDDEN2     {hidden2}\n")
        f.write(f"#define AS_MODEL_OUTPUT_SIZE {output_size}\n")
        f.write(f"#define AS_MODEL_THRESHOLD   {THRESHOLD:.2f}f\n\n")
        f.write(_array_to_c_float(mean, "as_ref_mean") + "\n\n")
        f.write(_array_to_c_float(std, "as_ref_std") + "\n\n")
        f.write(_array_to_c_float(w1, "as_ref_w1") + "\n\n")
        f.write(_array_to_c_float(b1, "as_ref_b1") + "\n\n")
        f.write(_array_to_c_float(w2, "as_ref_w2") + "\n\n")
        f.write(_array_to_c_float(b2, "as_ref_b2") + "\n\n")
        f.write(_array_to_c_float(w3, "as_ref_w3") + "\n\n")
        f.write(_array_to_c_float(b3, "as_ref_b3") + "\n\n")
        f.write(
            """\
/* Run float inference.  Returns 1 if any bit should be flipped. */
static int as_ref_predict(const float *features, int *flip_mask)
{
  float h1[AS_MODEL_HIDDEN1], h2[AS_MODEL_HIDDEN2], out[AS_MODEL_OUTPUT_SIZE];
  float normed[AS_MODEL_INPUT_SIZE];
  int i, j, any_flip = 0;

  for (i = 0; i < AS_MODEL_INPUT_SIZE; i++)
    normed[i] = (features[i] - as_ref_mean[i]) / as_ref_std[i];

  for (i = 0; i < AS_MODEL_HIDDEN1; i++) {
    float s = as_ref_b1[i];
    for (j = 0; j < AS_MODEL_INPUT_SIZE; j++) s += as_ref_w1[i*AS_MODEL_INPUT_SIZE+j] * normed[j];
    h1[i] = s > 0.0f ? s : 0.0f;
  }
  for (i = 0; i < AS_MODEL_HIDDEN2; i++) {
    float s = as_ref_b2[i];
    for (j = 0; j < AS_MODEL_HIDDEN1; j++) s += as_ref_w2[i*AS_MODEL_HIDDEN1+j] * h1[j];
    h2[i] = s > 0.0f ? s : 0.0f;
  }
  for (i = 0; i < AS_MODEL_OUTPUT_SIZE; i++) {
    float s = as_ref_b3[i];
    for (j = 0; j < AS_MODEL_HIDDEN2; j++) s += as_ref_w3[i*AS_MODEL_HIDDEN2+j] * h2[j];
    out[i] = 1.0f / (1.0f + expf(-s));
    flip_mask[i] = (out[i] > AS_MODEL_THRESHOLD) ? 1 : 0;
    if (flip_mask[i]) any_flip = 1;
  }
  return any_flip;
}
"""
        )
        f.write(f"\n#endif /* {guard} */\n")

    print(f"Exported float reference header -> {output_path}")


def export_weights_alias_header(output_path, ref_header_name):
    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* Compatibility alias: use float reference implementation. */\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#include \"{ref_header_name}\"\n\n")
        f.write("#define as_model_predict as_ref_predict\n\n")
        f.write(f"#endif /* {guard} */\n")

    print(f"Exported compatibility weights alias -> {output_path}")
