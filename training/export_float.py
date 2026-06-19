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


def _write_ref_header_mask_only(
    f,
    mean,
    std,
    w1,
    b1,
    w2,
    b2,
    w_flip,
    b_flip,
    output_bits,
    mask_logit_threshold,
):
    input_size = w1.shape[1]
    hidden1 = w1.shape[0]
    hidden2 = w2.shape[0]

    f.write(f"#define AS_MODEL_INPUT_SIZE  {input_size}\n")
    f.write(f"#define AS_MODEL_HIDDEN1     {hidden1}\n")
    f.write(f"#define AS_MODEL_HIDDEN2     {hidden2}\n")
    f.write(f"#define AS_MODEL_OUTPUT_SIZE {output_bits}\n")
    f.write(f"#define AS_MODEL_MASK_LOGIT_THRESHOLD {float(mask_logit_threshold):.6f}f\n")
    f.write(f"#define AS_MODEL_THRESHOLD   {float(mask_logit_threshold):.6f}f\n\n")

    f.write(_array_to_c_float(mean, "as_ref_mean") + "\n\n")
    f.write(_array_to_c_float(std, "as_ref_std") + "\n\n")
    f.write(_array_to_c_float(w1, "as_ref_w1") + "\n\n")
    f.write(_array_to_c_float(b1, "as_ref_b1") + "\n\n")
    f.write(_array_to_c_float(w2, "as_ref_w2") + "\n\n")
    f.write(_array_to_c_float(b2, "as_ref_b2") + "\n\n")
    f.write(_array_to_c_float(w_flip, "as_ref_w3") + "\n\n")
    f.write(_array_to_c_float(b_flip, "as_ref_b3") + "\n\n")

    f.write(
        """\
/* Run float inference. Returns direct mask prediction. */
static int as_ref_predict(const float *features, int *flip_mask)
{
  float h1[AS_MODEL_HIDDEN1], h2[AS_MODEL_HIDDEN2], out[AS_MODEL_OUTPUT_SIZE];
  float normed[AS_MODEL_INPUT_SIZE];
  int i, j;

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
    out[i] = s;
    flip_mask[i] = (s > AS_MODEL_MASK_LOGIT_THRESHOLD) ? 1 : 0;
  }

  return 1;
}
"""
    )


def export_ref_header(params, mean, std, output_bits, output_path):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    w_flip = params["w_flip"]
    b_flip = params["b_flip"]
    mask_logit_threshold = params.get("mask_logit_threshold", 0.0)

    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* AS escape model - float reference (full precision) */\n")
        f.write("/* Variant: direct mask-only decoding. */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")

        _write_ref_header_mask_only(
            f,
            mean,
            std,
            w1,
            b1,
            w2,
            b2,
            w_flip,
            b_flip,
            output_bits,
            mask_logit_threshold,
        )

        f.write(f"\n#endif /* {guard} */\n")

    print(f"Exported float reference header -> {output_path}")


def export_to_c_header(model, mean, std, output_path):
    state = model.state_dict()
    params = {
        "w1": state["fc1.weight"].numpy(),
        "b1": state["fc1.bias"].numpy(),
        "w2": state["fc2.weight"].numpy(),
        "b2": state["fc2.bias"].numpy(),
        "w_flip": state["flip_head.weight"].numpy(),
        "b_flip": state["flip_head.bias"].numpy(),
    }

    export_ref_header(params, mean, std, params["w_flip"].shape[0], output_path)


def export_weights_alias_header(output_path, ref_header_name):
    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* Compatibility alias: use float reference implementation. */\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#include \"{ref_header_name}\"\n\n")
        f.write("#define as_model_predict as_ref_predict\n\n")
        f.write(f"#endif /* {guard} */\n")

    print(f"Exported compatibility weights alias -> {output_path}")
