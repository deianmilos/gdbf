import math

import numpy as np


def _header_guard(output_path):
    stem = output_path.replace("\\", "/").split("/")[-1].replace(".", "_")
    return stem.upper()


def _arr8(a, name):
    flat = a.flatten().astype(np.int8)
    rows = [f"static const int8_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 16):
        rows.append("  " + ", ".join(str(int(v)) for v in flat[i : i + 16]) + ",")
    rows.append("};")
    return "\n".join(rows)


def _arr32(a, name):
    flat = a.flatten().astype(np.int32)
    rows = [f"static const int32_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 8):
        rows.append("  " + ", ".join(str(int(v)) for v in flat[i : i + 8]) + ",")
    rows.append("};")
    return "\n".join(rows)


def export_quantized_header(params, mean, std, output_bits, output_path):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    w_flip = params["w_flip"]
    b_flip = params["b_flip"]

    mask_logit_threshold = int(params.get("mask_logit_threshold", 0))

    eps = 1e-8
    std_safe = np.where(np.abs(std) < eps, 1.0, std)
    w1f = w1 / std_safe[np.newaxis, :]
    b1f = b1 - (w1 / std_safe[np.newaxis, :] * mean[np.newaxis, :]).sum(axis=1)

    def quantize(arr):
        amax = np.abs(arr).max()
        if amax < eps:
            return np.zeros_like(arr, dtype=np.int8), 1.0
        scale = 127.0 / amax
        return np.clip(np.round(arr * scale), -127, 127).astype(np.int8), scale

    w1q, sw1 = quantize(w1f)
    w2q, sw2 = quantize(w2)
    w_flip_q, sw_flip = quantize(w_flip)

    b1q = np.clip(np.round(b1f * sw1), -(2**30), 2**30).astype(np.int32)
    b2q = np.clip(np.round(b2 * sw2), -(2**30), 2**30).astype(np.int32)
    b_flip_q = np.clip(np.round(b_flip * sw_flip), -(2**30), 2**30).astype(np.int32)

    shift1 = max(0, int(math.floor(math.log2(sw1))))
    shift2 = max(0, int(math.floor(math.log2(sw2))))

    input_size = w1.shape[1]
    hidden1 = w1.shape[0]
    hidden2 = w2.shape[0]

    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* AS escape model - quantized fixed-point version */\n")
        f.write(f"/* Input:  {input_size} raw features (no normalization at runtime) */\n")
        f.write(f"/* Hidden: {hidden1} -> {hidden2}  (int8 weights, int32 accumulators) */\n")
        f.write(f"/* Output head: flip logits={output_bits} (mask-only variant) */\n")
        f.write("/* No float, no exp, no division - suitable for FPGA / MCU. */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define AS_QUANTIZED_INPUT_SIZE  {input_size}\n")
        f.write(f"#define AS_QUANTIZED_HIDDEN1     {hidden1}\n")
        f.write(f"#define AS_QUANTIZED_HIDDEN2     {hidden2}\n")
        f.write(f"#define AS_QUANTIZED_OUTPUT_SIZE {output_bits}\n")
        f.write("#define AS_QUANTIZED_BUDGET_SIZE 1\n")
        f.write("#define AS_QUANTIZED_VARIANT_MASK_ONLY 1\n")
        f.write(f"#define AS_QUANTIZED_MASK_LOGIT_THRESHOLD {mask_logit_threshold}\n\n")
        f.write(f"#define AS_QUANTIZED_SHIFT1 {shift1}\n")
        f.write(f"#define AS_QUANTIZED_SHIFT2 {shift2}\n\n")

        f.write(_arr8(w1q, "as_quantized_w1") + "\n\n")
        f.write(_arr32(b1q, "as_quantized_b1") + "\n\n")
        f.write(_arr8(w2q, "as_quantized_w2") + "\n\n")
        f.write(_arr32(b2q, "as_quantized_b2") + "\n\n")
        f.write(_arr8(w_flip_q, "as_quantized_w_flip") + "\n\n")
        f.write(_arr32(b_flip_q, "as_quantized_b_flip") + "\n\n")
        f.write(
            """\
/* Mask-only prediction: each candidate is predicted independently. */
static int as_quantized_predict(const int8_t *feat, int *flip_mask)
{
  int32_t h1[AS_QUANTIZED_HIDDEN1];
  int32_t h2[AS_QUANTIZED_HIDDEN2];
  int32_t out_flip[AS_QUANTIZED_OUTPUT_SIZE];
  int i, j;

  for (i = 0; i < AS_QUANTIZED_HIDDEN1; i++) {
    int32_t acc = as_quantized_b1[i];
    for (j = 0; j < AS_QUANTIZED_INPUT_SIZE; j++) {
      acc += (int32_t)as_quantized_w1[i * AS_QUANTIZED_INPUT_SIZE + j] *
             (int32_t)feat[j];
    }
    acc >>= AS_QUANTIZED_SHIFT1;
    h1[i] = (acc > 0) ? acc : 0;
  }

  for (i = 0; i < AS_QUANTIZED_HIDDEN2; i++) {
    int32_t acc = as_quantized_b2[i];
    for (j = 0; j < AS_QUANTIZED_HIDDEN1; j++) {
      acc += (int32_t)as_quantized_w2[i * AS_QUANTIZED_HIDDEN1 + j] * h1[j];
    }
    acc >>= AS_QUANTIZED_SHIFT2;
    h2[i] = (acc > 0) ? acc : 0;
  }

  for (i = 0; i < AS_QUANTIZED_OUTPUT_SIZE; i++) {
    int32_t acc = as_quantized_b_flip[i];
    for (j = 0; j < AS_QUANTIZED_HIDDEN2; j++) {
      acc += (int32_t)as_quantized_w_flip[i * AS_QUANTIZED_HIDDEN2 + j] * h2[j];
    }
    out_flip[i] = acc;
  }

  for (i = 0; i < AS_QUANTIZED_OUTPUT_SIZE; i++) {
    flip_mask[i] = (out_flip[i] > AS_QUANTIZED_MASK_LOGIT_THRESHOLD) ? 1 : 0;
  }

  return 1;
}

static void as_quantized_encode_features(const float *features, int8_t *feat_int)
{
  int i;
  for (i = 0; i < AS_QUANTIZED_INPUT_SIZE; i++) {
    float v = features[i];
    if (v > 127.0f) v = 127.0f;
    if (v < -127.0f) v = -127.0f;
    feat_int[i] = (int8_t)v;
  }
}
"""
        )

        f.write(f"\n#endif /* {guard} */\n")

    print(f"Exported quantized header -> {output_path}")
    print("  variant: mask_only")
    print(f"  int8 weight bytes : {w1q.size + w2q.size + w_flip_q.size}")
    print(f"  int32 bias bytes  : {(b1q.size + b2q.size + b_flip_q.size) * 4}")
    print(f"  w1 scale={sw1:.2f}  w2 scale={sw2:.2f}  w_flip scale={sw_flip:.2f}")
