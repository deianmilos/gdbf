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


def export_quantized_header(w1, b1, w2, b2, w3, b3, mean, std, output_path):
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
    w3q, sw3 = quantize(w3)

    b1q = np.clip(np.round(b1f * sw1), -(2**30), 2**30).astype(np.int32)
    b2q = np.clip(np.round(b2 * sw2), -(2**30), 2**30).astype(np.int32)
    b3q = np.clip(np.round(b3 * sw3), -(2**30), 2**30).astype(np.int32)

    shift1 = max(0, int(math.floor(math.log2(sw1))))
    shift2 = max(0, int(math.floor(math.log2(sw2))))

    input_size = w1.shape[1]
    hidden1 = w1.shape[0]
    hidden2 = w2.shape[0]
    output_size = w3.shape[0]

    guard = _header_guard(output_path)
    with open(output_path, "w") as f:
        f.write("/* AS escape model - quantized fixed-point version */\n")
        f.write(f"/* Input:  {input_size} raw features (no normalization at runtime) */\n")
        f.write(f"/* Hidden: {hidden1} -> {hidden2}  (int8 weights, int32 accumulators) */\n")
        f.write(f"/* Output: {output_size} raw int32 scores  (positive = predict error) */\n")
        f.write("/* No float, no exp, no division - suitable for FPGA / MCU. */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define AS_QUANTIZED_INPUT_SIZE  {input_size}\n")
        f.write(f"#define AS_QUANTIZED_HIDDEN1     {hidden1}\n")
        f.write(f"#define AS_QUANTIZED_HIDDEN2     {hidden2}\n")
        f.write(f"#define AS_QUANTIZED_OUTPUT_SIZE {output_size}\n\n")
        f.write("/* Right-shift applied after each hidden layer to prevent overflow\n")
        f.write("   in the next layer's int8*int8 dot product. */\n")
        f.write(f"#define AS_QUANTIZED_SHIFT1 {shift1}\n")
        f.write(f"#define AS_QUANTIZED_SHIFT2 {shift2}\n\n")
        f.write("/* Quantization scales (float reference only, not used at runtime):\n")
        f.write(f"   w1 scale = {sw1:.4f}  (1 LSB ~ {1/sw1:.6f} float units)\n")
        f.write(f"   w2 scale = {sw2:.4f}  (1 LSB ~ {1/sw2:.6f} float units)\n")
        f.write(f"   w3 scale = {sw3:.4f}  (1 LSB ~ {1/sw3:.6f} float units)\n")
        f.write("   Normalization folded into w1/b1 - raw features used directly. */\n\n")
        f.write(_arr8(w1q, "as_quantized_w1") + "\n\n")
        f.write(_arr32(b1q, "as_quantized_b1") + "\n\n")
        f.write(_arr8(w2q, "as_quantized_w2") + "\n\n")
        f.write(_arr32(b2q, "as_quantized_b2") + "\n\n")
        f.write(_arr8(w3q, "as_quantized_w3") + "\n\n")
        f.write(_arr32(b3q, "as_quantized_b3") + "\n\n")
        f.write(
            """\
/*
 * Fixed-point inference.
 * features[]: E0-E5 raw bitEnergy integers, F0-F5 in {0,1}
 */
static int as_quantized_predict(const int8_t *feat, int *flip_mask)
{
  int32_t h1[AS_QUANTIZED_HIDDEN1];
  int32_t h2[AS_QUANTIZED_HIDDEN2];
  int i, j, n_flips = 0;

  for (i = 0; i < AS_QUANTIZED_HIDDEN1; i++) {
    int32_t acc = as_quantized_b1[i];
    for (j = 0; j < AS_QUANTIZED_INPUT_SIZE; j++)
      acc += (int32_t)as_quantized_w1[i * AS_QUANTIZED_INPUT_SIZE + j] * (int32_t)feat[j];
    acc >>= AS_QUANTIZED_SHIFT1;
    h1[i] = acc > 0 ? acc : 0;
  }

  for (i = 0; i < AS_QUANTIZED_HIDDEN2; i++) {
    int32_t acc = as_quantized_b2[i];
    for (j = 0; j < AS_QUANTIZED_HIDDEN1; j++)
      acc += (int32_t)as_quantized_w2[i * AS_QUANTIZED_HIDDEN1 + j] * h1[j];
    acc >>= AS_QUANTIZED_SHIFT2;
    h2[i] = acc > 0 ? acc : 0;
  }

  for (i = 0; i < AS_QUANTIZED_OUTPUT_SIZE; i++) {
    int32_t acc = as_quantized_b3[i];
    for (j = 0; j < AS_QUANTIZED_HIDDEN2; j++)
      acc += (int32_t)as_quantized_w3[i * AS_QUANTIZED_HIDDEN2 + j] * h2[j];
    flip_mask[i] = (acc > 0) ? 1 : 0;
    if (flip_mask[i]) n_flips++;
  }
  return n_flips;
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

    w1_err = float(np.abs(w1f - w1q.astype(np.float32) / sw1).mean())
    w2_err = float(np.abs(w2 - w2q.astype(np.float32) / sw2).mean())
    w3_err = float(np.abs(w3 - w3q.astype(np.float32) / sw3).mean())

    print(f"Exported quantized header -> {output_path}")
    print(f"  int8 weight bytes : {w1q.size + w2q.size + w3q.size}")
    print(f"  int32 bias bytes  : {(b1q.size + b2q.size + b3q.size) * 4}")
    print(f"  w1 scale={sw1:.2f}  w2 scale={sw2:.2f}  w3 scale={sw3:.2f}")
    print(f"  SHIFT1={shift1}  SHIFT2={shift2}")
    print(f"  Mean weight quant error: w1={w1_err:.5f}  w2={w2_err:.5f}  w3={w3_err:.5f}")
