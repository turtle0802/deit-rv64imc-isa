import torch
import numpy as np


# ============================================================
# Configuration
# ============================================================

FP32_WORKLOAD_PATH = "results/fc1_fp32.npz"
W8A8_RESULT_PATH = "results/fc1_w8a8.npz"

# Fixed-point multiplier format:
#
#     real_multiplier ~= multiplier_int / 2^SHIFT
#
SHIFT = 31


# ============================================================
# Helper functions
# ============================================================

def round_away_from_zero_float(x):
    """
    Round floating-point values to nearest integer.
    Halfway cases are rounded away from zero.

    Examples:
        1.4  ->  1
        1.5  ->  2
       -1.4  -> -1
       -1.5  -> -2
    """

    abs_x = torch.abs(x)

    rounded_abs = torch.floor(
        abs_x + 0.5
    )

    return torch.where(
        x >= 0,
        rounded_abs,
        -rounded_abs
    )


def requantize_fixed_point(
    acc,
    multiplier_int,
    shift
):
    """
    Fixed-point requantization:

        INT32 accumulator
            ->
        INT64 multiply
            ->
        round-to-nearest
        ties away from zero
            ->
        arithmetic scaling by 2^shift
            ->
        INT8 saturation

    acc:
        shape [1, 197, 768]

    multiplier_int:
        shape [768]

    output:
        INT8 tensor
    """

    # INT32 -> INT64
    acc64 = acc.to(torch.int64)

    # Per-output-channel multiplier
    multiplier64 = multiplier_int.to(torch.int64)

    # INT32 x INT32-equivalent -> INT64
    product = (
        acc64
        * multiplier64[None, None, :]
    )

    # --------------------------------------------------------
    # Round-to-nearest, ties away from zero
    #
    # Instead of directly right-shifting negative integers,
    # operate on the magnitude first so the rounding behavior
    # is explicitly defined.
    # --------------------------------------------------------

    rounding_offset = 1 << (shift - 1)

    abs_product = torch.abs(product)

    rounded_abs = (
        abs_product + rounding_offset
    ) >> shift

    result = torch.where(
        product >= 0,
        rounded_abs,
        -rounded_abs
    )

    # Symmetric INT8 range used in this project
    result = torch.clamp(
        result,
        -127,
        127
    )

    return result.to(torch.int8)


# ============================================================
# Load fixed FP32 FC1 workload
# ============================================================

data = np.load(FP32_WORKLOAD_PATH)

x_fp = torch.from_numpy(
    data["input"]
).to(torch.float32)

weight_fp = torch.from_numpy(
    data["weight"]
).to(torch.float32)

bias_fp = torch.from_numpy(
    data["bias"]
).to(torch.float32)

y_fp = torch.from_numpy(
    data["output"]
).to(torch.float32)


print("=== Loaded Fixed FC1 Workload ===")
print()

print("Input shape :", tuple(x_fp.shape))
print("Weight shape:", tuple(weight_fp.shape))
print("Bias shape  :", tuple(bias_fp.shape))
print("Output shape:", tuple(y_fp.shape))


# ============================================================
# Quantize activation
#
# Per-tensor symmetric INT8
#
#     X ~= X_q * x_scale
#
# Zero point = 0
# ============================================================

x_max = torch.max(
    torch.abs(x_fp)
)

x_scale = x_max / 127.0

x_q = round_away_from_zero_float(
    x_fp / x_scale
)

x_q = torch.clamp(
    x_q,
    -127,
    127
).to(torch.int8)


# ============================================================
# Quantize weights
#
# Per-output-channel symmetric INT8
#
# weight shape:
#
#     [768, 192]
#
# Each output channel has its own scale:
#
#     W_j ~= W_q_j * weight_scale_j
#
# Zero point = 0
# ============================================================

weight_max = torch.amax(
    torch.abs(weight_fp),
    dim=1
)

weight_scale = (
    weight_max / 127.0
)

weight_q = round_away_from_zero_float(
    weight_fp
    / weight_scale[:, None]
)

weight_q = torch.clamp(
    weight_q,
    -127,
    127
).to(torch.int8)


# ============================================================
# Quantize bias
#
# Bias scale:
#
#     bias_scale_j
#       =
#     x_scale * weight_scale_j
#
# Therefore:
#
#     bias_q_j
#       =
#     round(
#         bias_fp_j / bias_scale_j
#     )
#
# Bias is stored in INT32.
# ============================================================

bias_scale = (
    x_scale
    * weight_scale
)

bias_q = round_away_from_zero_float(
    bias_fp / bias_scale
).to(torch.int32)


# ============================================================
# INT8 x INT8 -> INT32 GEMM
#
# FC1:
#
#   [197 x 192]
#       x
#   [192 x 768]
#
#       ->
#
#   [197 x 768]
#
# PyTorch stores weight as:
#
#   [768 x 192]
#
# so we use weight.t().
# ============================================================

x_int32 = x_q.to(torch.int32)

weight_int32 = (
    weight_q.to(torch.int32)
)

acc = torch.matmul(
    x_int32,
    weight_int32.t()
)

# Add INT32 bias
acc = (
    acc
    + bias_q[None, None, :]
)


# ============================================================
# Dequantize INT32 accumulator
#
# Used only for accuracy analysis.
#
# This floating-point operation will NOT be required by the
# final bare-metal implementation.
# ============================================================

y_dequant = (
    acc.to(torch.float32)
    * bias_scale[None, None, :]
)


# ============================================================
# INT32 -> INT8 Requantization
#
# Define output quantization:
#
#     Y ~= Y_q * y_scale
#
# Therefore:
#
#     Y_q
#       =
#     ACC
#       *
#     (x_scale * weight_scale / y_scale)
#
# Define:
#
#     requant_multiplier_j
#       =
#     bias_scale_j / y_scale
# ============================================================

y_max = torch.max(
    torch.abs(y_fp)
)

y_scale = (
    y_max / 127.0
)

requant_multiplier = (
    bias_scale
    / y_scale
)


# ============================================================
# Floating-point requantization reference
#
# IMPORTANT:
#
# This is only the mathematical golden reference.
# The final RV64IMC implementation will NOT execute this
# floating-point multiply.
# ============================================================

y_q_float = round_away_from_zero_float(
    acc.to(torch.float32)
    * requant_multiplier[
        None,
        None,
        :
    ]
)

y_q_float = torch.clamp(
    y_q_float,
    -127,
    127
).to(torch.int8)


# ============================================================
# Dequantize INT8 output
#
# Only for comparison with original FP32 FC1 output.
# ============================================================

y_requant_dequant = (
    y_q_float.to(torch.float32)
    * y_scale
)


# ============================================================
# Convert floating-point requantization multiplier
# to fixed-point integer multiplier.
#
#     multiplier_int
#       =
#     round(
#         requant_multiplier * 2^SHIFT
#     )
#
# Therefore:
#
#     requant_multiplier
#       ~=
#     multiplier_int / 2^SHIFT
# ============================================================

multiplier_int = (
    round_away_from_zero_float(
        requant_multiplier
        * (1 << SHIFT)
    )
    .to(torch.int64)
)


# ============================================================
# Fixed-point requantization
#
#     ACC
#       ->
#     INT64 multiply
#       ->
#     rounding
#       ->
#     right shift
#       ->
#     saturation
#       ->
#     INT8
# ============================================================

y_q_fixed = requantize_fixed_point(
    acc,
    multiplier_int,
    SHIFT
)


# ============================================================
# Dequantize fixed-point INT8 output
#
# Again, only for error analysis.
# ============================================================

y_fixed_dequant = (
    y_q_fixed.to(torch.float32)
    * y_scale
)


# ============================================================
# Error analysis:
#
# FP32
# vs
# INT8 x INT8 -> INT32 -> FP32
# ============================================================

error = (
    y_fp
    - y_dequant
)

mae = torch.mean(
    torch.abs(error)
)

max_error = torch.max(
    torch.abs(error)
)

signal_power = torch.mean(
    y_fp ** 2
)

noise_power = torch.mean(
    error ** 2
)

sqnr = (
    10.0
    * torch.log10(
        signal_power
        / noise_power
    )
)


# ============================================================
# Error analysis:
#
# FP32
# vs
# INT8 output after requantization
# ============================================================

requant_error = (
    y_fp
    - y_requant_dequant
)

requant_mae = torch.mean(
    torch.abs(
        requant_error
    )
)

requant_max_error = torch.max(
    torch.abs(
        requant_error
    )
)

requant_noise_power = torch.mean(
    requant_error ** 2
)

requant_sqnr = (
    10.0
    * torch.log10(
        signal_power
        / requant_noise_power
    )
)


# ============================================================
# Fixed-point vs floating-point requantization comparison
# ============================================================

fixed_diff = (
    y_q_float.to(torch.int16)
    - y_q_fixed.to(torch.int16)
)

num_different = torch.count_nonzero(
    fixed_diff
)

max_integer_diff = torch.max(
    torch.abs(
        fixed_diff
    )
)


# ============================================================
# Fixed-point output accuracy
# ============================================================

fixed_error = (
    y_fp
    - y_fixed_dequant
)

fixed_mae = torch.mean(
    torch.abs(
        fixed_error
    )
)

fixed_max_error = torch.max(
    torch.abs(
        fixed_error
    )
)

fixed_noise_power = torch.mean(
    fixed_error ** 2
)

fixed_sqnr = (
    10.0
    * torch.log10(
        signal_power
        / fixed_noise_power
    )
)


# ============================================================
# Print results
# ============================================================

print()
print("========================================")
print("FC1 W8A8 Quantization")
print("========================================")
print()

print(
    "FP input shape   :",
    tuple(x_fp.shape)
)

print(
    "INT8 input shape :",
    tuple(x_q.shape)
)

print(
    "Weight shape     :",
    tuple(weight_q.shape)
)

print(
    "Bias shape       :",
    tuple(bias_q.shape)
)

print(
    "Accumulator shape:",
    tuple(acc.shape)
)


# ------------------------------------------------------------
# Quantization parameters
# ------------------------------------------------------------

print()
print("=== Quantization Parameters ===")

print(
    "Activation scale:",
    x_scale.item()
)

print(
    "Weight scale range:",
    weight_scale.min().item(),
    "~",
    weight_scale.max().item()
)

print(
    "Output scale:",
    y_scale.item()
)

print(
    "Requant multiplier range:",
    requant_multiplier.min().item(),
    "~",
    requant_multiplier.max().item()
)


# ------------------------------------------------------------
# INT32 accumulator accuracy
# ------------------------------------------------------------

print()
print(
    "=== INT8 x INT8 -> INT32 Accuracy ==="
)

print(
    "MAE       :",
    mae.item()
)

print(
    "Max error :",
    max_error.item()
)

print(
    "SQNR (dB) :",
    sqnr.item()
)


# ------------------------------------------------------------
# Floating-point reference requantization
# ------------------------------------------------------------

print()
print(
    "=== Floating-Point Requantization Reference ==="
)

print(
    "MAE       :",
    requant_mae.item()
)

print(
    "Max error :",
    requant_max_error.item()
)

print(
    "SQNR (dB) :",
    requant_sqnr.item()
)


# ------------------------------------------------------------
# Fixed-point requantization
# ------------------------------------------------------------

print()
print(
    "=== Fixed-Point Requantization ==="
)

print(
    "Fixed-point shift:",
    SHIFT
)

print(
    "Integer multiplier range:",
    multiplier_int.min().item(),
    "~",
    multiplier_int.max().item()
)

print(
    "Different INT8 outputs:",
    num_different.item(),
    "/",
    y_q_float.numel()
)

print(
    "Maximum INT8 difference:",
    max_integer_diff.item()
)

print(
    "Fixed-point MAE:",
    fixed_mae.item()
)

print(
    "Fixed-point max error:",
    fixed_max_error.item()
)

print(
    "Fixed-point SQNR (dB):",
    fixed_sqnr.item()
)


# ------------------------------------------------------------
# Samples
# ------------------------------------------------------------

print()
print("=== Sample Values ===")

print()
print("Original FP32:")
print(
    y_fp[
        0,
        0,
        0:8
    ]
)

print()
print("INT8 x INT8 -> INT32 -> FP32:")
print(
    y_dequant[
        0,
        0,
        0:8
    ]
)

print()
print("Float-reference INT8:")
print(
    y_q_float[
        0,
        0,
        0:8
    ]
)

print()
print("Fixed-point INT8:")
print(
    y_q_fixed[
        0,
        0,
        0:8
    ]
)

print()
print("Fixed-point INT8 -> FP32:")
print(
    y_fixed_dequant[
        0,
        0,
        0:8
    ]
)


# ============================================================
# Save data for C / Spike implementation
# ============================================================

np.savez(
    W8A8_RESULT_PATH,

    # --------------------------------------------------------
    # INT8 workload
    # --------------------------------------------------------

    input_q=x_q.numpy(),

    weight_q=weight_q.numpy(),

    bias_q=bias_q.numpy(),

    # --------------------------------------------------------
    # Golden INT32 accumulator
    # --------------------------------------------------------

    accumulator=acc.numpy(),

    # --------------------------------------------------------
    # Quantization parameters
    # --------------------------------------------------------

    input_scale=np.array(
        x_scale.item(),
        dtype=np.float32
    ),

    weight_scale=(
        weight_scale.numpy()
    ),

    output_scale=np.array(
        y_scale.item(),
        dtype=np.float32
    ),

    # --------------------------------------------------------
    # Requantization parameters
    # --------------------------------------------------------

    requant_multiplier=(
        requant_multiplier.numpy()
    ),

    multiplier_int=(
        multiplier_int.numpy()
    ),

    requant_shift=np.array(
        SHIFT,
        dtype=np.int32
    ),

    # --------------------------------------------------------
    # Golden outputs
    # --------------------------------------------------------

    output_fp=(
        y_fp.numpy()
    ),

    output_dequant=(
        y_dequant.numpy()
    ),

    output_q_float=(
        y_q_float.numpy()
    ),

    output_q_fixed=(
        y_q_fixed.numpy()
    ),

    output_fixed_dequant=(
        y_fixed_dequant.numpy()
    )
)


print()
print(
    "Saved",
    W8A8_RESULT_PATH
)