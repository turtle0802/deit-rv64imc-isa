from pathlib import Path

import numpy as np


INPUT_FILE = Path("results/fc2_fp32.npz")
OUTPUT_FILE = Path("results/fc2_w8a8.npz")

QMAX = 127
REQUANT_SHIFT = 31


def symmetric_scale(x):
    max_abs = float(np.max(np.abs(x)))

    if max_abs == 0.0:
        return np.float32(1.0)

    return np.float32(max_abs / QMAX)


def sqnr_db(reference, approximation):
    reference = reference.astype(np.float64)
    approximation = approximation.astype(np.float64)

    signal_power = np.sum(reference * reference)

    error = reference - approximation
    noise_power = np.sum(error * error)

    if noise_power == 0.0:
        return float("inf")

    return 10.0 * np.log10(
        signal_power / noise_power
    )


def main():
    data = np.load(INPUT_FILE)

    x = data["input"].astype(np.float32)
    weight = data["weight"].astype(np.float32)
    bias = data["bias"].astype(np.float32)
    output_fp32 = data["output"].astype(np.float32)

    assert x.shape == (197, 768)
    assert weight.shape == (192, 768)
    assert bias.shape == (192,)
    assert output_fp32.shape == (197, 192)

    # ============================================================
    # 1. Input activation quantization
    #
    # Per-tensor symmetric INT8.
    # ============================================================
    input_scale = symmetric_scale(x)

    input_q = np.clip(
        np.rint(
            x / input_scale
        ),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # ============================================================
    # 2. Weight quantization
    #
    # Per-output-channel symmetric INT8.
    #
    # FC2 weight:
    # [192 output channels, 768 input channels]
    # ============================================================
    weight_max = np.max(
        np.abs(weight),
        axis=1,
    )

    weight_scale = np.where(
        weight_max == 0.0,
        1.0,
        weight_max / QMAX,
    ).astype(np.float32)

    weight_q = np.clip(
        np.rint(
            weight
            / weight_scale[:, None]
        ),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # ============================================================
    # 3. Bias quantization
    #
    # bias_scale[j] =
    #     input_scale * weight_scale[j]
    # ============================================================
    bias_scale = (
        input_scale
        * weight_scale
    ).astype(np.float32)

    bias_q = np.rint(
        bias / bias_scale
    ).astype(np.int32)

    # ============================================================
    # 4. INT8 x INT8 -> INT32 GEMM
    #
    # [197 x 768]
    # x
    # [768 x 192]
    #
    # ->
    #
    # [197 x 192]
    # ============================================================
    accumulator = (
        input_q.astype(np.int32)
        @
        weight_q.astype(np.int32).T
    )

    accumulator += bias_q[None, :]

    assert accumulator.shape == (197, 192)

    # ============================================================
    # 5. Dequantize accumulator
    #
    # Used only to measure quantization error.
    # ============================================================
    output_acc_fp32 = (
        accumulator.astype(np.float32)
        * bias_scale[None, :]
    )

    acc_error = (
        output_acc_fp32
        - output_fp32
    )

    acc_mae = float(
        np.mean(np.abs(acc_error))
    )

    acc_max_error = float(
        np.max(np.abs(acc_error))
    )

    acc_sqnr = sqnr_db(
        output_fp32,
        output_acc_fp32,
    )

    # ============================================================
    # 6. Output quantization scale
    #
    # Per-tensor symmetric INT8.
    # ============================================================
    output_scale = symmetric_scale(
        output_fp32
    )

    # ============================================================
    # 7. Requantization multiplier
    #
    # M[j] =
    #
    #   input_scale * weight_scale[j]
    #   --------------------------------
    #             output_scale
    #
    # ============================================================
    requant_multiplier = (
        bias_scale
        / output_scale
    ).astype(np.float64)

    # Floating-point golden requantization.
    output_q_float = np.clip(
        np.rint(
            accumulator.astype(np.float64)
            * requant_multiplier[None, :]
        ),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # ============================================================
    # 8. Q31 fixed-point multiplier
    #
    # M_int =
    #     round(M * 2^31)
    # ============================================================
    multiplier_int = np.rint(
        requant_multiplier
        * (1 << REQUANT_SHIFT)
    ).astype(np.int64)

    # ============================================================
    # 9. Bit-exact fixed-point requantization
    #
    # product =
    #     accumulator * multiplier_int
    #
    # Round-to-nearest,
    # ties-away-from-zero.
    # ============================================================
    product = (
        accumulator.astype(np.int64)
        * multiplier_int[None, :]
    )

    magnitude = np.where(
        product < 0,
        -product,
        product,
    )

    rounded_magnitude = (
        magnitude
        + (1 << (REQUANT_SHIFT - 1))
    ) >> REQUANT_SHIFT

    output_fixed_i64 = np.where(
        product < 0,
        -rounded_magnitude,
        rounded_magnitude,
    )

    output_q = np.clip(
        output_fixed_i64,
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # ============================================================
    # 10. Check fixed-point requantization against floating
    #     requantization.
    # ============================================================
    requant_difference = (
        output_q.astype(np.int16)
        - output_q_float.astype(np.int16)
    )

    requant_mismatch_count = int(
        np.count_nonzero(
            requant_difference
        )
    )

    requant_max_int8_difference = int(
        np.max(
            np.abs(
                requant_difference
            )
        )
    )

    # ============================================================
    # 11. Final dequantized output quality
    # ============================================================
    output_dequant = (
        output_q.astype(np.float32)
        * output_scale
    )

    output_error = (
        output_dequant
        - output_fp32
    )

    output_mae = float(
        np.mean(np.abs(output_error))
    )

    output_max_error = float(
        np.max(np.abs(output_error))
    )

    output_sqnr = sqnr_db(
        output_fp32,
        output_dequant,
    )

    # ============================================================
    # Save everything needed by C / Spike experiments.
    # ============================================================
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_FILE,

        input_q=input_q,
        weight_q=weight_q,
        bias_q=bias_q,

        accumulator=accumulator,

        input_scale=np.array(
            input_scale,
            dtype=np.float32,
        ),

        weight_scale=weight_scale,
        bias_scale=bias_scale,

        output_scale=np.array(
            output_scale,
            dtype=np.float32,
        ),

        requant_multiplier=requant_multiplier,
        multiplier_int=multiplier_int,

        shift=np.array(
            REQUANT_SHIFT,
            dtype=np.int32,
        ),

        output_q_float=output_q_float,
        output_q=output_q,

        output_fp32=output_fp32,
    )

    # ============================================================
    # Report
    # ============================================================
    print("Saved:", OUTPUT_FILE)

    print()
    print("FC2 dimensions")
    print("--------------------------------")
    print("Input       :", input_q.shape)
    print("Weight      :", weight_q.shape)
    print("Bias        :", bias_q.shape)
    print("Accumulator :", accumulator.shape)
    print("Output      :", output_q.shape)

    print()
    print("Quantization scales")
    print("--------------------------------")
    print(
        "Activation scale:",
        float(input_scale),
    )

    print(
        "Weight scale range:",
        float(np.min(weight_scale)),
        "to",
        float(np.max(weight_scale)),
    )

    print(
        "Output scale:",
        float(output_scale),
    )

    print()
    print("INT8xINT8 -> INT32 accuracy")
    print("--------------------------------")
    print("MAE       :", acc_mae)
    print("Max error :", acc_max_error)
    print("SQNR (dB) :", acc_sqnr)

    print()
    print("Requantization")
    print("--------------------------------")
    print(
        "Multiplier range:",
        float(np.min(requant_multiplier)),
        "to",
        float(np.max(requant_multiplier)),
    )

    print(
        "Integer multiplier range:",
        int(np.min(multiplier_int)),
        "to",
        int(np.max(multiplier_int)),
    )

    print(
        "Shift:",
        REQUANT_SHIFT,
    )

    print(
        "Fixed vs floating mismatches:",
        requant_mismatch_count,
        "/",
        output_q.size,
    )

    print(
        "Max INT8 difference:",
        requant_max_int8_difference,
    )

    print()
    print("Final W8A8 output accuracy")
    print("--------------------------------")
    print("MAE       :", output_mae)
    print("Max error :", output_max_error)
    print("SQNR (dB) :", output_sqnr)

    print()
    print("First output INT8 values:")
    print(output_q[0, :8])

    print()
    print("Accumulator range:")
    print(
        int(np.min(accumulator)),
        "to",
        int(np.max(accumulator)),
    )


if __name__ == "__main__":
    main()
