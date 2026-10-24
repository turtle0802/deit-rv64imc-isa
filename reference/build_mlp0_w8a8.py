from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

FC1_FP32 = Path("results/fc1_fp32.npz")
FC2_W8A8 = Path("results/fc2_w8a8.npz")
FC2_FP32 = Path("results/fc2_fp32.npz")
OUTPUT = Path("results/mlp0_w8a8.npz")

QMAX = 127
SHIFT = 31


def symmetric_scale(x):
    m = float(np.max(np.abs(x)))
    if m == 0.0:
        return np.float32(1.0)
    return np.float32(m / QMAX)


def fixed_requant(acc, multiplier):
    product = (
        acc.astype(np.int64)
        * multiplier[None, :].astype(np.int64)
    )

    magnitude = np.where(
        product < 0,
        -product,
        product
    )

    rounded = (
        magnitude + (1 << (SHIFT - 1))
    ) >> SHIFT

    rounded = np.where(
        product < 0,
        -rounded,
        rounded
    )

    return np.clip(
        rounded,
        -QMAX,
        QMAX
    ).astype(np.int8)


def sqnr_db(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum((ref - test) ** 2)

    if noise == 0:
        return float("inf")

    return 10.0 * np.log10(signal / noise)


def main():
    # ============================================================
    # Load data
    # ============================================================
    fc1 = np.load(FC1_FP32)
    fc2 = np.load(FC2_W8A8)
    fc2_fp32 = np.load(FC2_FP32)

    # ============================================================
    # Rebuild FC1 W8A8 from FP32
    # ============================================================
    x1 = fc1["input"].astype(np.float32)
    w1 = fc1["weight"].astype(np.float32)
    b1 = fc1["bias"].astype(np.float32)
    y1_fp32 = fc1["output"].astype(np.float32)

    # Remove batch dimension if present.
    if x1.ndim == 3 and x1.shape[0] == 1:
        x1 = x1[0]

    if y1_fp32.ndim == 3 and y1_fp32.shape[0] == 1:
        y1_fp32 = y1_fp32[0]

    assert x1.shape == (197, 192)
    assert w1.shape == (768, 192)
    assert b1.shape == (768,)
    assert y1_fp32.shape == (197, 768)

    sx1 = symmetric_scale(x1)

    x1_q = np.clip(
        np.rint(x1 / sx1),
        -QMAX,
        QMAX
    ).astype(np.int8)

    w1_max = np.max(
        np.abs(w1),
        axis=1
    )

    sw1 = np.where(
        w1_max == 0.0,
        1.0,
        w1_max / QMAX
    ).astype(np.float32)

    w1_q = np.clip(
        np.rint(
            w1 / sw1[:, None]
        ),
        -QMAX,
        QMAX
    ).astype(np.int8)

    sb1 = sx1 * sw1

    b1_q = np.rint(
        b1 / sb1
    ).astype(np.int32)

    acc1 = (
        x1_q.astype(np.int32)
        @
        w1_q.astype(np.int32).T
    )

    acc1 += b1_q[None, :]

    sy1 = symmetric_scale(y1_fp32)

    m1 = (
        sb1.astype(np.float64)
        / float(sy1)
    )

    m1_int = np.rint(
        m1 * (1 << SHIFT)
    ).astype(np.int64)

    fc1_output_q = fixed_requant(
        acc1,
        m1_int
    )

    print("FC1 rebuilt successfully")
    print("FC1 output shape:", fc1_output_q.shape)
    print("FC1 output scale:", float(sy1))

    # ============================================================
    # FC2 information
    # ============================================================
    fc2_input_scale = float(
        fc2["input_scale"]
    )

    fc2_weight_q = (
        fc2["weight_q"]
        .astype(np.int8)
    )

    fc2_bias_q = (
        fc2["bias_q"]
        .astype(np.int32)
    )

    fc2_multiplier = (
        fc2["multiplier_int"]
        .astype(np.int64)
    )

    fc2_output_scale = float(
        fc2["output_scale"]
    )

    # ============================================================
    # GELU LUT:
    #
    # FC1 output INT8
    #      ↓
    # dequant with FC1 output scale
    #      ↓
    # GELU
    #      ↓
    # quantize using FC2 input scale
    # ============================================================
    gelu_lut = np.zeros(
        256,
        dtype=np.int8
    )

    for q in range(-128, 128):
        x_real = (
            float(q)
            * float(sy1)
        )

        x_tensor = torch.tensor(
            x_real,
            dtype=torch.float32
        )

        y_real = float(
            F.gelu(
                x_tensor,
                approximate="none"
            ).item()
        )

        y_q = int(
            np.rint(
                y_real
                / fc2_input_scale
            )
        )

        y_q = max(
            -QMAX,
            min(QMAX, y_q)
        )

        gelu_lut[q & 0xff] = y_q

    # INT8 bit pattern -> LUT index
    lut_index = (
        fc1_output_q
        .view(np.uint8)
    )

    fc2_input_connected_q = (
        gelu_lut[lut_index]
    )

    # ============================================================
    # Compare connected GELU result with standalone FC2 input
    # ============================================================
    standalone_input_q = (
        fc2["input_q"]
        .astype(np.int8)
    )

    input_diff = (
        fc2_input_connected_q.astype(np.int16)
        -
        standalone_input_q.astype(np.int16)
    )

    input_mismatch = int(
        np.count_nonzero(input_diff)
    )

    input_max_diff = int(
        np.max(np.abs(input_diff))
    )

    # ============================================================
    # Connected FC2
    # ============================================================
    acc2 = (
        fc2_input_connected_q.astype(np.int32)
        @
        fc2_weight_q.astype(np.int32).T
    )

    acc2 += fc2_bias_q[None, :]

    fc2_output_connected_q = fixed_requant(
        acc2,
        fc2_multiplier
    )

    # ============================================================
    # Compare with standalone FC2
    # ============================================================
    standalone_output_q = (
        fc2["output_q"]
        .astype(np.int8)
    )

    output_diff = (
        fc2_output_connected_q.astype(np.int16)
        -
        standalone_output_q.astype(np.int16)
    )

    output_mismatch = int(
        np.count_nonzero(output_diff)
    )

    output_max_diff = int(
        np.max(np.abs(output_diff))
    )

    # ============================================================
    # Compare complete connected MLP with FP32 MLP output
    # ============================================================
    reference_fp32 = (
        fc2_fp32["output"]
        .astype(np.float32)
    )

    connected_fp32 = (
        fc2_output_connected_q.astype(np.float32)
        * fc2_output_scale
    )

    error = (
        connected_fp32
        - reference_fp32
    )

    mae = float(
        np.mean(np.abs(error))
    )

    max_error = float(
        np.max(np.abs(error))
    )

    sqnr = sqnr_db(
        reference_fp32,
        connected_fp32
    )

    # ============================================================
    # Save golden for C implementation
    # ============================================================
    np.savez(
        OUTPUT,

        gelu_lut=gelu_lut,

        fc1_input_q=x1_q,
        fc1_weight_q=w1_q,
        fc1_bias_q=b1_q,
        fc1_multiplier=m1_int,
        fc1_output_q=fc1_output_q,

        fc1_output_scale=np.array(
            sy1,
            dtype=np.float32
        ),

        fc2_input_scale=np.array(
            fc2_input_scale,
            dtype=np.float32
        ),

        fc2_input_q=fc2_input_connected_q,

        fc2_weight_q=fc2_weight_q,
        fc2_bias_q=fc2_bias_q,
        fc2_multiplier=fc2_multiplier,

        fc2_accumulator=acc2,
        fc2_output_q=fc2_output_connected_q,

        fc2_output_scale=np.array(
            fc2_output_scale,
            dtype=np.float32
        ),

        shift=np.array(
            SHIFT,
            dtype=np.int32
        )
    )

    # ============================================================
    # Report
    # ============================================================
    print()
    print("Saved:", OUTPUT)

    print()
    print("MLP0 connected pipeline")
    print("--------------------------------")
    print(
        "FC1 output      :",
        fc1_output_q.shape
    )
    print(
        "GELU / FC2 input:",
        fc2_input_connected_q.shape
    )
    print(
        "FC2 output      :",
        fc2_output_connected_q.shape
    )

    print()
    print("GELU LUT")
    print("--------------------------------")
    print(
        "Entries:",
        gelu_lut.size
    )
    print(
        "FC1 output scale:",
        float(sy1)
    )
    print(
        "FC2 input scale :",
        fc2_input_scale
    )

    print()
    print(
        "Connected vs standalone FC2 input"
    )
    print("--------------------------------")
    print(
        "Mismatch:",
        input_mismatch,
        "/",
        fc2_input_connected_q.size
    )
    print(
        "Max INT8 difference:",
        input_max_diff
    )

    print()
    print(
        "Connected vs standalone FC2 output"
    )
    print("--------------------------------")
    print(
        "Mismatch:",
        output_mismatch,
        "/",
        fc2_output_connected_q.size
    )
    print(
        "Max INT8 difference:",
        output_max_diff
    )

    print()
    print("Connected MLP vs FP32 MLP")
    print("--------------------------------")
    print("MAE       :", mae)
    print("Max error :", max_error)
    print("SQNR (dB) :", sqnr)

    print()
    print("First GELU output q:")
    print(
        fc2_input_connected_q[
            0, :8
        ]
    )

    print()
    print("First final output q:")
    print(
        fc2_output_connected_q[
            0, :8
        ]
    )

    print()
    print(
        "Connected FC2 accumulator range:"
    )
    print(
        int(np.min(acc2)),
        "to",
        int(np.max(acc2))
    )


if __name__ == "__main__":
    main()

