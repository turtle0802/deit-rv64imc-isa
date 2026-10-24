from pathlib import Path

import numpy as np
import torch
import timm


AV_FILE = Path("results/av0_w8a8.npz")
FC1_FILE = Path("results/fc1_fp32.npz")
OUTPUT = Path("results/proj0_w8a8.npz")

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

    mag = np.where(
        product < 0,
        -product,
        product,
    )

    rounded = (
        mag + (1 << (SHIFT - 1))
    ) >> SHIFT

    rounded = np.where(
        product < 0,
        -rounded,
        rounded,
    )

    return np.clip(
        rounded,
        -QMAX,
        QMAX,
    ).astype(np.int8)


def sqnr_db(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum((ref - test) ** 2)

    if noise == 0:
        return float("inf")

    return 10.0 * np.log10(
        signal / noise
    )


def main():
    av = np.load(AV_FILE)
    frozen = np.load(FC1_FILE)

    input_q = (
        av["concat_q"]
        .astype(np.int8)
    )

    input_scale = float(
        av["output_scale"]
    )

    assert input_q.shape == (197, 192)

    # ------------------------------------------------------------
    # Recreate same DeiT-Tiny and capture actual FP32 proj output.
    # ------------------------------------------------------------
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
    )

    model.eval()

    image = (
        frozen["image"]
        .astype(np.float32)
    )

    proj = model.blocks[0].attn.proj

    captured = {}

    def hook(module, inputs, output):
        captured["input"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        captured["output"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    h = proj.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(
            torch.from_numpy(image)
        )

    h.remove()

    proj_input_fp32 = captured["input"]
    proj_output_fp32 = captured["output"]

    if proj_input_fp32.ndim == 3:
        proj_input_fp32 = proj_input_fp32[0]

    if proj_output_fp32.ndim == 3:
        proj_output_fp32 = proj_output_fp32[0]

    weight = (
        proj.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    bias = (
        proj.bias
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    assert proj_input_fp32.shape == (197, 192)
    assert proj_output_fp32.shape == (197, 192)
    assert weight.shape == (192, 192)
    assert bias.shape == (192,)

    # ------------------------------------------------------------
    # Weight: per-output-channel INT8.
    # ------------------------------------------------------------
    wmax = np.max(
        np.abs(weight),
        axis=1,
    )

    weight_scale = np.where(
        wmax == 0.0,
        1.0,
        wmax / QMAX,
    ).astype(np.float32)

    weight_q = np.clip(
        np.rint(
            weight
            / weight_scale[:, None]
        ),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # ------------------------------------------------------------
    # Bias INT32.
    # ------------------------------------------------------------
    bias_scale = (
        input_scale
        * weight_scale
    )

    bias_q = np.rint(
        bias / bias_scale
    ).astype(np.int32)

    # ------------------------------------------------------------
    # Connected AV -> Projection GEMM.
    # ------------------------------------------------------------
    acc = (
        input_q.astype(np.int32)
        @
        weight_q.astype(np.int32).T
    )

    acc += bias_q[None, :]

    # ------------------------------------------------------------
    # Projection output quantization.
    # ------------------------------------------------------------
    output_scale = symmetric_scale(
        proj_output_fp32
    )

    multiplier = (
        bias_scale.astype(np.float64)
        / float(output_scale)
    )

    multiplier_int = np.rint(
        multiplier
        * (1 << SHIFT)
    ).astype(np.int64)

    output_q = fixed_requant(
        acc,
        multiplier_int,
    )

    output_dequant = (
        output_q.astype(np.float32)
        * output_scale
    )

    # ------------------------------------------------------------
    # Evaluate connected attention projection.
    # ------------------------------------------------------------
    input_dequant = (
        input_q.astype(np.float32)
        * input_scale
    )

    input_mae = float(
        np.mean(
            np.abs(
                input_dequant
                - proj_input_fp32
            )
        )
    )

    output_mae = float(
        np.mean(
            np.abs(
                output_dequant
                - proj_output_fp32
            )
        )
    )

    output_max = float(
        np.max(
            np.abs(
                output_dequant
                - proj_output_fp32
            )
        )
    )

    output_sqnr = sqnr_db(
        proj_output_fp32,
        output_dequant,
    )

    np.savez(
        OUTPUT,

        input_q=input_q,
        weight_q=weight_q,
        bias_q=bias_q,

        accumulator=acc,

        input_scale=np.array(
            input_scale,
            dtype=np.float32,
        ),

        weight_scale=weight_scale,

        output_scale=np.array(
            output_scale,
            dtype=np.float32,
        ),

        multiplier_int=multiplier_int,

        shift=np.array(
            SHIFT,
            dtype=np.int32,
        ),

        output_q=output_q,

        proj_input_fp32=proj_input_fp32,
        proj_output_fp32=proj_output_fp32,
    )

    macs = 197 * 192 * 192

    print("Saved:", OUTPUT)

    print()
    print("Projection")
    print("--------------------------------")
    print("Input :", input_q.shape)
    print("Weight:", weight_q.shape)
    print("Output:", output_q.shape)

    print()
    print("Workload")
    print("--------------------------------")
    print("MACs       :", f"{macs:,}")
    print("Xqmac8     :", f"{macs // 8:,}")
    print("Xrequant   :", f"{197 * 192:,}")
    print(
        "Custom total:",
        f"{macs // 8 + 197 * 192:,}",
    )

    print()
    print("Scales")
    print("--------------------------------")
    print("Input scale :", input_scale)
    print(
        "Weight scale:",
        float(np.min(weight_scale)),
        "to",
        float(np.max(weight_scale)),
    )
    print(
        "Output scale:",
        float(output_scale),
    )

    print()
    print("Connected quality")
    print("--------------------------------")
    print(
        "Projection input MAE:",
        input_mae,
    )
    print(
        "Projection output MAE:",
        output_mae,
    )
    print(
        "Projection max error:",
        output_max,
    )
    print(
        "Projection SQNR (dB):",
        output_sqnr,
    )

    print()
    print("Accumulator range")
    print("--------------------------------")
    print(
        int(np.min(acc)),
        "to",
        int(np.max(acc)),
    )

    print()
    print("First projection output:")
    print(output_q[0, :8])


if __name__ == "__main__":
    main()
