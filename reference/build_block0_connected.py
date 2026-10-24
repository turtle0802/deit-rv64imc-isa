from pathlib import Path
from math import isqrt

import numpy as np
import torch
import torch.nn.functional as F


BLOCK = Path("results/block0_fp32.npz")
MLP = Path("results/block0_mlp_fp32.npz")

QKV = Path("results/qkv_w8a8.npz")
SOFTMAX = Path("results/softmax0_int8.npz")
AV = Path("results/av0_w8a8.npz")
PROJ = Path("results/proj0_w8a8.npz")

OUTPUT = Path("results/block0_connected_int8.npz")

QMAX = 127
SHIFT = 31

LN_Z_Q = 15
LN_AFFINE_Q = 20


def symmetric_scale(x):
    m = float(np.max(np.abs(x)))

    if m == 0.0:
        return np.float32(1.0)

    return np.float32(m / QMAX)


def round_shift_signed(x, shift):
    x = np.asarray(x, dtype=np.int64)

    mag = np.where(
        x < 0,
        -x,
        x,
    )

    y = (
        mag
        + (1 << (shift - 1))
    ) >> shift

    return np.where(
        x < 0,
        -y,
        y,
    )


def requant_q31(x, multiplier):
    x = np.asarray(x, dtype=np.int64)
    m = np.asarray(multiplier, dtype=np.int64)

    product = x * m

    return round_shift_signed(
        product,
        SHIFT,
    )


def saturate_int8(x):
    return np.clip(
        x,
        -QMAX,
        QMAX,
    ).astype(np.int8)


def quantize_int8(x, scale):
    return saturate_int8(
        np.rint(
            x / float(scale)
        )
    )


def make_q31_multiplier(ratio):
    m = np.rint(
        np.asarray(ratio, dtype=np.float64)
        * (1 << SHIFT)
    ).astype(np.int64)

    if np.any(m >= (1 << 31)):
        raise RuntimeError(
            "Q31 multiplier >= 1.0; "
            "choose a larger destination scale."
        )

    if np.any(m < 0):
        raise RuntimeError(
            "Negative Q31 scale multiplier."
        )

    return m


def sqrt_nearest(x):
    r = isqrt(int(x))

    lo = int(x) - r * r
    hi = (r + 1) * (r + 1) - int(x)

    if hi < lo:
        r += 1

    return r


def div_round_signed(num, den):
    if den <= 0:
        raise RuntimeError("Invalid divisor")

    if num >= 0:
        return (
            num + den // 2
        ) // den

    return -(
        (-num + den // 2)
        // den
    )


def integer_layernorm(
    x_q,
    input_scale,
    gamma,
    beta,
    eps,
    output_scale,
):
    """
    Pure-integer LayerNorm core.

    For N elements:

      centered_num = x*N - sum(x)

      var_num =
          N*sum(x^2) - sum(x)^2

      z =
          centered_num /
          sqrt(
              var_num +
              eps*N^2/input_scale^2
          )

    z is represented as Q15.
    """

    x_q = np.asarray(
        x_q,
        dtype=np.int8,
    )

    rows, n = x_q.shape

    gamma = np.asarray(
        gamma,
        dtype=np.float64,
    )

    beta = np.asarray(
        beta,
        dtype=np.float64,
    )

    # y_q =
    #   z * gamma/output_scale
    #   + beta/output_scale
    #
    # gamma/output_scale and beta/output_scale
    # are stored as Q20.
    affine_a = np.rint(
        (
            gamma
            / float(output_scale)
        )
        * (1 << LN_AFFINE_Q)
    ).astype(np.int64)

    affine_b = np.rint(
        (
            beta
            / float(output_scale)
        )
        * (1 << LN_AFFINE_Q)
    ).astype(np.int64)

    eps_term = int(
        np.rint(
            float(eps)
            * n
            * n
            / (
                float(input_scale)
                * float(input_scale)
            )
        )
    )

    out = np.empty_like(
        x_q,
        dtype=np.int8,
    )

    z_store = np.empty(
        x_q.shape,
        dtype=np.int32,
    )

    for r in range(rows):
        row = x_q[r].astype(np.int64)

        s = int(np.sum(row))
        ss = int(np.sum(row * row))

        var_num = (
            n * ss
            - s * s
            + eps_term
        )

        if var_num <= 0:
            var_num = 1

        denom = sqrt_nearest(
            var_num
        )

        for c in range(n):
            centered_num = (
                int(row[c]) * n
                - s
            )

            z_q15 = div_round_signed(
                centered_num
                << LN_Z_Q,
                denom,
            )

            z_store[r, c] = z_q15

            value = (
                z_q15
                * int(affine_a[c])
                +
                (
                    int(affine_b[c])
                    << LN_Z_Q
                )
            )

            y = int(
                round_shift_signed(
                    np.array([value]),
                    LN_Z_Q
                    + LN_AFFINE_Q,
                )[0]
            )

            if y > QMAX:
                y = QMAX
            elif y < -QMAX:
                y = -QMAX

            out[r, c] = y

    return (
        out,
        affine_a,
        affine_b,
        eps_term,
        z_store,
    )


def softmax_integer(
    qkt_acc,
    exp_lut,
    clip_steps,
):
    row_max = np.max(
        qkt_acc,
        axis=2,
        keepdims=True,
    )

    distance = (
        row_max.astype(np.int64)
        -
        qkt_acc.astype(np.int64)
    )

    index = (
        distance * 255
        + clip_steps // 2
    ) // clip_steps

    index = np.clip(
        index,
        0,
        255,
    )

    exp_q = (
        exp_lut[index]
        .astype(np.int64)
    )

    exp_sum = np.sum(
        exp_q,
        axis=2,
        keepdims=True,
        dtype=np.int64,
    )

    prob = (
        exp_q * 127
        + exp_sum // 2
    ) // exp_sum

    return np.clip(
        prob,
        0,
        127,
    ).astype(np.int8)


def sqnr_db(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum(
        (ref - test) ** 2
    )

    if noise == 0:
        return float("inf")

    return 10.0 * np.log10(
        signal / noise
    )


def main():
    block = np.load(BLOCK)
    mlp = np.load(MLP)

    qkv_d = np.load(QKV)
    soft_d = np.load(SOFTMAX)
    av_d = np.load(AV)
    proj_d = np.load(PROJ)

    # ============================================================
    # X0
    # ============================================================
    x0_fp = block["x0"].astype(np.float32)

    x0_scale = symmetric_scale(
        x0_fp
    )

    x0_q = quantize_int8(
        x0_fp,
        x0_scale,
    )

    # ============================================================
    # LN1
    #
    # Force LN1 output scale to exactly match the QKV kernel's
    # expected input scale.
    # ============================================================
    ln1_scale = float(
        qkv_d["input_scale"]
    )

    ln1_q, ln1_a, ln1_b, ln1_eps_term, _ = (
        integer_layernorm(
            x0_q,
            x0_scale,
            block["norm1_gamma"],
            block["norm1_beta"],
            float(block["norm1_eps"]),
            ln1_scale,
        )
    )

    ln1_deq = (
        ln1_q.astype(np.float32)
        * ln1_scale
    )

    # ============================================================
    # QKV from CONNECTED integer LN1
    # ============================================================
    qkv_acc = (
        ln1_q.astype(np.int32)
        @
        qkv_d["weight_q"]
        .astype(np.int32).T
    )

    qkv_acc += (
        qkv_d["bias_q"]
        .astype(np.int32)[None, :]
    )

    qkv_out_q = saturate_int8(
        requant_q31(
            qkv_acc,
            qkv_d[
                "multiplier_int"
            ].astype(np.int64)[None, :],
        )
    )

    qkv_scale = float(
        qkv_d["output_scale"]
    )

    split = qkv_out_q.reshape(
        197,
        3,
        3,
        64,
    )

    q_q = (
        split[:, 0]
        .transpose(1, 0, 2)
    )

    k_q = (
        split[:, 1]
        .transpose(1, 0, 2)
    )

    v_q = (
        split[:, 2]
        .transpose(1, 0, 2)
    )

    # ============================================================
    # QK^T
    # ============================================================
    qkt_acc = np.empty(
        (3, 197, 197),
        dtype=np.int32,
    )

    for h in range(3):
        qkt_acc[h] = (
            q_q[h].astype(np.int32)
            @
            k_q[h].astype(np.int32).T
        )

    # ============================================================
    # Integer Softmax
    # ============================================================
    prob_q = softmax_integer(
        qkt_acc,
        soft_d["exp_lut"],
        int(
            soft_d["clip_steps"]
        ),
    )

    # ============================================================
    # Attention x V
    # ============================================================
    av_acc = np.empty(
        (3, 197, 64),
        dtype=np.int32,
    )

    for h in range(3):
        av_acc[h] = (
            prob_q[h].astype(np.int32)
            @
            v_q[h].astype(np.int32)
        )

    av_q = saturate_int8(
        requant_q31(
            av_acc,
            int(
                av_d["multiplier_int"]
            ),
        )
    )

    av_concat_q = (
        av_q
        .transpose(1, 0, 2)
        .reshape(197, 192)
    )

    # ============================================================
    # Projection
    # ============================================================
    proj_acc = (
        av_concat_q.astype(np.int32)
        @
        proj_d["weight_q"]
        .astype(np.int32).T
    )

    proj_acc += (
        proj_d["bias_q"]
        .astype(np.int32)[None, :]
    )

    proj_q = saturate_int8(
        requant_q31(
            proj_acc,
            proj_d[
                "multiplier_int"
            ].astype(np.int64)[None, :],
        )
    )

    proj_scale = float(
        proj_d["output_scale"]
    )

    # ============================================================
    # Residual 1
    #
    # Use a scale large enough that neither source needs
    # multiplier > 1.0.
    # ============================================================
    residual1_ref_scale = float(
        symmetric_scale(
            block["residual1"]
        )
    )

    residual1_scale = max(
        float(x0_scale),
        proj_scale,
        residual1_ref_scale,
    )

    x0_to_r1 = int(
        make_q31_multiplier(
            float(x0_scale)
            / residual1_scale
        )
    )

    proj_to_r1 = int(
        make_q31_multiplier(
            proj_scale
            / residual1_scale
        )
    )

    x0_r1 = requant_q31(
        x0_q,
        x0_to_r1,
    )

    proj_r1 = requant_q31(
        proj_q,
        proj_to_r1,
    )

    residual1_q = saturate_int8(
        x0_r1
        + proj_r1
    )

    # ============================================================
    # LN2
    #
    # Its output scale becomes the TRUE Block0 FC1 input scale.
    # ============================================================
    true_fc1_input_fp = (
        mlp["fc1_input"]
        .astype(np.float32)
    )

    fc1_input_scale = symmetric_scale(
        true_fc1_input_fp
    )

    ln2_q, ln2_a, ln2_b, ln2_eps_term, _ = (
        integer_layernorm(
            residual1_q,
            residual1_scale,
            block["norm2_gamma"],
            block["norm2_beta"],
            float(block["norm2_eps"]),
            fc1_input_scale,
        )
    )

    # ============================================================
    # TRUE FC1 quantization
    # ============================================================
    w1 = (
        mlp["fc1_weight"]
        .astype(np.float32)
    )

    b1 = (
        mlp["fc1_bias"]
        .astype(np.float32)
    )

    fc1_fp_output = (
        mlp["fc1_output"]
        .astype(np.float32)
    )

    w1_max = np.max(
        np.abs(w1),
        axis=1,
    )

    w1_scale = np.where(
        w1_max == 0,
        1.0,
        w1_max / QMAX,
    ).astype(np.float32)

    w1_q = saturate_int8(
        np.rint(
            w1
            / w1_scale[:, None]
        )
    )

    b1_scale = (
        float(fc1_input_scale)
        * w1_scale
    )

    b1_q = np.rint(
        b1 / b1_scale
    ).astype(np.int32)

    fc1_acc = (
        ln2_q.astype(np.int32)
        @
        w1_q.astype(np.int32).T
    )

    fc1_acc += b1_q[None, :]

    fc1_output_scale = symmetric_scale(
        fc1_fp_output
    )

    fc1_mult = make_q31_multiplier(
        b1_scale
        / float(fc1_output_scale)
    )

    fc1_q = saturate_int8(
        requant_q31(
            fc1_acc,
            fc1_mult[None, :],
        )
    )

    # ============================================================
    # GELU LUT -> true FC2 input scale
    # ============================================================
    true_fc2_input_fp = (
        mlp["fc2_input"]
        .astype(np.float32)
    )

    fc2_input_scale = symmetric_scale(
        true_fc2_input_fp
    )

    gelu_lut = np.zeros(
        256,
        dtype=np.int8,
    )

    for q in range(-128, 128):
        real_x = (
            q
            * float(fc1_output_scale)
        )

        real_y = float(
            F.gelu(
                torch.tensor(
                    real_x,
                    dtype=torch.float32,
                ),
                approximate="none",
            ).item()
        )

        gelu_lut[q & 0xff] = (
            quantize_int8(
                np.array([real_y]),
                fc2_input_scale,
            )[0]
        )

    fc2_input_q = (
        gelu_lut[
            fc1_q.view(np.uint8)
        ]
    )

    # ============================================================
    # TRUE FC2
    # ============================================================
    w2 = (
        mlp["fc2_weight"]
        .astype(np.float32)
    )

    b2 = (
        mlp["fc2_bias"]
        .astype(np.float32)
    )

    fc2_fp_output = (
        mlp["fc2_output"]
        .astype(np.float32)
    )

    w2_max = np.max(
        np.abs(w2),
        axis=1,
    )

    w2_scale = np.where(
        w2_max == 0,
        1.0,
        w2_max / QMAX,
    ).astype(np.float32)

    w2_q = saturate_int8(
        np.rint(
            w2
            / w2_scale[:, None]
        )
    )

    b2_scale = (
        float(fc2_input_scale)
        * w2_scale
    )

    b2_q = np.rint(
        b2 / b2_scale
    ).astype(np.int32)

    fc2_acc = (
        fc2_input_q.astype(np.int32)
        @
        w2_q.astype(np.int32).T
    )

    fc2_acc += b2_q[None, :]

    mlp_scale = symmetric_scale(
        fc2_fp_output
    )

    fc2_mult = make_q31_multiplier(
        b2_scale
        / float(mlp_scale)
    )

    mlp_q = saturate_int8(
        requant_q31(
            fc2_acc,
            fc2_mult[None, :],
        )
    )

    # ============================================================
    # Residual 2
    # ============================================================
    final_ref_scale = float(
        symmetric_scale(
            block["output"]
        )
    )

    final_scale = max(
        residual1_scale,
        float(mlp_scale),
        final_ref_scale,
    )

    r1_to_final = int(
        make_q31_multiplier(
            residual1_scale
            / final_scale
        )
    )

    mlp_to_final = int(
        make_q31_multiplier(
            float(mlp_scale)
            / final_scale
        )
    )

    final_q = saturate_int8(
        requant_q31(
            residual1_q,
            r1_to_final,
        )
        +
        requant_q31(
            mlp_q,
            mlp_to_final,
        )
    )

    final_deq = (
        final_q.astype(np.float32)
        * final_scale
    )

    # ============================================================
    # Quality reporting
    # ============================================================
    ln1_mae = float(
        np.mean(
            np.abs(
                ln1_deq
                -
                block["norm1"]
            )
        )
    )

    residual1_deq = (
        residual1_q.astype(np.float32)
        * residual1_scale
    )

    r1_mae = float(
        np.mean(
            np.abs(
                residual1_deq
                -
                block["residual1"]
            )
        )
    )

    ln2_deq = (
        ln2_q.astype(np.float32)
        * float(fc1_input_scale)
    )

    ln2_mae = float(
        np.mean(
            np.abs(
                ln2_deq
                -
                block["norm2"]
            )
        )
    )

    final_mae = float(
        np.mean(
            np.abs(
                final_deq
                -
                block["output"]
            )
        )
    )

    final_max = float(
        np.max(
            np.abs(
                final_deq
                -
                block["output"]
            )
        )
    )

    final_sqnr = sqnr_db(
        block["output"],
        final_deq,
    )

    # ============================================================
    # Save EVERYTHING needed for later C export.
    # ============================================================
    np.savez(
        OUTPUT,

        x0_q=x0_q,
        x0_scale=np.array(
            x0_scale,
            dtype=np.float32,
        ),

        ln1_q=ln1_q,
        ln1_scale=np.array(
            ln1_scale,
            dtype=np.float32,
        ),
        ln1_affine_a=ln1_a,
        ln1_affine_b=ln1_b,
        ln1_eps_term=np.array(
            ln1_eps_term,
            dtype=np.int64,
        ),

        qkv_acc=qkv_acc,
        qkv_q=qkv_out_q,

        q_q=q_q,
        k_q=k_q,
        v_q=v_q,

        qkt_acc=qkt_acc,
        prob_q=prob_q,

        av_acc=av_acc,
        av_q=av_q,

        proj_acc=proj_acc,
        proj_q=proj_q,

        residual1_q=residual1_q,
        residual1_scale=np.array(
            residual1_scale,
            dtype=np.float32,
        ),
        x0_to_r1=np.array(
            x0_to_r1,
            dtype=np.int64,
        ),
        proj_to_r1=np.array(
            proj_to_r1,
            dtype=np.int64,
        ),

        ln2_q=ln2_q,
        ln2_scale=np.array(
            fc1_input_scale,
            dtype=np.float32,
        ),
        ln2_affine_a=ln2_a,
        ln2_affine_b=ln2_b,
        ln2_eps_term=np.array(
            ln2_eps_term,
            dtype=np.int64,
        ),

        fc1_weight_q=w1_q,
        fc1_bias_q=b1_q,
        fc1_multiplier=fc1_mult,
        fc1_acc=fc1_acc,
        fc1_q=fc1_q,
        fc1_output_scale=np.array(
            fc1_output_scale,
            dtype=np.float32,
        ),

        gelu_lut=gelu_lut,

        fc2_input_q=fc2_input_q,
        fc2_weight_q=w2_q,
        fc2_bias_q=b2_q,
        fc2_multiplier=fc2_mult,
        fc2_acc=fc2_acc,
        mlp_q=mlp_q,
        mlp_scale=np.array(
            mlp_scale,
            dtype=np.float32,
        ),

        r1_to_final=np.array(
            r1_to_final,
            dtype=np.int64,
        ),
        mlp_to_final=np.array(
            mlp_to_final,
            dtype=np.int64,
        ),

        final_q=final_q,
        final_scale=np.array(
            final_scale,
            dtype=np.float32,
        ),
    )

    print("Saved:", OUTPUT)

    print()
    print("Connected Block0")
    print("--------------------------------")
    print("X0       :", x0_q.shape)
    print("LN1      :", ln1_q.shape)
    print("QKV      :", qkv_out_q.shape)
    print("QK^T     :", qkt_acc.shape)
    print("Softmax  :", prob_q.shape)
    print("AV       :", av_q.shape)
    print("Projection:", proj_q.shape)
    print("Residual1:", residual1_q.shape)
    print("LN2      :", ln2_q.shape)
    print("FC1      :", fc1_q.shape)
    print("FC2      :", mlp_q.shape)
    print("Final    :", final_q.shape)

    print()
    print("LayerNorm / residual quality")
    print("--------------------------------")
    print("LN1 MAE      :", ln1_mae)
    print("Residual1 MAE:", r1_mae)
    print("LN2 MAE      :", ln2_mae)

    print()
    print("Final Block0 quality")
    print("--------------------------------")
    print("MAE       :", final_mae)
    print("Max error :", final_max)
    print("SQNR (dB) :", final_sqnr)

    print()
    print("Scales")
    print("--------------------------------")
    print("X0       :", float(x0_scale))
    print("LN1      :", ln1_scale)
    print("Residual1:", residual1_scale)
    print("LN2      :", float(fc1_input_scale))
    print("FC1 out  :", float(fc1_output_scale))
    print("FC2 out  :", float(mlp_scale))
    print("Final    :", final_scale)

    print()
    print("First final INT8 values")
    print("--------------------------------")
    print(final_q[0, :16])


if __name__ == "__main__":
    main()
