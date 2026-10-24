from pathlib import Path
import numpy as np


INPUT = Path("results/qkt_w8a8.npz")
OUTPUT = Path("results/softmax0_int8.npz")

HEADS = 3
TOKENS = 197

LUT_SIZE = 256

# Scores smaller than max-8 have negligible exp().
CLIP_REAL = 8.0

# exp LUT is represented as unsigned Q15.
EXP_Q = 15
EXP_MAX = (1 << EXP_Q) - 1

# Signed INT8 probability representation:
# probability ~= prob_q / 127
PROB_QMAX = 127


def stable_softmax(x):
    x = x.astype(np.float64)

    x = (
        x
        - np.max(
            x,
            axis=-1,
            keepdims=True,
        )
    )

    e = np.exp(x)

    return (
        e
        /
        np.sum(
            e,
            axis=-1,
            keepdims=True,
        )
    )


def main():
    d = np.load(INPUT)

    acc = (
        d["accumulator"]
        .astype(np.int32)
    )

    score_scale = float(
        d["score_scale"]
    )

    score_ref_fp32 = (
        d["score_ref_fp32"]
        .astype(np.float32)
    )

    score_quant_fp32 = (
        d["score_quant_fp32"]
        .astype(np.float32)
    )

    assert acc.shape == (
        HEADS,
        TOKENS,
        TOKENS,
    )

    # ============================================================
    # 1. Subtract row maximum
    # ============================================================
    row_max = np.max(
        acc,
        axis=2,
        keepdims=True,
    )

    shifted = (
        acc.astype(np.int64)
        - row_max.astype(np.int64)
    )

    assert np.max(shifted) == 0

    # Distance from row maximum.
    distance = -shifted

    # ============================================================
    # 2. Convert real [-8, 0] range into 256 LUT bins.
    #
    # shifted_score_real =
    #     shifted_acc * score_scale
    #
    # clip_steps represents approximately -8.0.
    # ============================================================
    clip_steps = int(
        np.rint(
            CLIP_REAL
            / score_scale
        )
    )

    clip_steps = max(
        clip_steps,
        1,
    )

    # ============================================================
    # 3. Build exp LUT
    #
    # LUT[0]   = exp(0)
    # LUT[255] = exp(-8)
    #
    # represented in unsigned Q15.
    # ============================================================
    lut_x = (
        -CLIP_REAL
        * np.arange(
            LUT_SIZE,
            dtype=np.float64,
        )
        / (LUT_SIZE - 1)
    )

    exp_lut = np.rint(
        np.exp(lut_x)
        * EXP_MAX
    )

    exp_lut = np.clip(
        exp_lut,
        1,
        EXP_MAX,
    ).astype(np.uint16)

    # ============================================================
    # 4. Integer score -> LUT index
    #
    # index ~= distance / clip_steps * 255
    #
    # round-to-nearest
    # ============================================================
    lut_index = (
        distance * (LUT_SIZE - 1)
        + clip_steps // 2
    ) // clip_steps

    lut_index = np.clip(
        lut_index,
        0,
        LUT_SIZE - 1,
    ).astype(np.uint16)

    exp_q15 = (
        exp_lut[lut_index]
        .astype(np.int64)
    )

    # ============================================================
    # 5. Integer normalization
    #
    # p_q =
    #
    #      exp_i
    # ---------------- * 127
    # sum_j(exp_j)
    #
    # ============================================================
    exp_sum = np.sum(
        exp_q15,
        axis=2,
        keepdims=True,
        dtype=np.int64,
    )

    numerator = (
        exp_q15
        * PROB_QMAX
    )

    prob_q = (
        numerator
        + exp_sum // 2
    ) // exp_sum

    prob_q = np.clip(
        prob_q,
        0,
        PROB_QMAX,
    ).astype(np.int8)

    # ============================================================
    # 6. Convert integer output back to probability
    # ============================================================
    prob_int = (
        prob_q.astype(np.float32)
        / float(PROB_QMAX)
    )

    # Reference A:
    # FP32 Q and K
    prob_ref_fp32 = stable_softmax(
        score_ref_fp32
    )

    # Reference B:
    # exact softmax applied to quantized QK scores.
    prob_quant_score_ref = stable_softmax(
        score_quant_fp32
    )

    # ============================================================
    # 7. Quality relative to quantized-score softmax
    #
    # This isolates the Softmax approximation itself.
    # ============================================================
    error_quant = (
        prob_int
        - prob_quant_score_ref
    )

    mae_quant = float(
        np.mean(
            np.abs(error_quant)
        )
    )

    max_quant = float(
        np.max(
            np.abs(error_quant)
        )
    )

    # ============================================================
    # 8. End-to-end quality relative to original FP32 Q/K
    # ============================================================
    error_fp32 = (
        prob_int
        - prob_ref_fp32
    )

    mae_fp32 = float(
        np.mean(
            np.abs(error_fp32)
        )
    )

    max_fp32 = float(
        np.max(
            np.abs(error_fp32)
        )
    )

    # Top-1 attention position agreement.
    ref_argmax = np.argmax(
        prob_ref_fp32,
        axis=2,
    )

    int_argmax = np.argmax(
        prob_q,
        axis=2,
    )

    top1_agreement = float(
        np.mean(
            ref_argmax
            ==
            int_argmax
        )
    )

    row_sum_q = np.sum(
        prob_q.astype(np.int32),
        axis=2,
    )

    row_sum_real = np.sum(
        prob_int,
        axis=2,
    )

    # ============================================================
    # Save
    # ============================================================
    np.savez(
        OUTPUT,

        exp_lut=exp_lut,

        clip_steps=np.array(
            clip_steps,
            dtype=np.int32,
        ),

        score_scale=np.array(
            score_scale,
            dtype=np.float32,
        ),

        row_max=row_max.squeeze(2),

        lut_index=lut_index,

        exp_q15=exp_q15,

        exp_sum=exp_sum.squeeze(2),

        prob_q=prob_q,

        prob_int=prob_int,

        prob_ref_fp32=prob_ref_fp32,

        prob_quant_score_ref=prob_quant_score_ref,
    )

    print("Saved:", OUTPUT)

    print()
    print("Integer Softmax")
    print("--------------------------------")
    print("Input :", acc.shape)
    print("Output:", prob_q.shape)

    print()
    print("LUT configuration")
    print("--------------------------------")
    print("Entries     :", LUT_SIZE)
    print("Clip range  :", -CLIP_REAL, "to 0")
    print("Score scale :", score_scale)
    print("Clip steps  :", clip_steps)
    print("EXP Q       :", EXP_Q)
    print("Prob QMAX   :", PROB_QMAX)

    print()
    print("LUT endpoints")
    print("--------------------------------")
    print("exp(0)  :", int(exp_lut[0]))
    print("exp(-8) :", int(exp_lut[-1]))

    print()
    print("Softmax approximation error")
    print("(vs exact Softmax of quantized scores)")
    print("--------------------------------")
    print("MAE       :", mae_quant)
    print("Max error :", max_quant)

    print()
    print("End-to-end attention probability error")
    print("(vs original FP32 Q/K)")
    print("--------------------------------")
    print("MAE       :", mae_fp32)
    print("Max error :", max_fp32)
    print(
        "Top1 agreement:",
        top1_agreement,
    )

    print()
    print("INT8 probability row sums")
    print("--------------------------------")
    print(
        "Integer range:",
        int(np.min(row_sum_q)),
        "to",
        int(np.max(row_sum_q)),
    )

    print(
        "Real range:",
        float(np.min(row_sum_real)),
        "to",
        float(np.max(row_sum_real)),
    )

    print()
    print("First row first 16 probabilities")
    print("--------------------------------")
    print(prob_q[0, 0, :16])

    print()
    print("First row FP32 approximation")
    print("--------------------------------")
    print(prob_int[0, 0, :16])


if __name__ == "__main__":
    main()
