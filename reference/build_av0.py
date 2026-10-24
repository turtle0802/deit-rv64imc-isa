from pathlib import Path
import numpy as np


SOFTMAX = Path("results/softmax0_int8.npz")
QKV = Path("results/qkv_w8a8.npz")
QKV_FP32 = Path("results/qkv_fp32.npz")
QKT = Path("results/qkt_w8a8.npz")

OUTPUT = Path("results/av0_w8a8.npz")

HEADS = 3
TOKENS = 197
HEAD_DIM = 64
PAD_TOKENS = 200

QMAX = 127
SHIFT = 31


def symmetric_scale(x):
    m = float(np.max(np.abs(x)))

    if m == 0:
        return np.float32(1.0)

    return np.float32(m / QMAX)


def fixed_requant(acc, multiplier):
    product = (
        acc.astype(np.int64)
        * np.int64(multiplier)
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


def stable_softmax(x):
    x = x.astype(np.float64)

    x -= np.max(
        x,
        axis=-1,
        keepdims=True,
    )

    e = np.exp(x)

    return e / np.sum(
        e,
        axis=-1,
        keepdims=True,
    )


def sqnr_db(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum((ref - test) ** 2)

    if noise == 0:
        return float("inf")

    return 10 * np.log10(signal / noise)


def main():
    sd = np.load(SOFTMAX)
    qd = np.load(QKV)
    fd = np.load(QKV_FP32)
    td = np.load(QKT)

    prob_q = (
        sd["prob_q"]
        .astype(np.int8)
    )

    v_q = (
        qd["v_q"]
        .astype(np.int8)
    )

    v_fp32 = (
        fd["v"]
        .astype(np.float32)
    )

    score_ref_fp32 = (
        td["score_ref_fp32"]
        .astype(np.float32)
    )

    v_scale = float(
        qd["output_scale"]
    )

    assert prob_q.shape == (
        HEADS,
        TOKENS,
        TOKENS,
    )

    assert v_q.shape == (
        HEADS,
        TOKENS,
        HEAD_DIM,
    )

    # ============================================================
    # INT8 Softmax x INT8 V -> INT32
    # ============================================================
    acc = np.empty(
        (HEADS, TOKENS, HEAD_DIM),
        dtype=np.int32,
    )

    for h in range(HEADS):
        acc[h] = (
            prob_q[h].astype(np.int32)
            @
            v_q[h].astype(np.int32)
        )

    # prob ~= prob_q / 127
    # V ~= v_q * v_scale
    #
    # Therefore accumulator real scale:
    #
    # v_scale / 127
    acc_scale = (
        v_scale / 127.0
    )

    acc_fp32 = (
        acc.astype(np.float32)
        * acc_scale
    )

    # ============================================================
    # FP32 reference attention x V
    # ============================================================
    prob_ref = stable_softmax(
        score_ref_fp32
    )

    av_ref = np.empty(
        (HEADS, TOKENS, HEAD_DIM),
        dtype=np.float32,
    )

    for h in range(HEADS):
        av_ref[h] = (
            prob_ref[h]
            @
            v_fp32[h]
        )

    # ============================================================
    # Quantize AV output to INT8
    # ============================================================
    output_scale = symmetric_scale(
        av_ref
    )

    multiplier = (
        acc_scale
        / float(output_scale)
    )

    multiplier_int = int(
        np.rint(
            multiplier
            * (1 << SHIFT)
        )
    )

    if not (
        -(1 << 31)
        <= multiplier_int
        < (1 << 31)
    ):
        raise RuntimeError(
            "AV multiplier does not fit int32"
        )

    output_q = fixed_requant(
        acc,
        multiplier_int,
    )

    output_fp32 = (
        output_q.astype(np.float32)
        * output_scale
    )

    # ============================================================
    # Accuracy
    # ============================================================
    acc_mae = float(
        np.mean(
            np.abs(
                acc_fp32 - av_ref
            )
        )
    )

    final_mae = float(
        np.mean(
            np.abs(
                output_fp32 - av_ref
            )
        )
    )

    final_max = float(
        np.max(
            np.abs(
                output_fp32 - av_ref
            )
        )
    )

    final_sqnr = sqnr_db(
        av_ref,
        output_fp32,
    )

    # ============================================================
    # Concatenate heads:
    #
    # [3,197,64] -> [197,192]
    #
    # This becomes projection input.
    # ============================================================
    concat_q = (
        output_q
        .transpose(1, 0, 2)
        .reshape(TOKENS, HEADS * HEAD_DIM)
    )

    # ============================================================
    # Padding for Xqmac8
    #
    # probability:
    # [3,197,197] -> [3,197,200]
    #
    # V transpose:
    # [3,197,64]
    # -> [3,64,197]
    # -> [3,64,200]
    # ============================================================
    prob_padded = np.zeros(
        (HEADS, TOKENS, PAD_TOKENS),
        dtype=np.int8,
    )

    prob_padded[:, :, :TOKENS] = prob_q

    v_t_padded = np.zeros(
        (HEADS, HEAD_DIM, PAD_TOKENS),
        dtype=np.int8,
    )

    v_t_padded[:, :, :TOKENS] = (
        v_q.transpose(0, 2, 1)
    )

    # Verify padding does not change result.
    acc_padded = np.empty_like(acc)

    for h in range(HEADS):
        acc_padded[h] = (
            prob_padded[h].astype(np.int32)
            @
            v_t_padded[h].astype(np.int32).T
        )

    assert np.array_equal(
        acc,
        acc_padded,
    )

    np.savez(
        OUTPUT,

        prob_q=prob_q,
        v_q=v_q,

        prob_padded=prob_padded,
        v_t_padded=v_t_padded,

        accumulator=acc,

        acc_scale=np.array(
            acc_scale,
            dtype=np.float32,
        ),

        output_scale=np.array(
            output_scale,
            dtype=np.float32,
        ),

        multiplier_int=np.array(
            multiplier_int,
            dtype=np.int64,
        ),

        output_q=output_q,
        concat_q=concat_q,

        av_ref_fp32=av_ref,
    )

    useful_macs = (
        HEADS
        * TOKENS
        * HEAD_DIM
        * TOKENS
    )

    padded_macs = (
        HEADS
        * TOKENS
        * HEAD_DIM
        * PAD_TOKENS
    )

    xqmac_count = (
        padded_macs // 8
    )

    outputs = (
        HEADS
        * TOKENS
        * HEAD_DIM
    )

    print("Saved:", OUTPUT)

    print()
    print("Attention x V")
    print("--------------------------------")
    print("Probability :", prob_q.shape)
    print("V           :", v_q.shape)
    print("Accumulator :", acc.shape)
    print("Output      :", output_q.shape)
    print("Concat      :", concat_q.shape)

    print()
    print("Workload")
    print("--------------------------------")
    print(
        "Useful MACs :",
        f"{useful_macs:,}",
    )
    print(
        "Padded MACs :",
        f"{padded_macs:,}",
    )
    print(
        "Xqmac8      :",
        f"{xqmac_count:,}",
    )
    print(
        "Xrequant    :",
        f"{outputs:,}",
    )
    print(
        "Custom total:",
        f"{xqmac_count + outputs:,}",
    )

    print()
    print("Scales")
    print("--------------------------------")
    print("V scale     :", v_scale)
    print("ACC scale   :", acc_scale)
    print("Output scale:", float(output_scale))
    print("Multiplier  :", multiplier)
    print("Multiplier Q31:", multiplier_int)

    print()
    print("Accuracy")
    print("--------------------------------")
    print(
        "ACC-domain MAE:",
        acc_mae,
    )
    print("Final MAE      :", final_mae)
    print("Final max error:", final_max)
    print("Final SQNR (dB):", final_sqnr)

    print()
    print("Accumulator range")
    print("--------------------------------")
    print(
        int(np.min(acc)),
        "to",
        int(np.max(acc)),
    )

    print()
    print("First AV output:")
    print(output_q[0, 0, :8])

    print()
    print("First concatenated projection input:")
    print(concat_q[0, :16])


if __name__ == "__main__":
    main()
