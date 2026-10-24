from pathlib import Path
import numpy as np


QKV_Q = Path("results/qkv_w8a8.npz")
QKV_FP32 = Path("results/qkv_fp32.npz")
OUTPUT = Path("results/qkt_w8a8.npz")

HEADS = 3
TOKENS = 197
HEAD_DIM = 64


def sqnr_db(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum((ref - test) ** 2)

    if noise == 0:
        return float("inf")

    return 10.0 * np.log10(signal / noise)


def main():
    qd = np.load(QKV_Q)
    fd = np.load(QKV_FP32)

    q = qd["q_q"].astype(np.int8)
    k = qd["k_q"].astype(np.int8)

    q_fp32 = fd["q"].astype(np.float32)
    k_fp32 = fd["k"].astype(np.float32)

    qkv_output_scale = float(
        qd["output_scale"]
    )

    assert q.shape == (HEADS, TOKENS, HEAD_DIM)
    assert k.shape == (HEADS, TOKENS, HEAD_DIM)

    # ------------------------------------------------------------
    # Integer QK^T
    #
    # For each head:
    #
    # [197 x 64] x [64 x 197]
    # -> [197 x 197]
    # ------------------------------------------------------------
    acc = np.empty(
        (HEADS, TOKENS, TOKENS),
        dtype=np.int32,
    )

    for h in range(HEADS):
        acc[h] = (
            q[h].astype(np.int32)
            @
            k[h].astype(np.int32).T
        )

    # ------------------------------------------------------------
    # Real attention score:
    #
    # score = QK^T / sqrt(64)
    #       = QK^T / 8
    #
    # q_real = q_int8 * qkv_output_scale
    # k_real = k_int8 * qkv_output_scale
    #
    # therefore:
    #
    # score_real =
    # acc * output_scale^2 / 8
    # ------------------------------------------------------------
    score_scale = (
        qkv_output_scale
        * qkv_output_scale
        / 8.0
    )

    score_quant_fp32 = (
        acc.astype(np.float32)
        * score_scale
    )

    # Original FP32 attention score.
    score_ref_fp32 = np.empty(
        (HEADS, TOKENS, TOKENS),
        dtype=np.float32,
    )

    for h in range(HEADS):
        score_ref_fp32[h] = (
            q_fp32[h]
            @
            k_fp32[h].T
        ) / 8.0

    error = (
        score_quant_fp32
        - score_ref_fp32
    )

    mae = float(
        np.mean(np.abs(error))
    )

    max_error = float(
        np.max(np.abs(error))
    )

    sqnr = sqnr_db(
        score_ref_fp32,
        score_quant_fp32,
    )

    # Useful for the next Softmax stage.
    row_max = np.max(
        acc,
        axis=2,
    )

    shifted_acc = (
        acc
        - row_max[:, :, None]
    )

    np.savez(
        OUTPUT,

        q_q=q,
        k_q=k,

        accumulator=acc,

        qkv_output_scale=np.array(
            qkv_output_scale,
            dtype=np.float32,
        ),

        score_scale=np.array(
            score_scale,
            dtype=np.float32,
        ),

        score_quant_fp32=score_quant_fp32,
        score_ref_fp32=score_ref_fp32,

        row_max=row_max,
        shifted_acc=shifted_acc,
    )

    macs = (
        HEADS
        * TOKENS
        * TOKENS
        * HEAD_DIM
    )

    xqmac8 = macs // 8

    print("Saved:", OUTPUT)

    print()
    print("QK^T dimensions")
    print("--------------------------------")
    print("Q          :", q.shape)
    print("K          :", k.shape)
    print("Accumulator:", acc.shape)

    print()
    print("Workload")
    print("--------------------------------")
    print("MACs       :", f"{macs:,}")
    print("Xqmac8     :", f"{xqmac8:,}")
    print("MAC/output :", HEAD_DIM)
    print(
        "Xqmac/output:",
        HEAD_DIM // 8,
    )

    print()
    print("Scaling")
    print("--------------------------------")
    print(
        "QKV output scale:",
        qkv_output_scale,
    )
    print(
        "Attention score scale:",
        score_scale,
    )

    print()
    print("QK^T quantization quality")
    print("--------------------------------")
    print("MAE       :", mae)
    print("Max error :", max_error)
    print("SQNR (dB) :", sqnr)

    print()
    print("Accumulator range")
    print("--------------------------------")
    print(
        int(np.min(acc)),
        "to",
        int(np.max(acc)),
    )

    print()
    print("Shifted score range")
    print("--------------------------------")
    print(
        int(np.min(shifted_acc)),
        "to",
        int(np.max(shifted_acc)),
    )

    print()
    print("First head, first row, first 8 ACC:")
    print(
        acc[0, 0, :8]
    )

    print()
    print("First head, first row, first 8 scores:")
    print(
        score_quant_fp32[0, 0, :8]
    )


if __name__ == "__main__":
    main()
