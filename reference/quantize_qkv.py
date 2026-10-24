from pathlib import Path
import numpy as np


INPUT = Path("results/qkv_fp32.npz")
OUTPUT = Path("results/qkv_w8a8.npz")

QMAX = 127
SHIFT = 31


def scale_symmetric(x):
    m = float(np.max(np.abs(x)))

    if m == 0:
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


def sqnr(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    signal = np.sum(ref * ref)
    noise = np.sum((ref - test) ** 2)

    if noise == 0:
        return float("inf")

    return 10 * np.log10(
        signal / noise
    )


def main():
    d = np.load(INPUT)

    x = d["input"].astype(np.float32)
    w = d["weight"].astype(np.float32)
    b = d["bias"].astype(np.float32)
    y = d["output"].astype(np.float32)

    # Input quantization
    sx = scale_symmetric(x)

    xq = np.clip(
        np.rint(x / sx),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # Per-output-channel weight quantization
    wmax = np.max(
        np.abs(w),
        axis=1,
    )

    sw = np.where(
        wmax == 0,
        1.0,
        wmax / QMAX,
    ).astype(np.float32)

    wq = np.clip(
        np.rint(
            w / sw[:, None]
        ),
        -QMAX,
        QMAX,
    ).astype(np.int8)

    # Bias
    sb = sx * sw

    bq = np.rint(
        b / sb
    ).astype(np.int32)

    # INT8 GEMM
    acc = (
        xq.astype(np.int32)
        @
        wq.astype(np.int32).T
    )

    acc += bq[None, :]

    # Accumulator quality
    acc_fp = (
        acc.astype(np.float32)
        * sb[None, :]
    )

    acc_mae = float(
        np.mean(
            np.abs(acc_fp - y)
        )
    )

    acc_max = float(
        np.max(
            np.abs(acc_fp - y)
        )
    )

    acc_sqnr = sqnr(
        y,
        acc_fp,
    )

    # Output scale
    sy = scale_symmetric(y)

    multiplier = (
        sb.astype(np.float64)
        / float(sy)
    )

    multiplier_int = np.rint(
        multiplier
        * (1 << SHIFT)
    ).astype(np.int64)

    yq = fixed_requant(
        acc,
        multiplier_int,
    )

    y_deq = (
        yq.astype(np.float32)
        * sy
    )

    final_mae = float(
        np.mean(
            np.abs(y_deq - y)
        )
    )

    final_max = float(
        np.max(
            np.abs(y_deq - y)
        )
    )

    final_sqnr = sqnr(
        y,
        y_deq,
    )

    # Split Q/K/V from quantized QKV output.
    split = yq.reshape(
        197,
        3,
        3,
        64,
    )

    q_q = split[:, 0].transpose(1, 0, 2)
    k_q = split[:, 1].transpose(1, 0, 2)
    v_q = split[:, 2].transpose(1, 0, 2)

    np.savez(
        OUTPUT,

        input_q=xq,
        weight_q=wq,
        bias_q=bq,

        accumulator=acc,

        input_scale=np.array(
            sx,
            dtype=np.float32,
        ),

        weight_scale=sw,

        output_scale=np.array(
            sy,
            dtype=np.float32,
        ),

        multiplier_int=multiplier_int,

        shift=np.array(
            SHIFT,
            dtype=np.int32,
        ),

        output_q=yq,

        q_q=q_q,
        k_q=k_q,
        v_q=v_q,

        output_fp32=y,
    )

    print("Saved:", OUTPUT)

    print()
    print("QKV W8A8")
    print("--------------------------------")
    print("Input :", xq.shape)
    print("Weight:", wq.shape)
    print("Output:", yq.shape)
    print("Q:", q_q.shape)
    print("K:", k_q.shape)
    print("V:", v_q.shape)

    print()
    print("Scales")
    print("--------------------------------")
    print("Input :", float(sx))
    print(
        "Weight:",
        float(np.min(sw)),
        "to",
        float(np.max(sw)),
    )
    print("Output:", float(sy))

    print()
    print("INT32 accumulator quality")
    print("--------------------------------")
    print("MAE :", acc_mae)
    print("Max :", acc_max)
    print("SQNR:", acc_sqnr)

    print()
    print("Final W8A8 quality")
    print("--------------------------------")
    print("MAE :", final_mae)
    print("Max :", final_max)
    print("SQNR:", final_sqnr)

    print()
    print("Accumulator range:")
    print(
        int(np.min(acc)),
        "to",
        int(np.max(acc)),
    )

    print()
    print("First Q INT8:")
    print(q_q[0, 0, :8])


if __name__ == "__main__":
    main()
