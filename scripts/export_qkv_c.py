from pathlib import Path
import numpy as np


INPUT = Path("results/qkv_w8a8.npz")

HEADER = Path("src/generated/qkv_data.h")
SOURCE = Path("src/generated/qkv_data.c")

TOKENS = 197
IN = 192
OUT = 576
HEADS = 3
HEAD_DIM = 64


def write_1d(f, name, ctype, a, per_line=16):
    a = np.asarray(a).reshape(-1)

    f.write(
        f"const {ctype} {name}[{len(a)}] = {{\n"
    )

    for i in range(0, len(a), per_line):
        f.write(
            "    "
            + ", ".join(
                str(int(v))
                for v in a[i:i+per_line]
            )
            + ",\n"
        )

    f.write("};\n\n")


def write_2d(
    f,
    name,
    ctype,
    a,
    aligned=False,
):
    a = np.asarray(a)

    rows, cols = a.shape

    attr = (
        " __attribute__((aligned(8)))"
        if aligned
        else ""
    )

    f.write(
        f"const {ctype} "
        f"{name}[{rows}][{cols}]"
        f"{attr} = {{\n"
    )

    for row in a:
        f.write(
            "    {"
            + ", ".join(
                str(int(v))
                for v in row
            )
            + "},\n"
        )

    f.write("};\n\n")


def main():
    d = np.load(INPUT)

    x = d["input_q"]
    w = d["weight_q"]
    b = d["bias_q"]
    mult = d["multiplier_int"]
    acc = d["accumulator"]
    out = d["output_q"]

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEADER.write_text(
f"""#ifndef QKV_DATA_H
#define QKV_DATA_H

#include <stdint.h>

#define QKV_TOKENS       {TOKENS}
#define QKV_IN_FEATURES  {IN}
#define QKV_OUT_FEATURES {OUT}
#define QKV_HEADS        {HEADS}
#define QKV_HEAD_DIM     {HEAD_DIM}
#define QKV_REQUANT_SHIFT 31

extern const int8_t
qkv_input[QKV_TOKENS][QKV_IN_FEATURES];

extern const int8_t
qkv_weight[QKV_OUT_FEATURES][QKV_IN_FEATURES];

extern const int32_t
qkv_bias[QKV_OUT_FEATURES];

extern const int64_t
qkv_multiplier[QKV_OUT_FEATURES];

extern const int32_t
qkv_golden_acc[QKV_TOKENS][QKV_OUT_FEATURES];

extern const int8_t
qkv_golden_output[QKV_TOKENS][QKV_OUT_FEATURES];

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "qkv_data.h"\n\n'
        )

        write_2d(
            f,
            "qkv_input",
            "int8_t",
            x,
            aligned=True,
        )

        write_2d(
            f,
            "qkv_weight",
            "int8_t",
            w,
            aligned=True,
        )

        write_1d(
            f,
            "qkv_bias",
            "int32_t",
            b,
        )

        write_1d(
            f,
            "qkv_multiplier",
            "int64_t",
            mult,
            per_line=8,
        )

        write_2d(
            f,
            "qkv_golden_acc",
            "int32_t",
            acc,
        )

        write_2d(
            f,
            "qkv_golden_output",
            "int8_t",
            out,
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
