from pathlib import Path
import numpy as np


INPUT = Path("results/proj0_w8a8.npz")

HEADER = Path("src/generated/proj0_data.h")
SOURCE = Path("src/generated/proj0_data.c")

TOKENS = 197
IN = 192
OUT = 192


def write_1d(
    f,
    name,
    ctype,
    a,
    per_line=16,
):
    a = np.asarray(a).reshape(-1)

    f.write(
        f"const {ctype} "
        f"{name}[{len(a)}] = {{\n"
    )

    for i in range(
        0,
        len(a),
        per_line,
    ):
        f.write(
            "    "
            + ", ".join(
                str(int(v))
                for v
                in a[i:i+per_line]
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
    m = d["multiplier_int"]
    acc = d["accumulator"]
    out = d["output_q"]

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEADER.write_text(
f"""#ifndef PROJ0_DATA_H
#define PROJ0_DATA_H

#include <stdint.h>

#define PROJ0_TOKENS {TOKENS}
#define PROJ0_IN_FEATURES {IN}
#define PROJ0_OUT_FEATURES {OUT}
#define PROJ0_REQUANT_SHIFT 31

extern const int8_t
proj0_input[PROJ0_TOKENS][PROJ0_IN_FEATURES];

extern const int8_t
proj0_weight[PROJ0_OUT_FEATURES][PROJ0_IN_FEATURES];

extern const int32_t
proj0_bias[PROJ0_OUT_FEATURES];

extern const int64_t
proj0_multiplier[PROJ0_OUT_FEATURES];

extern const int32_t
proj0_golden_acc[PROJ0_TOKENS][PROJ0_OUT_FEATURES];

extern const int8_t
proj0_golden_output[PROJ0_TOKENS][PROJ0_OUT_FEATURES];

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "proj0_data.h"\n\n'
        )

        write_2d(
            f,
            "proj0_input",
            "int8_t",
            x,
            aligned=True,
        )

        write_2d(
            f,
            "proj0_weight",
            "int8_t",
            w,
            aligned=True,
        )

        write_1d(
            f,
            "proj0_bias",
            "int32_t",
            b,
        )

        write_1d(
            f,
            "proj0_multiplier",
            "int64_t",
            m,
            per_line=8,
        )

        write_2d(
            f,
            "proj0_golden_acc",
            "int32_t",
            acc,
        )

        write_2d(
            f,
            "proj0_golden_output",
            "int8_t",
            out,
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
