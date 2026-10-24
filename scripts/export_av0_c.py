from pathlib import Path
import numpy as np


INPUT = Path("results/av0_w8a8.npz")

HEADER = Path("src/generated/av0_data.h")
SOURCE = Path("src/generated/av0_data.c")

HEADS = 3
TOKENS = 197
PAD = 200
DIM = 64


def write_3d(f, name, ctype, a, aligned=False):
    a = np.asarray(a)

    d0, d1, d2 = a.shape

    attr = (
        " __attribute__((aligned(8)))"
        if aligned
        else ""
    )

    f.write(
        f"const {ctype} "
        f"{name}[{d0}][{d1}][{d2}]"
        f"{attr} = {{\n"
    )

    for i in range(d0):
        f.write("    {\n")

        for j in range(d1):
            f.write(
                "        {"
                + ", ".join(
                    str(int(v))
                    for v in a[i, j]
                )
                + "},\n"
            )

        f.write("    },\n")

    f.write("};\n\n")


def main():
    d = np.load(INPUT)

    prob = d["prob_padded"]
    vt = d["v_t_padded"]

    acc = d["accumulator"]
    out = d["output_q"]

    multiplier = int(
        d["multiplier_int"]
    )

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEADER.write_text(
f"""#ifndef AV0_DATA_H
#define AV0_DATA_H

#include <stdint.h>

#define AV0_HEADS {HEADS}
#define AV0_TOKENS {TOKENS}
#define AV0_PAD_TOKENS {PAD}
#define AV0_HEAD_DIM {DIM}

#define AV0_REQUANT_SHIFT 31

extern const int8_t
av0_prob[
    AV0_HEADS
][
    AV0_TOKENS
][
    AV0_PAD_TOKENS
];

extern const int8_t
av0_vt[
    AV0_HEADS
][
    AV0_HEAD_DIM
][
    AV0_PAD_TOKENS
];

extern const int32_t
av0_golden_acc[
    AV0_HEADS
][
    AV0_TOKENS
][
    AV0_HEAD_DIM
];

extern const int8_t
av0_golden_output[
    AV0_HEADS
][
    AV0_TOKENS
][
    AV0_HEAD_DIM
];

extern const int64_t av0_multiplier;

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "av0_data.h"\n\n'
        )

        f.write(
            f"const int64_t av0_multiplier = "
            f"{multiplier}LL;\n\n"
        )

        write_3d(
            f,
            "av0_prob",
            "int8_t",
            prob,
            aligned=True,
        )

        write_3d(
            f,
            "av0_vt",
            "int8_t",
            vt,
            aligned=True,
        )

        write_3d(
            f,
            "av0_golden_acc",
            "int32_t",
            acc,
        )

        write_3d(
            f,
            "av0_golden_output",
            "int8_t",
            out,
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
