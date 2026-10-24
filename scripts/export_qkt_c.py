from pathlib import Path
import numpy as np


INPUT = Path("results/qkt_w8a8.npz")
HEADER = Path("src/generated/qkt_data.h")
SOURCE = Path("src/generated/qkt_data.c")

HEADS = 3
TOKENS = 197
DIM = 64


def write_3d(
    f,
    name,
    ctype,
    a,
    aligned=False,
):
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

    q = d["q_q"]
    k = d["k_q"]
    acc = d["accumulator"]

    assert q.shape == (HEADS, TOKENS, DIM)
    assert k.shape == (HEADS, TOKENS, DIM)
    assert acc.shape == (HEADS, TOKENS, TOKENS)

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEADER.write_text(
f"""#ifndef QKT_DATA_H
#define QKT_DATA_H

#include <stdint.h>

#define QKT_HEADS    {HEADS}
#define QKT_TOKENS   {TOKENS}
#define QKT_HEAD_DIM {DIM}

extern const int8_t
qkt_q[QKT_HEADS][QKT_TOKENS][QKT_HEAD_DIM];

extern const int8_t
qkt_k[QKT_HEADS][QKT_TOKENS][QKT_HEAD_DIM];

extern const int32_t
qkt_golden_acc[QKT_HEADS][QKT_TOKENS][QKT_TOKENS];

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "qkt_data.h"\n\n'
        )

        write_3d(
            f,
            "qkt_q",
            "int8_t",
            q,
            aligned=True,
        )

        write_3d(
            f,
            "qkt_k",
            "int8_t",
            k,
            aligned=True,
        )

        write_3d(
            f,
            "qkt_golden_acc",
            "int32_t",
            acc,
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
