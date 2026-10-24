from pathlib import Path
import numpy as np


INPUT = Path("results/softmax0_int8.npz")

HEADER = Path(
    "src/generated/softmax0_data.h"
)

SOURCE = Path(
    "src/generated/softmax0_data.c"
)

HEADS = 3
TOKENS = 197


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


def write_3d(
    f,
    name,
    ctype,
    a,
):
    a = np.asarray(a)

    d0, d1, d2 = a.shape

    f.write(
        f"const {ctype} "
        f"{name}[{d0}][{d1}][{d2}] = {{\n"
    )

    for h in range(d0):
        f.write("    {\n")

        for i in range(d1):
            f.write(
                "        {"
                + ", ".join(
                    str(int(v))
                    for v in a[h, i]
                )
                + "},\n"
            )

        f.write("    },\n")

    f.write("};\n\n")


def main():
    d = np.load(INPUT)

    lut = d["exp_lut"]
    prob = d["prob_q"]

    clip_steps = int(
        d["clip_steps"]
    )

    assert lut.shape == (256,)
    assert prob.shape == (
        HEADS,
        TOKENS,
        TOKENS,
    )

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEADER.write_text(
f"""#ifndef SOFTMAX0_DATA_H
#define SOFTMAX0_DATA_H

#include <stdint.h>

#define SOFTMAX0_HEADS {HEADS}
#define SOFTMAX0_TOKENS {TOKENS}

#define SOFTMAX0_LUT_SIZE 256
#define SOFTMAX0_CLIP_STEPS {clip_steps}

#define SOFTMAX0_PROB_QMAX 127

extern const uint16_t
softmax0_exp_lut[SOFTMAX0_LUT_SIZE];

extern const int8_t
softmax0_golden[
    SOFTMAX0_HEADS
][
    SOFTMAX0_TOKENS
][
    SOFTMAX0_TOKENS
];

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "softmax0_data.h"\n\n'
        )

        write_1d(
            f,
            "softmax0_exp_lut",
            "uint16_t",
            lut,
        )

        write_3d(
            f,
            "softmax0_golden",
            "int8_t",
            prob,
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)

    print(
        "Clip steps:",
        clip_steps,
    )


if __name__ == "__main__":
    main()
