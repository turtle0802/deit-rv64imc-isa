from pathlib import Path
import numpy as np

INPUT = Path("results/mlp0_w8a8.npz")
HEADER = Path("src/generated/mlp0_data.h")
SOURCE = Path("src/generated/mlp0_data.c")

TOKENS = 197
HIDDEN = 768
OUTPUT = 192


def write_1d(f, name, ctype, a, per_line=16):
    a = np.asarray(a).reshape(-1)

    f.write(
        f"const {ctype} {name}[{len(a)}] = {{\n"
    )

    for i in range(0, len(a), per_line):
        row = a[i:i+per_line]

        f.write(
            "    "
            + ", ".join(str(int(x)) for x in row)
            + ",\n"
        )

    f.write("};\n\n")


def write_2d(f, name, ctype, a):
    a = np.asarray(a)

    rows, cols = a.shape

    f.write(
        f"const {ctype} "
        f"{name}[{rows}][{cols}] = {{\n"
    )

    for row in a:
        f.write(
            "    {"
            + ", ".join(str(int(x)) for x in row)
            + "},\n"
        )

    f.write("};\n\n")


def main():
    d = np.load(INPUT)

    lut = d["gelu_lut"]
    fc2_input = d["fc2_input_q"]
    fc2_acc = d["fc2_accumulator"]
    fc2_output = d["fc2_output_q"]

    assert lut.shape == (256,)
    assert fc2_input.shape == (TOKENS, HIDDEN)
    assert fc2_acc.shape == (TOKENS, OUTPUT)
    assert fc2_output.shape == (TOKENS, OUTPUT)

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    HEADER.write_text(
f"""#ifndef MLP0_DATA_H
#define MLP0_DATA_H

#include <stdint.h>

#define MLP0_TOKENS 197
#define MLP0_HIDDEN 768
#define MLP0_OUTPUT 192

extern const int8_t
mlp0_gelu_lut[256];

extern const int8_t
mlp0_fc2_input_golden[MLP0_TOKENS][MLP0_HIDDEN];

extern const int32_t
mlp0_fc2_acc_golden[MLP0_TOKENS][MLP0_OUTPUT];

extern const int8_t
mlp0_output_golden[MLP0_TOKENS][MLP0_OUTPUT];

#endif
"""
    )

    with SOURCE.open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            '#include "mlp0_data.h"\n\n'
        )

        write_1d(
            f,
            "mlp0_gelu_lut",
            "int8_t",
            lut
        )

        write_2d(
            f,
            "mlp0_fc2_input_golden",
            "int8_t",
            fc2_input
        )

        write_2d(
            f,
            "mlp0_fc2_acc_golden",
            "int32_t",
            fc2_acc
        )

        write_2d(
            f,
            "mlp0_output_golden",
            "int8_t",
            fc2_output
        )

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
