from pathlib import Path
import numpy as np


INPUT = Path("results/fc2_w8a8.npz")

HEADER = Path("src/generated/fc2_data.h")
SOURCE = Path("src/generated/fc2_data.c")


TOKENS = 197
IN_FEATURES = 768
OUT_FEATURES = 192


def write_1d(
    f,
    name,
    ctype,
    array,
    per_line=16,
):
    a = np.asarray(array).reshape(-1)

    f.write(
        f"const {ctype} {name}[{len(a)}] = {{\n"
    )

    for i in range(0, len(a), per_line):
        values = a[i:i + per_line]

        text = ", ".join(
            str(int(v))
            for v in values
        )

        f.write(f"    {text},\n")

    f.write("};\n\n")


def write_2d(
    f,
    name,
    ctype,
    array,
):
    a = np.asarray(array)

    rows, cols = a.shape

    f.write(
        f"const {ctype} "
        f"{name}[{rows}][{cols}] = {{\n"
    )

    for row in a:
        f.write("    {")

        f.write(
            ", ".join(
                str(int(v))
                for v in row
            )
        )

        f.write("},\n")

    f.write("};\n\n")


def main():
    data = np.load(INPUT)

    input_q = data["input_q"]
    weight_q = data["weight_q"]
    bias_q = data["bias_q"]

    accumulator = data["accumulator"]

    multiplier = data["multiplier_int"]
    shift = int(data["shift"])

    output_q = data["output_q"]

    assert input_q.shape == (
        TOKENS,
        IN_FEATURES,
    )

    assert weight_q.shape == (
        OUT_FEATURES,
        IN_FEATURES,
    )

    assert bias_q.shape == (
        OUT_FEATURES,
    )

    assert accumulator.shape == (
        TOKENS,
        OUT_FEATURES,
    )

    assert multiplier.shape == (
        OUT_FEATURES,
    )

    assert output_q.shape == (
        TOKENS,
        OUT_FEATURES,
    )

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    HEADER.write_text(
f"""#ifndef FC2_DATA_H
#define FC2_DATA_H

#include <stdint.h>

#define FC2_TOKENS       {TOKENS}
#define FC2_IN_FEATURES  {IN_FEATURES}
#define FC2_OUT_FEATURES {OUT_FEATURES}

#define FC2_REQUANT_SHIFT {shift}

extern const int8_t
fc2_input[FC2_TOKENS][FC2_IN_FEATURES];

extern const int8_t
fc2_weight[FC2_OUT_FEATURES][FC2_IN_FEATURES];

extern const int32_t
fc2_bias[FC2_OUT_FEATURES];

extern const int64_t
fc2_multiplier[FC2_OUT_FEATURES];

extern const int32_t
fc2_golden_acc[FC2_TOKENS][FC2_OUT_FEATURES];

extern const int8_t
fc2_golden_output[FC2_TOKENS][FC2_OUT_FEATURES];

#endif
"""
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------
    with SOURCE.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            '#include "fc2_data.h"\n\n'
        )

        write_2d(
            f,
            "fc2_input",
            "int8_t",
            input_q,
        )

        write_2d(
            f,
            "fc2_weight",
            "int8_t",
            weight_q,
        )

        write_1d(
            f,
            "fc2_bias",
            "int32_t",
            bias_q,
        )

        write_1d(
            f,
            "fc2_multiplier",
            "int64_t",
            multiplier,
            per_line=8,
        )

        write_2d(
            f,
            "fc2_golden_acc",
            "int32_t",
            accumulator,
        )

        write_2d(
            f,
            "fc2_golden_output",
            "int8_t",
            output_q,
        )

    print("Generated:")
    print(" ", HEADER)
    print(" ", SOURCE)

    print()
    print("FC2 data")
    print("------------------------------")
    print("Input       :", input_q.shape)
    print("Weight      :", weight_q.shape)
    print("Bias        :", bias_q.shape)
    print("Multiplier  :", multiplier.shape)
    print("Accumulator :", accumulator.shape)
    print("Output      :", output_q.shape)
    print("Shift       :", shift)


if __name__ == "__main__":
    main()
