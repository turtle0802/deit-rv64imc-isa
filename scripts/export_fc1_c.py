import os
import numpy as np


INPUT_PATH = "results/fc1_w8a8.npz"

OUTPUT_DIR = "src/generated"
HEADER_PATH = os.path.join(OUTPUT_DIR, "fc1_data.h")
SOURCE_PATH = os.path.join(OUTPUT_DIR, "fc1_data.c")


def format_array(data, values_per_line=16):
    flat = data.reshape(-1)

    lines = []

    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i + values_per_line]

        line = ", ".join(str(int(x)) for x in chunk)

        lines.append("    " + line)

    return ",\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = np.load(INPUT_PATH)

    input_q = data["input_q"].astype(np.int8)
    weight_q = data["weight_q"].astype(np.int8)
    bias_q = data["bias_q"].astype(np.int32)

    accumulator = data["accumulator"].astype(np.int32)

    multiplier_int = data["multiplier_int"].astype(np.int64)

    requant_shift = int(data["requant_shift"])

    output_q_fixed = data["output_q_fixed"].astype(np.int8)

    # Remove batch dimension
    input_q = input_q[0]
    accumulator = accumulator[0]
    output_q_fixed = output_q_fixed[0]

    print("Input       :", input_q.shape)
    print("Weight      :", weight_q.shape)
    print("Bias        :", bias_q.shape)
    print("Accumulator :", accumulator.shape)
    print("Multiplier  :", multiplier_int.shape)
    print("Output      :", output_q_fixed.shape)
    print("Shift       :", requant_shift)

    # ========================================================
    # Header
    # ========================================================

    header = f"""#ifndef FC1_DATA_H
#define FC1_DATA_H

#include <stdint.h>

#define FC1_TOKENS      197
#define FC1_IN_FEATURES 192
#define FC1_OUT_FEATURES 768

#define FC1_REQUANT_SHIFT {requant_shift}

extern const int8_t fc1_input[FC1_TOKENS][FC1_IN_FEATURES];

extern const int8_t fc1_weight[FC1_OUT_FEATURES][FC1_IN_FEATURES];

extern const int32_t fc1_bias[FC1_OUT_FEATURES];

extern const int64_t fc1_multiplier[FC1_OUT_FEATURES];

extern const int32_t fc1_golden_acc[FC1_TOKENS][FC1_OUT_FEATURES];

extern const int8_t fc1_golden_output[FC1_TOKENS][FC1_OUT_FEATURES];

#endif
"""

    with open(HEADER_PATH, "w", newline="\n") as f:
        f.write(header)

    # ========================================================
    # Source
    # ========================================================

    with open(SOURCE_PATH, "w", newline="\n") as f:
        f.write('#include "fc1_data.h"\n\n')

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        f.write(
            "const int8_t fc1_input"
            "[FC1_TOKENS][FC1_IN_FEATURES] = {\n"
        )

        for i in range(input_q.shape[0]):
            f.write("    {\n")
            f.write(format_array(input_q[i], 16))
            f.write("\n    }")

            if i != input_q.shape[0] - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # ----------------------------------------------------
        # Weight
        # ----------------------------------------------------

        f.write(
            "const int8_t fc1_weight"
            "[FC1_OUT_FEATURES][FC1_IN_FEATURES] = {\n"
        )

        for j in range(weight_q.shape[0]):
            f.write("    {\n")
            f.write(format_array(weight_q[j], 16))
            f.write("\n    }")

            if j != weight_q.shape[0] - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # ----------------------------------------------------
        # Bias
        # ----------------------------------------------------

        f.write(
            "const int32_t fc1_bias"
            "[FC1_OUT_FEATURES] = {\n"
        )

        f.write(format_array(bias_q, 8))

        f.write("\n};\n\n")

        # ----------------------------------------------------
        # Requant multiplier
        # ----------------------------------------------------

        f.write(
            "const int64_t fc1_multiplier"
            "[FC1_OUT_FEATURES] = {\n"
        )

        f.write(format_array(multiplier_int, 8))

        f.write("\n};\n\n")

        # ----------------------------------------------------
        # Golden accumulator
        # ----------------------------------------------------

        f.write(
            "const int32_t fc1_golden_acc"
            "[FC1_TOKENS][FC1_OUT_FEATURES] = {\n"
        )

        for i in range(accumulator.shape[0]):
            f.write("    {\n")
            f.write(format_array(accumulator[i], 8))
            f.write("\n    }")

            if i != accumulator.shape[0] - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # ----------------------------------------------------
        # Golden INT8 output
        # ----------------------------------------------------

        f.write(
            "const int8_t fc1_golden_output"
            "[FC1_TOKENS][FC1_OUT_FEATURES] = {\n"
        )

        for i in range(output_q_fixed.shape[0]):
            f.write("    {\n")
            f.write(format_array(output_q_fixed[i], 16))
            f.write("\n    }")

            if i != output_q_fixed.shape[0] - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n")

    print()
    print("Generated:")
    print(HEADER_PATH)
    print(SOURCE_PATH)


if __name__ == "__main__":
    main()