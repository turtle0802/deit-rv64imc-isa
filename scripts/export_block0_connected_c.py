from pathlib import Path
import numpy as np

BLOCK = Path("results/block0_connected_int8.npz")
QKV = Path("results/qkv_w8a8.npz")
SOFTMAX = Path("results/softmax0_int8.npz")
AV = Path("results/av0_w8a8.npz")
PROJ = Path("results/proj0_w8a8.npz")

HEADER = Path("src/generated/block0_data.h")
SOURCE = Path("src/generated/block0_data.c")


def write_array(f, name, ctype, x, aligned=False, per_line=16):
    a = np.asarray(x).reshape(-1)

    attr = " __attribute__((aligned(8)))" if aligned else ""

    f.write(
        f"const {ctype} {name}[{a.size}]{attr} = {{\n"
    )

    for i in range(0, a.size, per_line):
        vals = ", ".join(
            str(int(v))
            for v in a[i:i + per_line]
        )
        f.write(f"    {vals},\n")

    f.write("};\n\n")


def main():
    b = np.load(BLOCK)
    q = np.load(QKV)
    s = np.load(SOFTMAX)
    a = np.load(AV)
    p = np.load(PROJ)

    clip_steps = int(s["clip_steps"])

    HEADER.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    HEADER.write_text(
f"""#ifndef BLOCK0_DATA_H
#define BLOCK0_DATA_H

#include <stdint.h>

#define B0_TOKENS      197
#define B0_EMBED       192
#define B0_HIDDEN      768
#define B0_QKV_OUT     576

#define B0_HEADS       3
#define B0_HEAD_DIM    64

#define B0_PAD_TOKENS  200

#define B0_REQUANT_SHIFT 31

#define B0_LN_Z_Q       15
#define B0_LN_AFFINE_Q  20

#define B0_SOFTMAX_CLIP_STEPS {clip_steps}
#define B0_PROB_QMAX 127

extern const int8_t b0_x0_q[B0_TOKENS * B0_EMBED];

extern const int64_t b0_ln1_a[B0_EMBED];
extern const int64_t b0_ln1_b[B0_EMBED];
extern const int64_t b0_ln1_eps_term;

extern const int8_t b0_qkv_weight[B0_QKV_OUT * B0_EMBED];
extern const int32_t b0_qkv_bias[B0_QKV_OUT];
extern const int64_t b0_qkv_multiplier[B0_QKV_OUT];

extern const uint16_t b0_exp_lut[256];

extern const int64_t b0_av_multiplier;

extern const int8_t b0_proj_weight[B0_EMBED * B0_EMBED];
extern const int32_t b0_proj_bias[B0_EMBED];
extern const int64_t b0_proj_multiplier[B0_EMBED];

extern const int64_t b0_x0_to_r1;
extern const int64_t b0_proj_to_r1;

extern const int64_t b0_ln2_a[B0_EMBED];
extern const int64_t b0_ln2_b[B0_EMBED];
extern const int64_t b0_ln2_eps_term;

extern const int8_t b0_fc1_weight[B0_HIDDEN * B0_EMBED];
extern const int32_t b0_fc1_bias[B0_HIDDEN];
extern const int64_t b0_fc1_multiplier[B0_HIDDEN];

extern const int8_t b0_gelu_lut[256];

extern const int8_t b0_fc2_weight[B0_EMBED * B0_HIDDEN];
extern const int32_t b0_fc2_bias[B0_EMBED];
extern const int64_t b0_fc2_multiplier[B0_EMBED];

extern const int64_t b0_r1_to_final;
extern const int64_t b0_mlp_to_final;


/* Stage goldens */
extern const int8_t b0_gold_ln1[B0_TOKENS * B0_EMBED];
extern const int8_t b0_gold_qkv[B0_TOKENS * B0_QKV_OUT];

extern const int8_t b0_gold_prob[
    B0_HEADS * B0_TOKENS * B0_TOKENS
];

extern const int8_t b0_gold_av[
    B0_HEADS * B0_TOKENS * B0_HEAD_DIM
];

extern const int8_t b0_gold_proj[B0_TOKENS * B0_EMBED];
extern const int8_t b0_gold_r1[B0_TOKENS * B0_EMBED];
extern const int8_t b0_gold_ln2[B0_TOKENS * B0_EMBED];

extern const int8_t b0_gold_fc1[B0_TOKENS * B0_HIDDEN];
extern const int8_t b0_gold_fc2_input[B0_TOKENS * B0_HIDDEN];

extern const int8_t b0_gold_mlp[B0_TOKENS * B0_EMBED];
extern const int8_t b0_gold_final[B0_TOKENS * B0_EMBED];

#endif
"""
    )

    with SOURCE.open("w", encoding="utf-8") as f:
        f.write('#include "block0_data.h"\n\n')

        write_array(
            f, "b0_x0_q", "int8_t",
            b["x0_q"], aligned=True
        )

        write_array(f, "b0_ln1_a", "int64_t", b["ln1_affine_a"])
        write_array(f, "b0_ln1_b", "int64_t", b["ln1_affine_b"])

        f.write(
            f"const int64_t b0_ln1_eps_term = "
            f"{int(b['ln1_eps_term'])}LL;\n\n"
        )

        write_array(
            f, "b0_qkv_weight", "int8_t",
            q["weight_q"], aligned=True
        )

        write_array(
            f, "b0_qkv_bias", "int32_t",
            q["bias_q"]
        )

        write_array(
            f, "b0_qkv_multiplier", "int64_t",
            q["multiplier_int"]
        )

        write_array(
            f, "b0_exp_lut", "uint16_t",
            s["exp_lut"]
        )

        f.write(
            f"const int64_t b0_av_multiplier = "
            f"{int(a['multiplier_int'])}LL;\n\n"
        )

        write_array(
            f, "b0_proj_weight", "int8_t",
            p["weight_q"], aligned=True
        )

        write_array(
            f, "b0_proj_bias", "int32_t",
            p["bias_q"]
        )

        write_array(
            f, "b0_proj_multiplier", "int64_t",
            p["multiplier_int"]
        )

        f.write(
            f"const int64_t b0_x0_to_r1 = "
            f"{int(b['x0_to_r1'])}LL;\n"
        )

        f.write(
            f"const int64_t b0_proj_to_r1 = "
            f"{int(b['proj_to_r1'])}LL;\n\n"
        )

        write_array(f, "b0_ln2_a", "int64_t", b["ln2_affine_a"])
        write_array(f, "b0_ln2_b", "int64_t", b["ln2_affine_b"])

        f.write(
            f"const int64_t b0_ln2_eps_term = "
            f"{int(b['ln2_eps_term'])}LL;\n\n"
        )

        write_array(
            f, "b0_fc1_weight", "int8_t",
            b["fc1_weight_q"], aligned=True
        )

        write_array(
            f, "b0_fc1_bias", "int32_t",
            b["fc1_bias_q"]
        )

        write_array(
            f, "b0_fc1_multiplier", "int64_t",
            b["fc1_multiplier"]
        )

        write_array(
            f, "b0_gelu_lut", "int8_t",
            b["gelu_lut"]
        )

        write_array(
            f, "b0_fc2_weight", "int8_t",
            b["fc2_weight_q"], aligned=True
        )

        write_array(
            f, "b0_fc2_bias", "int32_t",
            b["fc2_bias_q"]
        )

        write_array(
            f, "b0_fc2_multiplier", "int64_t",
            b["fc2_multiplier"]
        )

        f.write(
            f"const int64_t b0_r1_to_final = "
            f"{int(b['r1_to_final'])}LL;\n"
        )

        f.write(
            f"const int64_t b0_mlp_to_final = "
            f"{int(b['mlp_to_final'])}LL;\n\n"
        )

        # Goldens
        write_array(f, "b0_gold_ln1", "int8_t", b["ln1_q"])
        write_array(f, "b0_gold_qkv", "int8_t", b["qkv_q"])
        write_array(f, "b0_gold_prob", "int8_t", b["prob_q"])
        write_array(f, "b0_gold_av", "int8_t", b["av_q"])
        write_array(f, "b0_gold_proj", "int8_t", b["proj_q"])
        write_array(f, "b0_gold_r1", "int8_t", b["residual1_q"])
        write_array(f, "b0_gold_ln2", "int8_t", b["ln2_q"])
        write_array(f, "b0_gold_fc1", "int8_t", b["fc1_q"])
        write_array(f, "b0_gold_fc2_input", "int8_t", b["fc2_input_q"])
        write_array(f, "b0_gold_mlp", "int8_t", b["mlp_q"])
        write_array(f, "b0_gold_final", "int8_t", b["final_q"])

    print("Generated:")
    print(HEADER)
    print(SOURCE)


if __name__ == "__main__":
    main()
