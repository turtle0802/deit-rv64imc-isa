#ifndef BLOCK0_DATA_H
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

#define B0_SOFTMAX_CLIP_STEPS 16887
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
