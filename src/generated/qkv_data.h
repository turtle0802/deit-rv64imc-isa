#ifndef QKV_DATA_H
#define QKV_DATA_H

#include <stdint.h>

#define QKV_TOKENS       197
#define QKV_IN_FEATURES  192
#define QKV_OUT_FEATURES 576
#define QKV_HEADS        3
#define QKV_HEAD_DIM     64
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
