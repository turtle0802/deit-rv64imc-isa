#ifndef AV0_DATA_H
#define AV0_DATA_H

#include <stdint.h>

#define AV0_HEADS 3
#define AV0_TOKENS 197
#define AV0_PAD_TOKENS 200
#define AV0_HEAD_DIM 64

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
