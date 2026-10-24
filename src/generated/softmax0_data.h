#ifndef SOFTMAX0_DATA_H
#define SOFTMAX0_DATA_H

#include <stdint.h>

#define SOFTMAX0_HEADS 3
#define SOFTMAX0_TOKENS 197

#define SOFTMAX0_LUT_SIZE 256
#define SOFTMAX0_CLIP_STEPS 16887

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
