#ifndef PROJ0_DATA_H
#define PROJ0_DATA_H

#include <stdint.h>

#define PROJ0_TOKENS 197
#define PROJ0_IN_FEATURES 192
#define PROJ0_OUT_FEATURES 192
#define PROJ0_REQUANT_SHIFT 31

extern const int8_t
proj0_input[PROJ0_TOKENS][PROJ0_IN_FEATURES];

extern const int8_t
proj0_weight[PROJ0_OUT_FEATURES][PROJ0_IN_FEATURES];

extern const int32_t
proj0_bias[PROJ0_OUT_FEATURES];

extern const int64_t
proj0_multiplier[PROJ0_OUT_FEATURES];

extern const int32_t
proj0_golden_acc[PROJ0_TOKENS][PROJ0_OUT_FEATURES];

extern const int8_t
proj0_golden_output[PROJ0_TOKENS][PROJ0_OUT_FEATURES];

#endif
