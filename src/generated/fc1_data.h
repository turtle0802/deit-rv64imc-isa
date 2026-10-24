#ifndef FC1_DATA_H
#define FC1_DATA_H

#include <stdint.h>

#define FC1_TOKENS      197
#define FC1_IN_FEATURES 192
#define FC1_OUT_FEATURES 768

#define FC1_REQUANT_SHIFT 31

extern const int8_t fc1_input[FC1_TOKENS][FC1_IN_FEATURES];

extern const int8_t fc1_weight[FC1_OUT_FEATURES][FC1_IN_FEATURES];

extern const int32_t fc1_bias[FC1_OUT_FEATURES];

extern const int64_t fc1_multiplier[FC1_OUT_FEATURES];

extern const int32_t fc1_golden_acc[FC1_TOKENS][FC1_OUT_FEATURES];

extern const int8_t fc1_golden_output[FC1_TOKENS][FC1_OUT_FEATURES];

#endif
