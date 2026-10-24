#ifndef FC2_DATA_H
#define FC2_DATA_H

#include <stdint.h>

#define FC2_TOKENS       197
#define FC2_IN_FEATURES  768
#define FC2_OUT_FEATURES 192

#define FC2_REQUANT_SHIFT 31

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
