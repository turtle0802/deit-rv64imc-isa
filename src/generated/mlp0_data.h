#ifndef MLP0_DATA_H
#define MLP0_DATA_H

#include <stdint.h>

#define MLP0_TOKENS 197
#define MLP0_HIDDEN 768
#define MLP0_OUTPUT 192

extern const int8_t
mlp0_gelu_lut[256];

extern const int8_t
mlp0_fc2_input_golden[MLP0_TOKENS][MLP0_HIDDEN];

extern const int32_t
mlp0_fc2_acc_golden[MLP0_TOKENS][MLP0_OUTPUT];

extern const int8_t
mlp0_output_golden[MLP0_TOKENS][MLP0_OUTPUT];

#endif
