#include <stdint.h>

#include "generated/proj0_data.h"


static int8_t requantize(
    int32_t acc,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)acc * multiplier;

    int64_t mag =
        product < 0
        ? -product
        : product;

    int64_t rounded =
        (
            mag + (1LL << 30)
        ) >> 31;

    int64_t result =
        product < 0
        ? -rounded
        : rounded;

    if (result > 127)
        result = 127;

    if (result < -127)
        result = -127;

    return (int8_t)result;
}


int main(void)
{
    for (
        int token = 0;
        token < PROJ0_TOKENS;
        token++
    ) {
        for (
            int out = 0;
            out < PROJ0_OUT_FEATURES;
            out++
        ) {
            int32_t acc =
                proj0_bias[out];

            for (
                int k = 0;
                k < PROJ0_IN_FEATURES;
                k++
            ) {
                acc +=
                    (int32_t)proj0_input[token][k]
                    *
                    (int32_t)proj0_weight[out][k];
            }

            if (
                acc
                !=
                proj0_golden_acc[token][out]
            ) {
                return 1;
            }

            int8_t result =
                requantize(
                    acc,
                    proj0_multiplier[out]
                );

            if (
                result
                !=
                proj0_golden_output[token][out]
            ) {
                return 2;
            }
        }
    }

    return 0;
}
