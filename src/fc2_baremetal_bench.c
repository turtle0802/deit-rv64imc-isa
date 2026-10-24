#include <stdint.h>

#include "generated/fc2_data.h"

volatile uint64_t benchmark_sink;


static int8_t requantize(
    int32_t accumulator,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)accumulator
        * multiplier;

    int64_t abs_product =
        (product >= 0)
        ? product
        : -product;

    int64_t rounded_abs =
        (
            abs_product
            + (1LL << (FC2_REQUANT_SHIFT - 1))
        )
        >> FC2_REQUANT_SHIFT;

    int64_t result =
        (product >= 0)
        ? rounded_abs
        : -rounded_abs;

    if (result > 127) {
        result = 127;
    }

    if (result < -127) {
        result = -127;
    }

    return (int8_t)result;
}


int main(void)
{
    uint64_t checksum = 0;

    for (
        int token = 0;
        token < FC2_TOKENS;
        token++
    ) {
        for (
            int channel = 0;
            channel < FC2_OUT_FEATURES;
            channel++
        ) {
            int32_t acc =
                fc2_bias[channel];

            for (
                int k = 0;
                k < FC2_IN_FEATURES;
                k++
            ) {
                acc +=
                    (int32_t)fc2_input[token][k]
                    *
                    (int32_t)fc2_weight[channel][k];
            }

            int8_t output =
                requantize(
                    acc,
                    fc2_multiplier[channel]
                );

            checksum +=
                (uint8_t)output;
        }
    }

    benchmark_sink = checksum;

    return 0;
}
