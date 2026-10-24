#include <stdint.h>

#include "generated/qkv_data.h"


static int8_t requantize(
    int32_t acc,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)acc
        * multiplier;

    int64_t mag =
        product < 0
        ? -product
        : product;

    int64_t rounded =
        (
            mag
            + (1LL << 30)
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
        token < QKV_TOKENS;
        token++
    ) {
        for (
            int out = 0;
            out < QKV_OUT_FEATURES;
            out++
        ) {
            int32_t acc =
                qkv_bias[out];

            for (
                int k = 0;
                k < QKV_IN_FEATURES;
                k++
            ) {
                acc +=
                    (int32_t)qkv_input[token][k]
                    *
                    (int32_t)qkv_weight[out][k];
            }

            if (
                acc
                !=
                qkv_golden_acc[token][out]
            ) {
                return 1;
            }

            int8_t result =
                requantize(
                    acc,
                    qkv_multiplier[out]
                );

            if (
                result
                !=
                qkv_golden_output[token][out]
            ) {
                return 2;
            }
        }
    }

    return 0;
}
