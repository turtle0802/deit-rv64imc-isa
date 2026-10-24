#include <stdint.h>

#include "generated/fc1_data.h"


static int8_t requantize(
    int32_t accumulator,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)accumulator * multiplier;

    int64_t abs_product =
        (product >= 0)
        ? product
        : -product;

    int64_t rounded_abs =
        (
            abs_product
            + (1LL << (FC1_REQUANT_SHIFT - 1))
        )
        >> FC1_REQUANT_SHIFT;

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


/*
 * B1:
 * Software-optimized RV64IMC
 *
 * - No custom instruction
 * - 8-way manually unrolled MAC loop
 * - Same numerical semantics as B0
 */
static int32_t fc1_accumulate_b1(
    int token,
    int channel
)
{
    const int8_t *x =
        &fc1_input[token][0];

    const int8_t *w =
        &fc1_weight[channel][0];

    int32_t acc =
        fc1_bias[channel];

    for (
        int k = 0;
        k < FC1_IN_FEATURES;
        k += 8
    ) {
        acc += (int32_t)x[0] * (int32_t)w[0];
        acc += (int32_t)x[1] * (int32_t)w[1];
        acc += (int32_t)x[2] * (int32_t)w[2];
        acc += (int32_t)x[3] * (int32_t)w[3];

        acc += (int32_t)x[4] * (int32_t)w[4];
        acc += (int32_t)x[5] * (int32_t)w[5];
        acc += (int32_t)x[6] * (int32_t)w[6];
        acc += (int32_t)x[7] * (int32_t)w[7];

        x += 8;
        w += 8;
    }

    return acc;
}


int main(void)
{
    /*
     * 0 = PASS
     * 1 = accumulator mismatch
     * 2 = requantized output mismatch
     */

    for (
        int token = 0;
        token < FC1_TOKENS;
        token++
    ) {
        for (
            int channel = 0;
            channel < FC1_OUT_FEATURES;
            channel++
        ) {
            int32_t acc =
                fc1_accumulate_b1(
                    token,
                    channel
                );

            if (
                acc
                !=
                fc1_golden_acc[token][channel]
            ) {
                return 1;
            }

            int8_t output =
                requantize(
                    acc,
                    fc1_multiplier[channel]
                );

            if (
                output
                !=
                fc1_golden_output[token][channel]
            ) {
                return 2;
            }
        }
    }

    return 0;
}
