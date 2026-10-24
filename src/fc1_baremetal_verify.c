#include <stdint.h>

#include "generated/fc1_data.h"


/* ============================================================
 * Fixed-point requantization
 *
 * Semantics must exactly match Python golden reference:
 *
 * product =
 *     accumulator * multiplier
 *
 * output =
 *     round_to_nearest_away_from_zero(
 *         product / 2^31
 *     )
 *
 * then saturate to [-127, 127].
 * ============================================================
 */

static int8_t requantize(
    int32_t accumulator,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)accumulator
        * multiplier;

    int64_t abs_product;

    if (product >= 0) {
        abs_product = product;
    }
    else {
        abs_product = -product;
    }

    int64_t rounded_abs =
        (
            abs_product
            + (1LL << (FC1_REQUANT_SHIFT - 1))
        )
        >> FC1_REQUANT_SHIFT;

    int64_t result;

    if (product >= 0) {
        result = rounded_abs;
    }
    else {
        result = -rounded_abs;
    }

    if (result > 127) {
        result = 127;
    }

    if (result < -127) {
        result = -127;
    }

    return (int8_t)result;
}


/* ============================================================
 * Scalar RV64IMC FC1
 *
 * FC1:
 *
 * input  : [197][192]
 * weight : [768][192]
 * output : [197][768]
 * ============================================================
 */

static int32_t fc1_accumulate(
    int token,
    int channel
)
{
    int32_t acc =
        fc1_bias[channel];

    for (
        int k = 0;
        k < FC1_IN_FEATURES;
        k++
    ) {
        acc +=
            (int32_t)fc1_input[token][k]
            *
            (int32_t)fc1_weight[channel][k];
    }

    return acc;
}


int main(void)
{
    /*
     * Return code:
     *
     * 0 = PASS
     * 1 = accumulator mismatch
     * 2 = requantization mismatch
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
                fc1_accumulate(
                    token,
                    channel
                );

            /*
             * Verify INT32 accumulator
             * against Python golden.
             */

            if (
                acc
                !=
                fc1_golden_acc[token][channel]
            ) {
                return 1;
            }

            /*
             * Verify fixed-point
             * requantization.
             */

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
