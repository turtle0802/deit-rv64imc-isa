#include <stdint.h>

#include "generated/fc1_data.h"
#include "generated/fc2_data.h"
#include "generated/mlp0_data.h"


static int8_t requantize(
    int32_t acc,
    int64_t multiplier,
    int shift
)
{
    int64_t product =
        (int64_t)acc * multiplier;

    int64_t magnitude =
        (product < 0)
        ? -product
        : product;

    int64_t rounded =
        (
            magnitude
            + (1LL << (shift - 1))
        ) >> shift;

    int64_t result =
        (product < 0)
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
    static int8_t hidden[FC1_OUT_FEATURES];

    for (
        int token = 0;
        token < FC1_TOKENS;
        token++
    ) {
        /*
         * FC1
         */
        for (
            int out = 0;
            out < FC1_OUT_FEATURES;
            out++
        ) {
            int32_t acc =
                fc1_bias[out];

            for (
                int k = 0;
                k < FC1_IN_FEATURES;
                k++
            ) {
                acc +=
                    (int32_t)fc1_input[token][k]
                    *
                    (int32_t)fc1_weight[out][k];
            }

            int8_t fc1_q =
                requantize(
                    acc,
                    fc1_multiplier[out],
                    FC1_REQUANT_SHIFT
                );

            /*
             * GELU LUT
             *
             * uint8_t cast gives exactly the same
             * bit-pattern indexing as Python q & 0xff.
             */
            hidden[out] =
                mlp0_gelu_lut[
                    (uint8_t)fc1_q
                ];

            if (
                hidden[out]
                !=
                mlp0_fc2_input_golden[token][out]
            ) {
                return 1;
            }
        }

        /*
         * FC2
         */
        for (
            int out = 0;
            out < FC2_OUT_FEATURES;
            out++
        ) {
            int32_t acc =
                fc2_bias[out];

            for (
                int k = 0;
                k < FC2_IN_FEATURES;
                k++
            ) {
                acc +=
                    (int32_t)hidden[k]
                    *
                    (int32_t)fc2_weight[out][k];
            }

            if (
                acc
                !=
                mlp0_fc2_acc_golden[token][out]
            ) {
                return 2;
            }

            int8_t result =
                requantize(
                    acc,
                    fc2_multiplier[out],
                    FC2_REQUANT_SHIFT
                );

            if (
                result
                !=
                mlp0_output_golden[token][out]
            ) {
                return 3;
            }
        }
    }

    return 0;
}
