#include <stdint.h>

#include "generated/av0_data.h"


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
        int h = 0;
        h < AV0_HEADS;
        h++
    ) {
        for (
            int i = 0;
            i < AV0_TOKENS;
            i++
        ) {
            for (
                int d = 0;
                d < AV0_HEAD_DIM;
                d++
            ) {
                int32_t acc = 0;

                for (
                    int j = 0;
                    j < AV0_PAD_TOKENS;
                    j++
                ) {
                    acc +=
                        (int32_t)av0_prob[h][i][j]
                        *
                        (int32_t)av0_vt[h][d][j];
                }

                if (
                    acc
                    !=
                    av0_golden_acc[h][i][d]
                ) {
                    return 1;
                }

                int8_t result =
                    requantize(
                        acc,
                        av0_multiplier
                    );

                if (
                    result
                    !=
                    av0_golden_output[h][i][d]
                ) {
                    return 2;
                }
            }
        }
    }

    return 0;
}
