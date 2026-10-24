#include <stdint.h>

#include "generated/qkt_data.h"
#include "generated/softmax0_data.h"

volatile uint64_t benchmark_sink;


static uint32_t get_exp_q15(
    int32_t score,
    int32_t row_max
)
{
    uint32_t distance =
        (uint32_t)(
            row_max - score
        );

    uint64_t numerator =
        (uint64_t)distance
        * 255u;

    numerator +=
        SOFTMAX0_CLIP_STEPS / 2;

    uint32_t index =
        (uint32_t)(
            numerator
            / SOFTMAX0_CLIP_STEPS
        );

    if (index > 255)
        index = 255;

    return
        softmax0_exp_lut[index];
}


int main(void)
{
    uint64_t checksum = 0;

    for (
        int h = 0;
        h < SOFTMAX0_HEADS;
        h++
    ) {
        for (
            int i = 0;
            i < SOFTMAX0_TOKENS;
            i++
        ) {
            int32_t row_max =
                qkt_golden_acc[h][i][0];

            for (
                int j = 1;
                j < SOFTMAX0_TOKENS;
                j++
            ) {
                int32_t x =
                    qkt_golden_acc[h][i][j];

                if (x > row_max)
                    row_max = x;
            }

            uint64_t exp_sum = 0;

            for (
                int j = 0;
                j < SOFTMAX0_TOKENS;
                j++
            ) {
                exp_sum +=
                    get_exp_q15(
                        qkt_golden_acc[h][i][j],
                        row_max
                    );
            }

            for (
                int j = 0;
                j < SOFTMAX0_TOKENS;
                j++
            ) {
                uint64_t exp_value =
                    get_exp_q15(
                        qkt_golden_acc[h][i][j],
                        row_max
                    );

                uint64_t numerator =
                    exp_value
                    * SOFTMAX0_PROB_QMAX;

                uint32_t probability =
                    (uint32_t)(
                        (
                            numerator
                            + exp_sum / 2
                        )
                        / exp_sum
                    );

                if (
                    probability
                    > SOFTMAX0_PROB_QMAX
                ) {
                    probability =
                        SOFTMAX0_PROB_QMAX;
                }

                checksum += probability;
            }
        }
    }

    benchmark_sink = checksum;

    return 0;
}
