#include <stdint.h>

#include "generated/fc1_data.h"
#include "generated/fc2_data.h"
#include "generated/mlp0_data.h"


static inline uint64_t load64(
    const int8_t *p
)
{
    uint64_t v;

    __asm__ volatile (
        "ld %0, 0(%1)"
        : "=r"(v)
        : "r"(p)
    );

    return v;
}


static inline int32_t xqmac8(
    int32_t acc,
    uint64_t a,
    uint64_t b
)
{
    int64_t rd = acc;

    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "+r"(rd)
        : "r"(a), "r"(b)
    );

    return (int32_t)rd;
}


static inline int8_t xrequant(
    int32_t acc,
    int64_t multiplier
)
{
    int64_t result;

    __asm__ volatile (
        ".insn r 0x0b, 1, 0, %0, %1, %2"
        : "=r"(result)
        : "r"((int64_t)acc),
          "r"(multiplier)
    );

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
         * FC1:
         * 192 input channels
         * -> 24 x Xqmac8 per output
         */
        for (
            int out = 0;
            out < FC1_OUT_FEATURES;
            out++
        ) {
            const int8_t *x =
                &fc1_input[token][0];

            const int8_t *w =
                &fc1_weight[out][0];

            int32_t acc =
                fc1_bias[out];

            for (
                int k = 0;
                k < FC1_IN_FEATURES;
                k += 8
            ) {
                uint64_t px =
                    load64(x + k);

                uint64_t pw =
                    load64(w + k);

                acc = xqmac8(
                    acc,
                    px,
                    pw
                );
            }

            int8_t fc1_q =
                xrequant(
                    acc,
                    fc1_multiplier[out]
                );

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
         * FC2:
         * 768 input channels
         * -> 96 x Xqmac8 per output
         */
        for (
            int out = 0;
            out < FC2_OUT_FEATURES;
            out++
        ) {
            const int8_t *x =
                &hidden[0];

            const int8_t *w =
                &fc2_weight[out][0];

            int32_t acc =
                fc2_bias[out];

            for (
                int k = 0;
                k < FC2_IN_FEATURES;
                k += 8
            ) {
                uint64_t px =
                    load64(x + k);

                uint64_t pw =
                    load64(w + k);

                acc = xqmac8(
                    acc,
                    px,
                    pw
                );
            }

            if (
                acc
                !=
                mlp0_fc2_acc_golden[token][out]
            ) {
                return 2;
            }

            int8_t result =
                xrequant(
                    acc,
                    fc2_multiplier[out]
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
