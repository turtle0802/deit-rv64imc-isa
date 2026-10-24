#include <stdint.h>

#include "generated/qkv_data.h"


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
            const int8_t *x =
                &qkv_input[token][0];

            const int8_t *w =
                &qkv_weight[out][0];

            int32_t acc =
                qkv_bias[out];

            for (
                int k = 0;
                k < QKV_IN_FEATURES;
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
                qkv_golden_acc[token][out]
            ) {
                return 1;
            }

            int8_t result =
                xrequant(
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
