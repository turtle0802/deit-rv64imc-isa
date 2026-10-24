#include <stdint.h>

#include "generated/qkv_data.h"

volatile uint64_t benchmark_sink;


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
    uint64_t checksum = 0;

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
                acc = xqmac8(
                    acc,
                    load64(x + k),
                    load64(w + k)
                );
            }

            checksum +=
                (uint8_t)xrequant(
                    acc,
                    qkv_multiplier[out]
                );
        }
    }

    benchmark_sink = checksum;

    return 0;
}
