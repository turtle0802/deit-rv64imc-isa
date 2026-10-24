#include <stdint.h>

#include "generated/fc1_data.h"

volatile uint64_t benchmark_sink;


static inline uint64_t load_u64(
    const void *ptr
)
{
    uint64_t value;

    __asm__ volatile (
        "ld %0, 0(%1)"
        : "=r"(value)
        : "r"(ptr)
        : "memory"
    );

    return value;
}


static inline int64_t xqmac8(
    int64_t acc,
    uint64_t a,
    uint64_t b
)
{
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "+r"(acc)
        : "r"(a),
          "r"(b)
    );

    return acc;
}


static inline int32_t xrequant(
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

    return (int32_t)result;
}


static int32_t fc1_accumulate_xqmac8(
    int token,
    int channel
)
{
    int64_t acc =
        (int64_t)fc1_bias[channel];

    for (
        int k = 0;
        k < FC1_IN_FEATURES;
        k += 8
    ) {
        uint64_t input_pack =
            load_u64(
                &fc1_input[token][k]
            );

        uint64_t weight_pack =
            load_u64(
                &fc1_weight[channel][k]
            );

        acc =
            xqmac8(
                acc,
                input_pack,
                weight_pack
            );
    }

    return (int32_t)acc;
}


int main(void)
{
    uint64_t checksum = 0;

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
                fc1_accumulate_xqmac8(
                    token,
                    channel
                );

            int32_t output =
                xrequant(
                    acc,
                    fc1_multiplier[channel]
                );

            checksum +=
                (uint8_t)(int8_t)output;
        }
    }

    benchmark_sink = checksum;

    return 0;
}
