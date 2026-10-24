#include <stdint.h>

#include "generated/fc2_data.h"


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


/*
 * XQMAC8
 *
 * rd = rd +
 *      sum_{i=0}^{7}
 *      int8(rs1[i]) * int8(rs2[i])
 */
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


/*
 * XREQUANT
 *
 * rs1[31:0] = INT32 accumulator
 * rs2[31:0] = Q31 multiplier
 *
 * rd =
 *   sat[-127,127](
 *     round_away_from_zero(
 *       acc * multiplier / 2^31
 *     )
 *   )
 */
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


static int32_t fc2_accumulate_xqmac8(
    int token,
    int channel
)
{
    int64_t acc =
        (int64_t)fc2_bias[channel];

    for (
        int k = 0;
        k < FC2_IN_FEATURES;
        k += 8
    ) {
        uint64_t input_pack =
            load_u64(
                &fc2_input[token][k]
            );

        uint64_t weight_pack =
            load_u64(
                &fc2_weight[channel][k]
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
    /*
     * 0 = PASS
     * 1 = accumulator mismatch
     * 2 = requantized output mismatch
     */

    for (
        int token = 0;
        token < FC2_TOKENS;
        token++
    ) {
        for (
            int channel = 0;
            channel < FC2_OUT_FEATURES;
            channel++
        ) {
            int32_t acc =
                fc2_accumulate_xqmac8(
                    token,
                    channel
                );

            if (
                acc
                !=
                fc2_golden_acc[token][channel]
            ) {
                return 1;
            }

            int32_t output =
                xrequant(
                    acc,
                    fc2_multiplier[channel]
                );

            if (
                (int8_t)output
                !=
                fc2_golden_output[token][channel]
            ) {
                return 2;
            }
        }
    }

    return 0;
}
