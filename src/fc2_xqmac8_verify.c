#include <stdint.h>

#include "generated/fc2_data.h"


/* ============================================================
 * 64-bit packed load
 *
 * Each load fetches 8 consecutive INT8 values.
 * ============================================================
 */
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


/* ============================================================
 * XQMAC8
 *
 * acc =
 *     int32(acc)
 *     +
 *     sum_{i=0}^{7}
 *         int8(a[i]) * int8(b[i])
 *
 * Spike returns the INT32 result sign-extended to RV64.
 *
 * Use int64_t as the C register container so GCC does not need
 * to insert an unnecessary sext.w after every XQMAC8.
 * ============================================================
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
        : "r"(a), "r"(b)
    );

    return acc;
}


/* ============================================================
 * Fixed-point requantization
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

    int64_t abs_product =
        (product >= 0)
        ? product
        : -product;

    int64_t rounded_abs =
        (
            abs_product
            + (1LL << (FC2_REQUANT_SHIFT - 1))
        )
        >> FC2_REQUANT_SHIFT;

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


/* ============================================================
 * XQMAC8 FC2
 *
 * K = 192
 *
 * 192 / 8 = 24 XQMAC8 instructions per output.
 * ============================================================
 */
static int32_t fc2_accumulate_xqmac8(
    int token,
    int channel
)
{
    /*
     * Bias is INT32.
     * lw on RV64 already sign-extends it.
     */
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
     * 2 = requant output mismatch
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

            int8_t output =
                requantize(
                    acc,
                    fc2_multiplier[channel]
                );

            if (
                output
                !=
                fc2_golden_output[token][channel]
            ) {
                return 2;
            }
        }
    }

    return 0;
}
