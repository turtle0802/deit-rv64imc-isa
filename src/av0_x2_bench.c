#include <stdint.h>

#include "generated/av0_data.h"


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
        : "r"(a),
          "r"(b)
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
                const int8_t *p =
                    &av0_prob[h][i][0];

                const int8_t *v =
                    &av0_vt[h][d][0];

                int32_t acc = 0;

                for (
                    int j = 0;
                    j < AV0_PAD_TOKENS;
                    j += 8
                ) {
                    acc = xqmac8(
                        acc,
                        load64(p + j),
                        load64(v + j)
                    );
                }

                if (
                    acc
                    !=
                    av0_golden_acc[h][i][d]
                ) {
                    return 1;
                }

                int8_t result =
                    xrequant(
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
