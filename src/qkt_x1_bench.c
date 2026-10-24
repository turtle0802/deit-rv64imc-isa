#include <stdint.h>

#include "generated/qkt_data.h"

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
        : "r"(a),
          "r"(b)
    );

    return (int32_t)rd;
}


int main(void)
{
    uint64_t checksum = 0;

    for (
        int h = 0;
        h < QKT_HEADS;
        h++
    ) {
        for (
            int i = 0;
            i < QKT_TOKENS;
            i++
        ) {
            for (
                int j = 0;
                j < QKT_TOKENS;
                j++
            ) {
                const int8_t *q =
                    &qkt_q[h][i][0];

                const int8_t *k =
                    &qkt_k[h][j][0];

                int32_t acc = 0;

                for (
                    int d = 0;
                    d < QKT_HEAD_DIM;
                    d += 8
                ) {
                    acc = xqmac8(
                        acc,
                        load64(q + d),
                        load64(k + d)
                    );
                }

                checksum +=
                    (uint32_t)acc;
            }
        }
    }

    benchmark_sink = checksum;

    return 0;
}
