#include <stdint.h>

#define ITERATIONS 1000000

static volatile int8_t a[8]
    __attribute__((aligned(8))) =
{
     1, -2,  3, -4,
     5, -6,  7, -8
};

static volatile int8_t b[8]
    __attribute__((aligned(8))) =
{
     9, 10, -11, -12,
    13, 14, -15, -16
};

volatile int32_t benchmark_sink;


static inline uint64_t load_u64(
    const volatile void *ptr
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
    uint64_t packed_a,
    uint64_t packed_b
)
{
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "+r"(acc)
        : "r"(packed_a),
          "r"(packed_b)
    );

    return acc;
}


int main(void)
{
    int64_t acc = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        uint64_t packed_a =
            load_u64(a);

        uint64_t packed_b =
            load_u64(b);

        acc =
            xqmac8(
                acc,
                packed_a,
                packed_b
            );
    }

    benchmark_sink =
        (int32_t)acc;

    return (
        (int32_t)acc == 8000000
    ) ? 0 : 1;
}
