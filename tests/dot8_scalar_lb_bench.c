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


static inline int32_t load_i8(
    const volatile int8_t *ptr
)
{
    int64_t value;

    __asm__ volatile (
        "lb %0, 0(%1)"
        : "=r"(value)
        : "r"(ptr)
        : "memory"
    );

    return (int32_t)value;
}


int main(void)
{
    int32_t acc = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        acc += load_i8(&a[0]) * load_i8(&b[0]);
        acc += load_i8(&a[1]) * load_i8(&b[1]);
        acc += load_i8(&a[2]) * load_i8(&b[2]);
        acc += load_i8(&a[3]) * load_i8(&b[3]);

        acc += load_i8(&a[4]) * load_i8(&b[4]);
        acc += load_i8(&a[5]) * load_i8(&b[5]);
        acc += load_i8(&a[6]) * load_i8(&b[6]);
        acc += load_i8(&a[7]) * load_i8(&b[7]);
    }

    benchmark_sink = acc;

    return (acc == 8000000) ? 0 : 1;
}
