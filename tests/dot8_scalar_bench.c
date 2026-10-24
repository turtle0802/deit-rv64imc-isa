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

int main(void)
{
    int32_t acc = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        acc += (int32_t)a[0] * (int32_t)b[0];
        acc += (int32_t)a[1] * (int32_t)b[1];
        acc += (int32_t)a[2] * (int32_t)b[2];
        acc += (int32_t)a[3] * (int32_t)b[3];

        acc += (int32_t)a[4] * (int32_t)b[4];
        acc += (int32_t)a[5] * (int32_t)b[5];
        acc += (int32_t)a[6] * (int32_t)b[6];
        acc += (int32_t)a[7] * (int32_t)b[7];
    }

    benchmark_sink = acc;

    /*
     * dot(a,b) = 8
     * 1,000,000 iterations -> 8,000,000
     */
    return (acc == 8000000) ? 0 : 1;
}
