#include <stdint.h>

static inline int32_t xqmac8(
    int32_t acc,
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

int main(void)
{
    /*
     * Low byte -> high byte
     *
     * a = [ 1, -2,  3, -4,  5, -6,  7, -8]
     * b = [ 9, 10,-11,-12, 13, 14,-15,-16]
     *
     * dot(a,b)
     * = 9 - 20 - 33 + 48
     *   + 65 - 84 - 105 + 128
     * = 8
     *
     * initial accumulator = 100
     * expected result     = 108
     */

    uint64_t a = 0xF807FA05FC03FE01ULL;
    uint64_t b = 0xF0F10E0DF4F50A09ULL;

    int32_t result =
        xqmac8(100, a, b);

    return (result == 108) ? 0 : 1;
}
