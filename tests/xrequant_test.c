#include <stdint.h>


static inline int32_t xrequant(
    int32_t acc,
    int32_t multiplier
)
{
    int64_t result;

    /*
     * CUSTOM-0
     * funct3 = 001
     * funct7 = 0000000
     */
    __asm__ volatile (
        ".insn r 0x0b, 1, 0, %0, %1, %2"
        : "=r"(result)
        : "r"((int64_t)acc),
          "r"((int64_t)multiplier)
    );

    return (int32_t)result;
}


int main(void)
{
    /*
     * Q31 representation of 0.5:
     *
     * 0.5 * 2^31 = 2^30
     */
    const int32_t HALF_Q31 =
        (int32_t)(1U << 30);


    /*
     * Case 1:
     *
     * 100 * 0.5 = 50
     */
    if (
        xrequant(
            100,
            HALF_Q31
        ) != 50
    ) {
        return 1;
    }


    /*
     * Case 2:
     *
     * -100 * 0.5 = -50
     */
    if (
        xrequant(
            -100,
            HALF_Q31
        ) != -50
    ) {
        return 2;
    }


    /*
     * Case 3:
     *
     * +0.5
     * ties away from zero
     *
     * expected = +1
     */
    if (
        xrequant(
            1,
            HALF_Q31
        ) != 1
    ) {
        return 3;
    }


    /*
     * Case 4:
     *
     * -0.5
     * ties away from zero
     *
     * expected = -1
     */
    if (
        xrequant(
            -1,
            HALF_Q31
        ) != -1
    ) {
        return 4;
    }


    /*
     * Case 5:
     *
     * 300 * 0.5 = 150
     *
     * saturation:
     * 150 -> 127
     */
    if (
        xrequant(
            300,
            HALF_Q31
        ) != 127
    ) {
        return 5;
    }


    /*
     * Case 6:
     *
     * -300 * 0.5 = -150
     *
     * saturation:
     * -150 -> -127
     */
    if (
        xrequant(
            -300,
            HALF_Q31
        ) != -127
    ) {
        return 6;
    }


    /*
     * Case 7:
     *
     * zero input
     */
    if (
        xrequant(
            0,
            HALF_Q31
        ) != 0
    ) {
        return 7;
    }


    return 0;
}
