#include <stdint.h>

#define XQMAC8_RANDOM_CASES   10000
#define XREQUANT_RANDOM_CASES 10000


/*
 * Deterministic PRNG.
 *
 * Fixed seed makes the verification exactly reproducible.
 */
static uint64_t prng_state =
    0x123456789abcdef0ULL;

static uint64_t prng_next(void)
{
    uint64_t x = prng_state;

    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    prng_state = x;

    return x;
}


/*
 * Interpret a 32-bit bit pattern as signed INT32
 * without relying on out-of-range unsigned->signed cast.
 */
static int32_t bits_to_i32(
    uint32_t value
)
{
    if ((value & 0x80000000U) == 0) {
        return (int32_t)value;
    }

    return (int32_t)(
        (int64_t)value
        - 0x100000000LL
    );
}


/* ============================================================
 * XQMAC8 custom instruction
 * ============================================================
 */
static inline int32_t xqmac8(
    int32_t acc,
    uint64_t a,
    uint64_t b
)
{
    int64_t result =
        (int64_t)acc;

    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "+r"(result)
        : "r"(a),
          "r"(b)
    );

    return (int32_t)result;
}


/*
 * Scalar golden model for XQMAC8.
 *
 * Semantics:
 *
 *   rd =
 *     int32(
 *       acc +
 *       sum_{i=0}^{7}
 *         int8(a[i]) * int8(b[i])
 *     )
 *
 * INT32 wrap-around is explicitly modeled.
 */
static int32_t xqmac8_reference(
    int32_t acc,
    uint64_t a,
    uint64_t b
)
{
    int64_t sum =
        (int64_t)acc;

    for (int i = 0; i < 8; i++) {
        int8_t av =
            (int8_t)(
                (a >> (8 * i))
                & 0xff
            );

        int8_t bv =
            (int8_t)(
                (b >> (8 * i))
                & 0xff
            );

        sum +=
            (int32_t)av
            *
            (int32_t)bv;
    }

    /*
     * Architectural result wraps to INT32,
     * then is sign-extended by Spike.
     */
    uint32_t wrapped =
        (uint32_t)sum;

    return bits_to_i32(wrapped);
}


/* ============================================================
 * XREQUANT custom instruction
 * ============================================================
 */
static inline int32_t xrequant(
    int32_t acc,
    int32_t multiplier
)
{
    int64_t result;

    __asm__ volatile (
        ".insn r 0x0b, 1, 0, %0, %1, %2"
        : "=r"(result)
        : "r"((int64_t)acc),
          "r"((int64_t)multiplier)
    );

    return (int32_t)result;
}


/*
 * Golden model matching the current Q31 requantization:
 *
 * product = INT32(acc) * INT32(multiplier)
 *
 * round-to-nearest, ties-away-from-zero
 * shift = 31
 *
 * saturation = [-127, 127]
 * zero point = 0
 */
static int32_t xrequant_reference(
    int32_t acc,
    int32_t multiplier
)
{
    int64_t product =
        (int64_t)acc
        *
        (int64_t)multiplier;

    uint64_t magnitude;

    if (product < 0) {
        magnitude =
            (uint64_t)(-product);
    }
    else {
        magnitude =
            (uint64_t)product;
    }

    int64_t rounded =
        (int64_t)(
            (
                magnitude
                + (1ULL << 30)
            )
            >> 31
        );

    if (product < 0) {
        rounded = -rounded;
    }

    if (rounded > 127) {
        rounded = 127;
    }

    if (rounded < -127) {
        rounded = -127;
    }

    return (int32_t)rounded;
}


/*
 * Failure information is kept in memory so that a failing
 * case can be inspected with a debugger if necessary.
 */
volatile uint32_t failure_test;
volatile uint32_t failure_case;

volatile int32_t failure_acc;
volatile int32_t failure_multiplier;

volatile uint64_t failure_a;
volatile uint64_t failure_b;

volatile int32_t failure_expected;
volatile int32_t failure_actual;


static int test_xqmac8_random(void)
{
    for (
        uint32_t i = 0;
        i < XQMAC8_RANDOM_CASES;
        i++
    ) {
        int32_t acc =
            bits_to_i32(
                (uint32_t)prng_next()
            );

        uint64_t a =
            prng_next();

        uint64_t b =
            prng_next();

        int32_t expected =
            xqmac8_reference(
                acc,
                a,
                b
            );

        int32_t actual =
            xqmac8(
                acc,
                a,
                b
            );

        if (actual != expected) {
            failure_test = 1;
            failure_case = i;

            failure_acc = acc;
            failure_a = a;
            failure_b = b;

            failure_expected = expected;
            failure_actual = actual;

            return 1;
        }
    }

    return 0;
}


static int test_xrequant_random(void)
{
    for (
        uint32_t i = 0;
        i < XREQUANT_RANDOM_CASES;
        i++
    ) {
        int32_t acc =
            bits_to_i32(
                (uint32_t)prng_next()
            );

        int32_t multiplier =
            bits_to_i32(
                (uint32_t)prng_next()
            );

        int32_t expected =
            xrequant_reference(
                acc,
                multiplier
            );

        int32_t actual =
            xrequant(
                acc,
                multiplier
            );

        if (actual != expected) {
            failure_test = 2;
            failure_case = i;

            failure_acc = acc;
            failure_multiplier =
                multiplier;

            failure_expected = expected;
            failure_actual = actual;

            return 1;
        }
    }

    return 0;
}


int main(void)
{
    failure_test = 0;
    failure_case = 0;

    /*
     * Return codes:
     *
     * 0 = all randomized tests PASS
     * 1 = XQMAC8 randomized test failed
     * 2 = XREQUANT randomized test failed
     */

    if (test_xqmac8_random() != 0) {
        return 1;
    }

    if (test_xrequant_random() != 0) {
        return 2;
    }

    return 0;
}
