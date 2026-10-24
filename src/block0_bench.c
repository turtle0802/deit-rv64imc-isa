#include <stdint.h>

#include "generated/block0_data.h"

#ifndef USE_X2
#define USE_X2 0
#endif

#ifndef VERIFY
#define VERIFY 0
#endif


/* ============================================================
 * Runtime buffers
 * ============================================================ */

static int8_t ln1[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t qkv_out[
    B0_TOKENS * B0_QKV_OUT
] __attribute__((aligned(8)));

static int8_t prob_pad[
    B0_HEADS * B0_TOKENS * B0_PAD_TOKENS
] __attribute__((aligned(8)));

static int8_t v_t_pad[
    B0_HEADS * B0_HEAD_DIM * B0_PAD_TOKENS
] __attribute__((aligned(8)));

static int8_t av_out[
    B0_HEADS * B0_TOKENS * B0_HEAD_DIM
] __attribute__((aligned(8)));

static int8_t proj_input[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t proj_out[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t residual1[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t ln2[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t fc1_out[
    B0_TOKENS * B0_HIDDEN
] __attribute__((aligned(8)));

static int8_t mlp_out[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

static int8_t final_out[
    B0_TOKENS * B0_EMBED
] __attribute__((aligned(8)));

volatile uint64_t benchmark_sink;


/* ============================================================
 * Basic helpers
 * ============================================================ */

static inline int8_t sat8(
    int64_t x
)
{
    if (x > 127)
        return 127;

    if (x < -127)
        return -127;

    return (int8_t)x;
}


static int64_t round_shift_signed(
    int64_t x,
    int shift
)
{
    int64_t half =
        (int64_t)1 << (shift - 1);

    if (x >= 0) {
        return
            (x + half) >> shift;
    }

    return -(
        (
            -x + half
        ) >> shift
    );
}


/* ============================================================
 * Scalar requant baseline
 * ============================================================ */

static inline int8_t scalar_requant(
    int32_t acc,
    int64_t multiplier
)
{
    int64_t product =
        (int64_t)acc
        * multiplier;

    int64_t mag =
        product < 0
        ? -product
        : product;

    int64_t rounded =
        (
            mag
            + (1LL << 30)
        ) >> 31;

    int64_t result =
        product < 0
        ? -rounded
        : rounded;

    return sat8(result);
}


/* ============================================================
 * X2 instructions
 * ============================================================ */

#if USE_X2

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

#endif


/* ============================================================
 * Common MAC interface
 *
 * B0 : scalar INT8 MAC
 * X2 : Xqmac8
 * ============================================================ */

static inline int32_t dot_acc(
    int32_t acc,
    const int8_t *a,
    const int8_t *b,
    int n
)
{
#if USE_X2

    for (
        int k = 0;
        k < n;
        k += 8
    ) {
        acc = xqmac8(
            acc,
            load64(a + k),
            load64(b + k)
        );
    }

#else

    for (
        int k = 0;
        k < n;
        k++
    ) {
        acc +=
            (int32_t)a[k]
            *
            (int32_t)b[k];
    }

#endif

    return acc;
}


static inline int8_t requant(
    int32_t acc,
    int64_t multiplier
)
{
#if USE_X2
    return xrequant(
        acc,
        multiplier
    );
#else
    return scalar_requant(
        acc,
        multiplier
    );
#endif
}


/* ============================================================
 * Integer sqrt / LayerNorm
 * ============================================================ */

static uint64_t isqrt_floor_u64(
    uint64_t x
)
{
    uint64_t op = x;
    uint64_t res = 0;

    uint64_t one =
        (uint64_t)1 << 62;

    while (one > op)
        one >>= 2;

    while (one != 0) {
        if (op >= res + one) {
            op -= res + one;

            res =
                (res >> 1)
                + one;
        }
        else {
            res >>= 1;
        }

        one >>= 2;
    }

    return res;
}


static uint64_t sqrt_nearest_u64(
    uint64_t x
)
{
    uint64_t r =
        isqrt_floor_u64(x);

    uint64_t lo =
        x - r * r;

    uint64_t rp =
        r + 1;

    uint64_t hi =
        rp * rp - x;

    if (hi < lo)
        r = rp;

    return r;
}


static int64_t div_round_signed(
    int64_t num,
    int64_t den
)
{
    if (num >= 0) {
        return
            (
                num
                + den / 2
            ) / den;
    }

    return -(
        (
            -num
            + den / 2
        ) / den
    );
}


static void integer_layernorm(
    const int8_t *input,
    int8_t *output,
    const int64_t *affine_a,
    const int64_t *affine_b,
    int64_t eps_term
)
{
    const int N =
        B0_EMBED;

    for (
        int token = 0;
        token < B0_TOKENS;
        token++
    ) {
        const int8_t *row =
            &input[
                token * N
            ];

        int64_t sum = 0;
        int64_t sumsq = 0;

        for (
            int c = 0;
            c < N;
            c++
        ) {
            int64_t x =
                row[c];

            sum += x;
            sumsq += x * x;
        }

        int64_t var_num =
            (int64_t)N * sumsq
            - sum * sum
            + eps_term;

        if (var_num <= 0)
            var_num = 1;

        int64_t denom =
            (int64_t)
            sqrt_nearest_u64(
                (uint64_t)var_num
            );

        for (
            int c = 0;
            c < N;
            c++
        ) {
            int64_t centered_num =
                (int64_t)row[c]
                * N
                - sum;

            int64_t z_q15 =
                div_round_signed(
                    centered_num
                    * (
                        (int64_t)1
                        << B0_LN_Z_Q
                    ),
                    denom
                );

            int64_t value =
                z_q15
                * affine_a[c]
                +
                affine_b[c]
                * (
                    (int64_t)1
                    << B0_LN_Z_Q
                );

            int64_t y =
                round_shift_signed(
                    value,
                    B0_LN_Z_Q
                    + B0_LN_AFFINE_Q
                );

            output[
                token * N + c
            ] = sat8(y);
        }
    }
}


/* ============================================================
 * Integer Softmax helper
 * ============================================================ */

static uint32_t get_exp_q15(
    int32_t score,
    int32_t row_max
)
{
    uint32_t distance =
        (uint32_t)(
            row_max - score
        );

    uint64_t numerator =
        (uint64_t)distance
        * 255u;

    numerator +=
        B0_SOFTMAX_CLIP_STEPS
        / 2;

    uint32_t index =
        (uint32_t)(
            numerator
            /
            B0_SOFTMAX_CLIP_STEPS
        );

    if (index > 255)
        index = 255;

    return
        b0_exp_lut[index];
}


/* ============================================================
 * Main
 * ============================================================ */

int main(void)
{
    /* ========================================================
     * 1. LN1
     * ======================================================== */

    integer_layernorm(
        b0_x0_q,
        ln1,
        b0_ln1_a,
        b0_ln1_b,
        b0_ln1_eps_term
    );


    /* ========================================================
     * 2. QKV
     * ======================================================== */

    for (
        int t = 0;
        t < B0_TOKENS;
        t++
    ) {
        const int8_t *x =
            &ln1[
                t * B0_EMBED
            ];

        for (
            int o = 0;
            o < B0_QKV_OUT;
            o++
        ) {
            const int8_t *w =
                &b0_qkv_weight[
                    o * B0_EMBED
                ];

            int32_t acc =
                dot_acc(
                    b0_qkv_bias[o],
                    x,
                    w,
                    B0_EMBED
                );

            qkv_out[
                t * B0_QKV_OUT
                + o
            ] = requant(
                acc,
                b0_qkv_multiplier[o]
            );
        }
    }


    /* ========================================================
     * 3. QK^T + Softmax
     * ======================================================== */

    int32_t scores[
        B0_TOKENS
    ];

    for (
        int h = 0;
        h < B0_HEADS;
        h++
    ) {
        for (
            int i = 0;
            i < B0_TOKENS;
            i++
        ) {
            const int8_t *q =
                &qkv_out[
                    i * B0_QKV_OUT
                    + h * B0_HEAD_DIM
                ];

            int32_t row_max = 0;

            for (
                int j = 0;
                j < B0_TOKENS;
                j++
            ) {
                const int8_t *k =
                    &qkv_out[
                        j * B0_QKV_OUT
                        + B0_EMBED
                        + h * B0_HEAD_DIM
                    ];

                int32_t acc =
                    dot_acc(
                        0,
                        q,
                        k,
                        B0_HEAD_DIM
                    );

                scores[j] = acc;

                if (
                    j == 0
                    || acc > row_max
                ) {
                    row_max = acc;
                }
            }

            uint64_t exp_sum = 0;

            for (
                int j = 0;
                j < B0_TOKENS;
                j++
            ) {
                exp_sum +=
                    get_exp_q15(
                        scores[j],
                        row_max
                    );
            }

            for (
                int j = 0;
                j < B0_TOKENS;
                j++
            ) {
                uint64_t e =
                    get_exp_q15(
                        scores[j],
                        row_max
                    );

                uint64_t numerator =
                    e
                    * B0_PROB_QMAX;

                uint32_t p =
                    (uint32_t)(
                        (
                            numerator
                            + exp_sum / 2
                        )
                        / exp_sum
                    );

                if (
                    p > B0_PROB_QMAX
                ) {
                    p =
                        B0_PROB_QMAX;
                }

                prob_pad[
                    (
                        h * B0_TOKENS
                        + i
                    )
                    * B0_PAD_TOKENS
                    + j
                ] = (int8_t)p;
            }

            for (
                int j = B0_TOKENS;
                j < B0_PAD_TOKENS;
                j++
            ) {
                prob_pad[
                    (
                        h * B0_TOKENS
                        + i
                    )
                    * B0_PAD_TOKENS
                    + j
                ] = 0;
            }
        }
    }


    /* ========================================================
     * 4. Prepare padded V^T
     * ======================================================== */

    for (
        int h = 0;
        h < B0_HEADS;
        h++
    ) {
        for (
            int d = 0;
            d < B0_HEAD_DIM;
            d++
        ) {
            for (
                int j = 0;
                j < B0_PAD_TOKENS;
                j++
            ) {
                int8_t value = 0;

                if (
                    j < B0_TOKENS
                ) {
                    value =
                        qkv_out[
                            j * B0_QKV_OUT
                            + 2 * B0_EMBED
                            + h * B0_HEAD_DIM
                            + d
                        ];
                }

                v_t_pad[
                    (
                        h * B0_HEAD_DIM
                        + d
                    )
                    * B0_PAD_TOKENS
                    + j
                ] = value;
            }
        }
    }


    /* ========================================================
     * 5. Attention × V
     * ======================================================== */

    for (
        int h = 0;
        h < B0_HEADS;
        h++
    ) {
        for (
            int i = 0;
            i < B0_TOKENS;
            i++
        ) {
            const int8_t *p =
                &prob_pad[
                    (
                        h * B0_TOKENS
                        + i
                    )
                    * B0_PAD_TOKENS
                ];

            for (
                int d = 0;
                d < B0_HEAD_DIM;
                d++
            ) {
                const int8_t *v =
                    &v_t_pad[
                        (
                            h * B0_HEAD_DIM
                            + d
                        )
                        * B0_PAD_TOKENS
                    ];

                int32_t acc =
                    dot_acc(
                        0,
                        p,
                        v,
                        B0_PAD_TOKENS
                    );

                av_out[
                    (
                        h * B0_TOKENS
                        + i
                    )
                    * B0_HEAD_DIM
                    + d
                ] = requant(
                    acc,
                    b0_av_multiplier
                );
            }
        }
    }


    /* ========================================================
     * 6. Concatenate heads
     * ======================================================== */

    for (
        int t = 0;
        t < B0_TOKENS;
        t++
    ) {
        for (
            int h = 0;
            h < B0_HEADS;
            h++
        ) {
            for (
                int d = 0;
                d < B0_HEAD_DIM;
                d++
            ) {
                proj_input[
                    t * B0_EMBED
                    + h * B0_HEAD_DIM
                    + d
                ] =
                    av_out[
                        (
                            h * B0_TOKENS
                            + t
                        )
                        * B0_HEAD_DIM
                        + d
                    ];
            }
        }
    }


    /* ========================================================
     * 7. Projection
     * ======================================================== */

    for (
        int t = 0;
        t < B0_TOKENS;
        t++
    ) {
        const int8_t *x =
            &proj_input[
                t * B0_EMBED
            ];

        for (
            int o = 0;
            o < B0_EMBED;
            o++
        ) {
            const int8_t *w =
                &b0_proj_weight[
                    o * B0_EMBED
                ];

            int32_t acc =
                dot_acc(
                    b0_proj_bias[o],
                    x,
                    w,
                    B0_EMBED
                );

            proj_out[
                t * B0_EMBED + o
            ] = requant(
                acc,
                b0_proj_multiplier[o]
            );
        }
    }


    /* ========================================================
     * 8. Residual 1
     * ======================================================== */

    for (
        int i = 0;
        i < B0_TOKENS * B0_EMBED;
        i++
    ) {
        int16_t a =
            requant(
                b0_x0_q[i],
                b0_x0_to_r1
            );

        int16_t b =
            requant(
                proj_out[i],
                b0_proj_to_r1
            );

        residual1[i] =
            sat8(
                (int64_t)a + b
            );
    }


    /* ========================================================
     * 9. LN2
     * ======================================================== */

    integer_layernorm(
        residual1,
        ln2,
        b0_ln2_a,
        b0_ln2_b,
        b0_ln2_eps_term
    );


    /* ========================================================
     * 10. FC1
     * ======================================================== */

    for (
        int t = 0;
        t < B0_TOKENS;
        t++
    ) {
        const int8_t *x =
            &ln2[
                t * B0_EMBED
            ];

        for (
            int o = 0;
            o < B0_HIDDEN;
            o++
        ) {
            const int8_t *w =
                &b0_fc1_weight[
                    o * B0_EMBED
                ];

            int32_t acc =
                dot_acc(
                    b0_fc1_bias[o],
                    x,
                    w,
                    B0_EMBED
                );

            fc1_out[
                t * B0_HIDDEN
                + o
            ] = requant(
                acc,
                b0_fc1_multiplier[o]
            );
        }
    }


    /* ========================================================
     * 11. GELU LUT
     * ======================================================== */

    for (
        int i = 0;
        i < B0_TOKENS * B0_HIDDEN;
        i++
    ) {
        fc1_out[i] =
            b0_gelu_lut[
                (uint8_t)fc1_out[i]
            ];
    }


    /* ========================================================
     * 12. FC2
     * ======================================================== */

    for (
        int t = 0;
        t < B0_TOKENS;
        t++
    ) {
        const int8_t *x =
            &fc1_out[
                t * B0_HIDDEN
            ];

        for (
            int o = 0;
            o < B0_EMBED;
            o++
        ) {
            const int8_t *w =
                &b0_fc2_weight[
                    o * B0_HIDDEN
                ];

            int32_t acc =
                dot_acc(
                    b0_fc2_bias[o],
                    x,
                    w,
                    B0_HIDDEN
                );

            mlp_out[
                t * B0_EMBED
                + o
            ] = requant(
                acc,
                b0_fc2_multiplier[o]
            );
        }
    }


    /* ========================================================
     * 13. Residual 2
     * ======================================================== */

    for (
        int i = 0;
        i < B0_TOKENS * B0_EMBED;
        i++
    ) {
        int16_t a =
            requant(
                residual1[i],
                b0_r1_to_final
            );

        int16_t b =
            requant(
                mlp_out[i],
                b0_mlp_to_final
            );

        final_out[i] =
            sat8(
                (int64_t)a + b
            );
    }


    /* ========================================================
     * Correctness check when requested
     * ======================================================== */

#if VERIFY

    for (
        int i = 0;
        i < B0_TOKENS * B0_EMBED;
        i++
    ) {
        if (
            final_out[i]
            !=
            b0_gold_final[i]
        ) {
            return 1;
        }
    }

#endif


    /* ========================================================
     * Prevent dead-code elimination
     * ======================================================== */

    uint64_t checksum = 0;

    for (
        int i = 0;
        i < B0_TOKENS * B0_EMBED;
        i++
    ) {
        checksum =
            checksum * 131u
            +
            (uint8_t)final_out[i];
    }

    benchmark_sink =
        checksum;

    return 0;
}
