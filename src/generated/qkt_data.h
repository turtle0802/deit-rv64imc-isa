#ifndef QKT_DATA_H
#define QKT_DATA_H

#include <stdint.h>

#define QKT_HEADS    3
#define QKT_TOKENS   197
#define QKT_HEAD_DIM 64

extern const int8_t
qkt_q[QKT_HEADS][QKT_TOKENS][QKT_HEAD_DIM];

extern const int8_t
qkt_k[QKT_HEADS][QKT_TOKENS][QKT_HEAD_DIM];

extern const int32_t
qkt_golden_acc[QKT_HEADS][QKT_TOKENS][QKT_TOKENS];

#endif
