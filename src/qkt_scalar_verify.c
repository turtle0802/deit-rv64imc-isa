#include <stdint.h>

#include "generated/qkt_data.h"


int main(void)
{
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
                int32_t acc = 0;

                for (
                    int d = 0;
                    d < QKT_HEAD_DIM;
                    d++
                ) {
                    acc +=
                        (int32_t)qkt_q[h][i][d]
                        *
                        (int32_t)qkt_k[h][j][d];
                }

                if (
                    acc
                    !=
                    qkt_golden_acc[h][i][j]
                ) {
                    return 1;
                }
            }
        }
    }

    return 0;
}
