#ifndef LIGHTWATCH_BPTT_H
#define LIGHTWATCH_BPTT_H

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "lightwatch_types.h"
#include "lightwatch_kernels.h"

// Train one batch (BATCH x SEQ_LEN) with TBPTT=TBPTT_LEN; returns avg CE loss for the sequence
inline float lstm_train_batch(LSTM& M, const uint8_t* batch /*size: BATCH*SEQ_LEN*/) {
    const int B = BATCH, T = SEQ_LEN, H = HIDDEN, V = VOCAB_SIZE;
    // TBPTT = TBPTT_LEN; // Not used in current implementation

    // temps
    float* Hprev = (float*)lightwatch_aligned_malloc((size_t)B*H*sizeof(float));
    float* Cprev = (float*)lightwatch_aligned_malloc((size_t)B*H*sizeof(float));
    float* Hcur  = (float*)lightwatch_aligned_malloc((size_t)B*H*sizeof(float));
    float* Ccur  = (float*)lightwatch_aligned_malloc((size_t)B*H*sizeof(float));
    float* Z     = (float*)lightwatch_aligned_malloc((size_t)B*V*sizeof(float));
    float* dZ    = (float*)lightwatch_aligned_malloc((size_t)B*V*sizeof(float));
    std::memset(Hprev, 0, (size_t)B*H*sizeof(float));
    std::memset(Cprev, 0, (size_t)B*H*sizeof(float));

    float loss_sum = 0.0f;

    for (int t = 0; t < T-1; ++t) {
        const uint8_t* xt = &batch[(size_t)t * B];
        const uint8_t* yt = &batch[(size_t)(t + 1) * B];

        lstm_forward(M, xt, Hprev, Cprev, Hcur, Ccur, Z, B);
        float loss = softmax_ce_and_grad(Z, yt, dZ, B, V);
        loss_sum += loss;
        // For simplicity, use lstm_backward_update as placeholder; full LSTM backward needed
        lstm_backward_update(M, xt, Hprev, Cprev, Hcur, Ccur, dZ, LR, B);

        // roll
        std::memcpy(Hprev, Hcur, (size_t)B*H*sizeof(float));
        std::memcpy(Cprev, Ccur, (size_t)B*H*sizeof(float));
    }

    lightwatch_aligned_free(Hprev); lightwatch_aligned_free(Cprev);
    lightwatch_aligned_free(Hcur); lightwatch_aligned_free(Ccur);
    lightwatch_aligned_free(Z);     lightwatch_aligned_free(dZ);
    return loss_sum / (float)(T-1);
}

#endif // LIGHTWATCH_BPTT_H
