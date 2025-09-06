#ifndef ARC_GENERATE_H
#define ARC_GENERATE_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "arc_types.h"
#include "arc_kernels.h"

// sample from logits row with temperature
inline uint8_t sample_row(const float* z, float temp){
    std::vector<float> p(VOCAB_SIZE);
    float m=z[0]; for(int i=1;i<VOCAB_SIZE;i++) if(z[i]>m) m=z[i];
    float sum=0.f;
    for(int i=0;i<VOCAB_SIZE;i++){
        float v = (z[i]-m)/std::max(1e-6f, temp);
        float e = std::exp(v);
        p[i]=e; sum+=e;
    }
    float u = (float)rand() / (float)RAND_MAX * sum;
    for(int i=0;i<VOCAB_SIZE;i++){
        if ((u-=p[i])<=0) return (uint8_t)i;
    }
    return (uint8_t)(VOCAB_SIZE-1);
}

inline void generate_text(LSTM& M, uint8_t start, int n_tokens, float temp){
    const int B=1, H=HIDDEN, V=VOCAB_SIZE;
    std::vector<float> Hprev(H,0.f), Hcur(H,0.f), Cprev(H,0.f), Ccur(H,0.f), Z(V,0.f);
    uint8_t tok = start;
    for(int t=0;t<n_tokens;t++){
        lstm_forward(M, &tok, Hprev.data(), Cprev.data(), Hcur.data(), Ccur.data(), Z.data(), B);
        tok = sample_row(Z.data(), temp);
        std::putchar((char)tok);
        std::fflush(stdout);
        std::memcpy(Hprev.data(), Hcur.data(), (size_t)H*sizeof(float));
        std::memcpy(Cprev.data(), Ccur.data(), (size_t)H*sizeof(float));
    }
    std::putchar('\n');
}

#endif // ARC_GENERATE_H

