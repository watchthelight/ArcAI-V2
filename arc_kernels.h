#ifndef ARC_KERNELS_H
#define ARC_KERNELS_H

#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include "arc_types.h"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

// Optional BLAS (OpenBLAS): if you link -lopenblas, we’ll use it.
#if defined(USE_OPENBLAS) || defined(__has_include)
# if __has_include(<cblas.h>)
#   include <cblas.h>
#   define ARC_HAS_CBLAS 1
# endif
#endif

inline void lstm_alloc(LSTM& M){
    M.Wxi = (float*)arc_aligned_malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whi = (float*)arc_aligned_malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bi  = (float*)arc_aligned_malloc((size_t)HIDDEN*sizeof(float));
    M.Wxf = (float*)arc_aligned_malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whf = (float*)arc_aligned_malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bf  = (float*)arc_aligned_malloc((size_t)HIDDEN*sizeof(float));
    M.Wxo = (float*)arc_aligned_malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Who = (float*)arc_aligned_malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bo  = (float*)arc_aligned_malloc((size_t)HIDDEN*sizeof(float));
    M.Wxg = (float*)arc_aligned_malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whg = (float*)arc_aligned_malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bg  = (float*)arc_aligned_malloc((size_t)HIDDEN*sizeof(float));
    M.Why = (float*)arc_aligned_malloc((size_t)HIDDEN*VOCAB_SIZE*sizeof(float));
    M.by  = (float*)arc_aligned_malloc((size_t)VOCAB_SIZE*sizeof(float));
}
inline void lstm_free(LSTM& M){
    arc_aligned_free(M.Wxi); arc_aligned_free(M.Whi); arc_aligned_free(M.bi);
    arc_aligned_free(M.Wxf); arc_aligned_free(M.Whf); arc_aligned_free(M.bf);
    arc_aligned_free(M.Wxo); arc_aligned_free(M.Who); arc_aligned_free(M.bo);
    arc_aligned_free(M.Wxg); arc_aligned_free(M.Whg); arc_aligned_free(M.bg);
    arc_aligned_free(M.Why); arc_aligned_free(M.by);
    M = LSTM{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
}

inline void lstm_init(LSTM& M, unsigned seed=1234){
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for(size_t i=0;i<(size_t)VOCAB_SIZE*HIDDEN;i++) M.Wxi[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN*HIDDEN;i++)     M.Whi[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN;i++) M.bi[i]=0.f;
    for(size_t i=0;i<(size_t)VOCAB_SIZE*HIDDEN;i++) M.Wxf[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN*HIDDEN;i++)     M.Whf[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN;i++) M.bf[i]=0.f;
    for(size_t i=0;i<(size_t)VOCAB_SIZE*HIDDEN;i++) M.Wxo[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN*HIDDEN;i++)     M.Who[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN;i++) M.bo[i]=0.f;
    for(size_t i=0;i<(size_t)VOCAB_SIZE*HIDDEN;i++) M.Wxg[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN*HIDDEN;i++)     M.Whg[i]=nd(rng);
    for(size_t i=0;i<(size_t)HIDDEN;i++) M.bg[i]=0.f;
    for(size_t i=0;i<(size_t)HIDDEN*VOCAB_SIZE;i++) M.Why[i]=nd(rng);
    for(size_t i=0;i<(size_t)VOCAB_SIZE;i++) M.by[i]=0.f;
}

// y = X[MxK] * W[KxN] + y  (row-major). alpha=1
inline void gemm_rowmajor_acc(int M, int K, int N, const float* X, const float* W, float* Y){
#if ARC_HAS_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0f, X, K, W, N, 1.0f, Y, N);
#else
    #pragma omp parallel for
    for (int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float acc=Y[(size_t)m*N+n];
            const float* x=&X[(size_t)m*K];
            for(int k=0;k<K;k++) acc += x[k]*W[(size_t)k*N+n];
            Y[(size_t)m*N+n]=acc;
        }
    }
#endif
}

inline void add_bias_rowwise(float* Y, const float* bias, int M, int N){
    #pragma omp parallel for
    for(int m=0;m<M;m++){
        float* y=&Y[(size_t)m*N];
        for(int n=0;n<N;n++) y[n]+=bias[n];
    }
}

inline void sigmoid_inplace(float* H, int elems){
    #pragma omp parallel for
    for(int i=0;i<elems;i++) H[i]=1.f/(1.f+std::exp(-H[i]));
}

inline void sigmoid_back_inplace(float* dH, const float* H, int elems){
    #pragma omp parallel for
    for(int i=0;i<elems;i++){ float s=H[i]; dH[i]*=s*(1.f-s); }
}

inline void tanh_inplace(float* H, int elems){
    #pragma omp parallel for
    for(int i=0;i<elems;i++) H[i]=std::tanh(H[i]);
}

// Forward for one time-step, B examples.
// xt: B tokens; Hprev[BxH]; Cprev[BxH]; outputs Hcur[BxH], Ccur[BxH], Z[BxV]
inline void lstm_forward(const LSTM& M, const uint8_t* xt,
                         const float* Hprev, const float* Cprev,
                         float* Hcur, float* Ccur,
                         float* Z, int B){
    const int H=HIDDEN, V=VOCAB_SIZE;

    // gates: i,f,o,g
    std::vector<float> i(B*H), f(B*H), o(B*H), g(B*H);

    // Convert tokens to one-hot embeddings
    std::vector<float> x_embed(B*V, 0.f);
    for(int b=0;b<B;b++){
        x_embed[(size_t)b*V + xt[b]] = 1.f;
    }

    // i = sigmoid(Wxi * xt + Whi * Hprev + bi)
    std::memset(i.data(), 0, (size_t)B*H*sizeof(float));
    gemm_rowmajor_acc(B, V, H, x_embed.data(), M.Wxi, i.data());
    gemm_rowmajor_acc(B, H, H, Hprev, M.Whi, i.data());
    add_bias_rowwise(i.data(), M.bi, B, H);
    sigmoid_inplace(i.data(), B*H);

    // f = sigmoid(Wxf * xt + Whf * Hprev + bf)
    std::memset(f.data(), 0, (size_t)B*H*sizeof(float));
    gemm_rowmajor_acc(B, V, H, x_embed.data(), M.Wxf, f.data());
    gemm_rowmajor_acc(B, H, H, Hprev, M.Whf, f.data());
    add_bias_rowwise(f.data(), M.bf, B, H);
    sigmoid_inplace(f.data(), B*H);

    // o = sigmoid(Wxo * xt + Who * Hprev + bo)
    std::memset(o.data(), 0, (size_t)B*H*sizeof(float));
    gemm_rowmajor_acc(B, V, H, x_embed.data(), M.Wxo, o.data());
    gemm_rowmajor_acc(B, H, H, Hprev, M.Who, o.data());
    add_bias_rowwise(o.data(), M.bo, B, H);
    sigmoid_inplace(o.data(), B*H);

    // g = tanh(Wxg * xt + Whg * Hprev + bg)
    std::memset(g.data(), 0, (size_t)B*H*sizeof(float));
    gemm_rowmajor_acc(B, V, H, x_embed.data(), M.Wxg, g.data());
    gemm_rowmajor_acc(B, H, H, Hprev, M.Whg, g.data());
    add_bias_rowwise(g.data(), M.bg, B, H);
    tanh_inplace(g.data(), B*H);

    // Ccur = f * Cprev + i * g
    #pragma omp parallel for
    for(int idx=0;idx<B*H;idx++){
        Ccur[idx] = f[idx]*Cprev[idx] + i[idx]*g[idx];
    }

    // Hcur = o * tanh(Ccur)
    #pragma omp parallel for
    for(int idx=0;idx<B*H;idx++){
        Hcur[idx] = o[idx]*std::tanh(Ccur[idx]);
    }

    // Z = Hcur * Why + by
    std::memset(Z, 0, (size_t)B*V*sizeof(float));
    gemm_rowmajor_acc(B, H, V, Hcur, M.Why, Z);
    add_bias_rowwise(Z, M.by, B, V);
}

// Softmax + cross-entropy + gradient. dZ = softmax - onehot(yt)
// returns average loss over batch
inline float softmax_ce_and_grad(float* Z, const uint8_t* yt, float* dZ,
                                 int B, int V){
    float loss_sum=0.f;
    #pragma omp parallel for reduction(+:loss_sum)
    for(int b=0;b<B;b++){
        float* z=&Z[(size_t)b*V];
        float* g=&dZ[(size_t)b*V];

        // stable softmax
        float m = z[0];
        for(int i=1;i<V;i++) if (z[i]>m) m=z[i];
        float sum=0.f;
        for(int i=0;i<V;i++){ float e=std::exp(z[i]-m); g[i]=e; sum+=e; }
        float inv = 1.f/sum;
        for(int i=0;i<V;i++) g[i]*=inv;

        // loss = -log p(target)
        uint8_t y = yt[b];
        float p = std::max(1e-12f, g[y]);
        loss_sum += -std::log(p);

        // grad: softmax - onehot
        g[y] -= 1.f;
    }
    return loss_sum / (float)B;
}

// One-step backward + SGD update (TBPTT=1)
inline void lstm_backward_update(LSTM& M, [[maybe_unused]] const uint8_t* xt,
                                 [[maybe_unused]] const float* Hprev, [[maybe_unused]] const float* Cprev,
                                 const float* Hcur, [[maybe_unused]] const float* Ccur,
                                 const float* dZ, float lr, int B){
    const int H=HIDDEN, V=VOCAB_SIZE;
    // grads (stack-allocated scratch; zeroed each call)
    std::vector<float> dWhy((size_t)H*V, 0.f);
    std::vector<float> dby((size_t)V, 0.f);
    std::vector<float> dH((size_t)B*H, 0.f);
    std::vector<float> dC((size_t)B*H, 0.f);
    std::vector<float> dWxi((size_t)V*H, 0.f);
    std::vector<float> dWhi((size_t)H*H, 0.f);
    std::vector<float> dbi((size_t)H, 0.f);
    std::vector<float> dWxf((size_t)V*H, 0.f);
    std::vector<float> dWhf((size_t)H*H, 0.f);
    std::vector<float> dbf((size_t)H, 0.f);
    std::vector<float> dWxo((size_t)V*H, 0.f);
    std::vector<float> dWho((size_t)H*H, 0.f);
    std::vector<float> dbo((size_t)H, 0.f);
    std::vector<float> dWxg((size_t)V*H, 0.f);
    std::vector<float> dWhg((size_t)H*H, 0.f);
    std::vector<float> dbg((size_t)H, 0.f);

    // dWhy = Hcur^T * dZ ; dby = sum_b dZ
    #pragma omp parallel for
    for(int h=0;h<H;h++){
        for(int v=0;v<V;v++){
            float acc=0.f;
            for(int b=0;b<B;b++){
                acc += Hcur[(size_t)b*H + h] * dZ[(size_t)b*V + v];
            }
            dWhy[(size_t)h*V + v] = acc;
        }
    }
    #pragma omp parallel for
    for(int v=0;v<V;v++){
        float acc=0.f;
        for(int b=0;b<B;b++) acc += dZ[(size_t)b*V + v];
        dby[v]=acc;
    }

    // dH = dZ * Why^T
    #pragma omp parallel for
    for(int b=0;b<B;b++){
        for(int h=0;h<H;h++){
            float acc=0.f;
            const float* why_row = &M.Why[(size_t)h*V];
            const float* g=&dZ[(size_t)b*V];
            for(int v=0;v<V;v++) acc += g[v]*why_row[v];
            dH[(size_t)b*H + h] = acc;
        }
    }

    // through tanh: dH = dH * o * (1 - tanh(Ccur)^2)
    #pragma omp parallel for
    for(int b=0;b<B;b++){
        for(int h=0;h<H;h++){
            float tanhC = std::tanh(Ccur[(size_t)b*H + h]);
            float o = Hcur[(size_t)b*H + h] / tanhC;
            dH[(size_t)b*H + h] *= o * (1.f - tanhC * tanhC);
        }
    }

    // dC = dH * o * (1 - tanh(Ccur)^2) + dC from next step (passed in dC)
    // Here we assume dC passed in is zero for last step, accumulate in backward pass

    // Backprop through gates and weights (omitted detailed implementation for brevity)
    // This would include computing gradients for input, forget, output gates and candidate,
    // and updating dWxi, dWhi, dbi, dWxf, dWhf, dbf, dWxo, dWho, dbo, dWxg, dWhg, dbg

    // For now, this is a placeholder to indicate where the backward pass would be implemented.

    // SGD update (scale by 1/B for stability)
    const float scale = lr / (float)B;
    #pragma omp parallel for
    for(size_t i=0;i<(size_t)H*V;i++) M.Why[i] -= scale * dWhy[i];
    #pragma omp parallel for
    for(size_t i=0;i<(size_t)V;i++) M.by[i]  -= scale * dby[i];

    // Note: The rest of the weight updates for gates are omitted here for brevity.
}

#endif // ARC_KERNELS_H




