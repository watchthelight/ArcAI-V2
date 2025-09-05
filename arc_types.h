#ifndef ARC_TYPES_H
#define ARC_TYPES_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>

#ifndef VOCAB_SIZE
#define VOCAB_SIZE 256
#endif
#ifndef HIDDEN
#define HIDDEN 256
#endif
#ifndef BATCH
#define BATCH 16
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 64
#endif
#ifndef LR
#define LR 0.001f
#endif
#ifndef TBPTT_LEN
#define TBPTT_LEN 32
#endif

// LSTM Model
struct LSTM {
    // Input gate: Wxi, Whi, bi
    float* Wxi; // [VOCAB_SIZE x HIDDEN]
    float* Whi; // [HIDDEN x HIDDEN]
    float* bi;  // [HIDDEN]
    // Forget gate: Wxf, Whf, bf
    float* Wxf; // [VOCAB_SIZE x HIDDEN]
    float* Whf; // [HIDDEN x HIDDEN]
    float* bf;  // [HIDDEN]
    // Output gate: Wxo, Who, bo
    float* Wxo; // [VOCAB_SIZE x HIDDEN]
    float* Who; // [HIDDEN x HIDDEN]
    float* bo;  // [HIDDEN]
    // Candidate: Wxg, Whg, bg
    float* Wxg; // [VOCAB_SIZE x HIDDEN]
    float* Whg; // [HIDDEN x HIDDEN]
    float* bg;  // [HIDDEN]
    // Output: Why, by
    float* Why; // [HIDDEN x VOCAB_SIZE]
    float* by;  // [VOCAB_SIZE]
};

// Simple aligned allocation helpers (POSIX).
inline void* arc_aligned_malloc(size_t bytes, size_t align=64) {
    void* p = nullptr;
#if defined(_POSIX_VERSION)
    if (posix_memalign(&p, align, bytes) != 0) p = nullptr;
#else
    p = std::aligned_alloc(align, ((bytes + align - 1)/align)*align);
#endif
    if (!p) { std::fprintf(stderr, "OOM %zu bytes\n", bytes); std::abort(); }
    std::memset(p, 0, bytes);
    return p;
}
inline void arc_aligned_free(void* p){ std::free(p); }

#endif // ARC_TYPES_H

