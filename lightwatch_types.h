#ifndef LIGHTWATCH_TYPES_H
#define LIGHTWATCH_TYPES_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>

// Runtime configuration variables (initialized by config system)
extern int VOCAB_SIZE;
extern int HIDDEN;
extern int BATCH;
extern int SEQ_LEN;
extern float LR;
extern int TBPTT_LEN;

// Default values
#ifndef DEFAULT_VOCAB_SIZE
#define DEFAULT_VOCAB_SIZE 256
#endif
#ifndef DEFAULT_HIDDEN
#define DEFAULT_HIDDEN 256
#endif
#ifndef DEFAULT_BATCH
#define DEFAULT_BATCH 16
#endif
#ifndef DEFAULT_SEQ_LEN
#define DEFAULT_SEQ_LEN 64
#endif
#ifndef DEFAULT_LR
#define DEFAULT_LR 0.001f
#endif
#ifndef DEFAULT_TBPTT_LEN
#define DEFAULT_TBPTT_LEN 32
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

// Initialize runtime configuration
void initializeLightwatchConfig(int hidden_size, int tbptt_len);

// Simple aligned allocation helpers (POSIX).
inline void* lightwatch_aligned_malloc(size_t bytes, size_t align=64) {
    void* p = nullptr;
#if defined(_POSIX_VERSION)
    if (posix_memalign(&p, align, bytes) != 0) p = nullptr;
#elif defined(_MSC_VER)
    p = _aligned_malloc(bytes, align);
#else
    p = std::aligned_alloc(align, ((bytes + align - 1)/align)*align);
#endif
    if (!p) { std::fprintf(stderr, "OOM %zu bytes\n", bytes); std::abort(); }
    std::memset(p, 0, bytes);
    return p;
}

inline void lightwatch_aligned_free(void* p){
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

#endif // LIGHTWATCH_TYPES_H
