#ifndef LIGHTWATCH_DATASET_H
#define LIGHTWATCH_DATASET_H

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "lightwatch_types.h"

struct Dataset {
    std::vector<uint8_t> tokens;
    size_t cursor = 0;
};

inline bool dataset_load(const char* path, Dataset& ds) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("fopen"); return false; }
    ds.tokens.clear();
    uint8_t buf[1<<15];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        ds.tokens.insert(ds.tokens.end(), buf, buf+n);
    }
    std::fclose(f);
    if (ds.tokens.empty()) {
        std::fprintf(stderr, "dataset_load: empty file\n");
        return false;
    }
    return true;
}

// Fill exactly BATCH*SEQ_LEN bytes, loop dataset as needed.
inline void dataset_next_batch(Dataset& ds, uint8_t* out) {
    const size_t need = (size_t)BATCH * SEQ_LEN;
    const size_t N = ds.tokens.size();
    if (N == 0) { std::fprintf(stderr,"dataset empty\n"); std::abort(); }
    size_t c = ds.cursor;
    for (size_t i=0;i<need;i++) {
        out[i] = ds.tokens[c];
        c++; if (c >= N) c = 0;
    }
    ds.cursor = c;
}

#endif // LIGHTWATCH_DATASET_H
