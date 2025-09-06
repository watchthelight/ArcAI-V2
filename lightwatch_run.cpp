#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "lightwatch_types.h"
#include "lightwatch_kernels.h"
#include "lightwatch_generate.h"
#include "lightwatch_config.h"

// ----------------- CLI -----------------
struct Args {
    std::string checkpoint;
    int   len  = 100;
    float temp = 1.0f;
    int   topk = 50;      // 0 or <=0 = no top-k (use full vocab)
    std::string seed = "";
};

static uint64_t hash64(const std::string& s){
    // FNV-1a 64-bit
    uint64_t h = 1469598103934665603ull;
    for(unsigned char c: s){ h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}

static Args parse_args(int argc, char** argv){
    Args a;
    std::vector<std::string> remaining_args;
    
    // Skip config flags
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hidden-size") == 0) {
            ++i; // skip value
        } else if (std::strcmp(argv[i], "--tbptt") == 0) {
            ++i; // skip value
        } else {
            remaining_args.push_back(argv[i]);
        }
    }
    
    if (remaining_args.empty()){
        std::fprintf(stderr, "usage: %s [--hidden-size N] [--tbptt N] checkpoint.bin [--len N] [--temp T] [--topk K] [--seed STR]\n", argv[0]);
        std::exit(1);
    }
    
    a.checkpoint = remaining_args[0];
    for (size_t i = 1; i < remaining_args.size(); ++i){
        if      (remaining_args[i] == "--len"  && i+1 < remaining_args.size()) a.len  = std::stoi(remaining_args[++i]);
        else if (remaining_args[i] == "--temp" && i+1 < remaining_args.size()) a.temp = std::stof(remaining_args[++i]);
        else if (remaining_args[i] == "--topk" && i+1 < remaining_args.size()) a.topk = std::stoi(remaining_args[++i]);
        else if (remaining_args[i] == "--seed" && i+1 < remaining_args.size()) a.seed = remaining_args[++i];
    }
    return a;
}

// ------------- Model buffers -------------
static void alloc_model(LSTM& M){
    M.Wxi = (float*)std::malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whi = (float*)std::malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bi  = (float*)std::malloc((size_t)HIDDEN*sizeof(float));
    M.Wxf = (float*)std::malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whf = (float*)std::malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bf  = (float*)std::malloc((size_t)HIDDEN*sizeof(float));
    M.Wxo = (float*)std::malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Who = (float*)std::malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bo  = (float*)std::malloc((size_t)HIDDEN*sizeof(float));
    M.Wxg = (float*)std::malloc((size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    M.Whg = (float*)std::malloc((size_t)HIDDEN*HIDDEN*sizeof(float));
    M.bg  = (float*)std::malloc((size_t)HIDDEN*sizeof(float));
    M.Why = (float*)std::malloc((size_t)HIDDEN*VOCAB_SIZE*sizeof(float));
    M.by  = (float*)std::malloc((size_t)VOCAB_SIZE*sizeof(float));
    if(!M.Wxi||!M.Whi||!M.bi||!M.Wxf||!M.Whf||!M.bf||!M.Wxo||!M.Who||!M.bo||!M.Wxg||!M.Whg||!M.bg||!M.Why||!M.by){
        std::fprintf(stderr, "OOM allocating model buffers\n");
        std::exit(1);
    }
}

static void free_model(LSTM& M){
    std::free(M.Wxi); std::free(M.Whi); std::free(M.bi);
    std::free(M.Wxf); std::free(M.Whf); std::free(M.bf);
    std::free(M.Wxo); std::free(M.Who); std::free(M.bo);
    std::free(M.Wxg); std::free(M.Whg); std::free(M.bg);
    std::free(M.Why); std::free(M.by);
}

// Same binary format as in lightwatch_train.cpp
static bool load_checkpoint(LSTM& M, const std::string& path){
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::perror("fopen load"); return false; }
    auto read = [&](void* p, size_t n){
        if(std::fread(p,1,n,f)!=n){ std::perror("fread"); std::exit(1); }
    };
    read(M.Wxi, (size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whi, (size_t)HIDDEN*HIDDEN*sizeof(float));
    read(M.bi,  (size_t)HIDDEN*sizeof(float));
    read(M.Wxf, (size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whf, (size_t)HIDDEN*HIDDEN*sizeof(float));
    read(M.bf,  (size_t)HIDDEN*sizeof(float));
    read(M.Wxo, (size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Who, (size_t)HIDDEN*HIDDEN*sizeof(float));
    read(M.bo,  (size_t)HIDDEN*sizeof(float));
    read(M.Wxg, (size_t)VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whg, (size_t)HIDDEN*HIDDEN*sizeof(float));
    read(M.bg,  (size_t)HIDDEN*sizeof(float));
    read(M.Why, (size_t)HIDDEN*VOCAB_SIZE*sizeof(float));
    read(M.by,  (size_t)VOCAB_SIZE*sizeof(float));
    std::fclose(f);
    return true;
}

// ------------- Sampling -------------
static int sample_from_logits(const float* Z, int V, float temp, int topk, std::mt19937& rng){
    temp = std::max(0.01f, temp);
    int K = (topk<=0 || topk>V) ? V : topk;

    // Top-K indices by logit
    std::vector<int> idx(V);
    for(int i=0;i<V;i++) idx[i]=i;
    if (K < V){
        std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
                          [&](int a, int b){ return Z[a] > Z[b]; });
        idx.resize(K);
    }

    // Softmax over selected indices with temperature
    float m = Z[idx[0]];
    for(int j=1;j<K;j++) if (Z[idx[j]] > m) m = Z[idx[j]];
    std::vector<double> p(K);
    double sum = 0.0;
    for(int j=0;j<K;j++){ double e = std::exp((Z[idx[j]] - m)/temp); p[j]=e; sum+=e; }
    for(int j=0;j<K;j++) p[j] /= sum;

    std::discrete_distribution<int> dist(p.begin(), p.end());
    return idx[dist(rng)];
}

// ------------- Main -------------
int main(int argc, char** argv){
    // Get configuration first
    LightwatchConfig config = getLightwatchConfig(argc, argv);
    if (!validateLightwatchConfig(config)) {
        return 1;
    }

    // Initialize runtime parameters
    initializeLightwatchConfig(config.hidden_size, config.tbptt_len);

    const Args args = parse_args(argc, argv);

    LSTM M;
    alloc_model(M);
    if (!load_checkpoint(M, args.checkpoint)){
        std::fprintf(stderr, "Failed to load checkpoint: %s\n", args.checkpoint.c_str());
        free_model(M);
        return 1;
    }
    std::printf("Loaded checkpoint: %s (HIDDEN=%d, VOCAB=%d)\n", args.checkpoint.c_str(), HIDDEN, VOCAB_SIZE);

    const int H = HIDDEN;
    const int V = VOCAB_SIZE;
    std::vector<float> Hprev(H,0.f), Cprev(H,0.f), Hcur(H,0.f), Ccur(H,0.f), Z(V,0.f);

    // Deterministic RNG from seed string
    std::mt19937 rng((uint32_t)(hash64(args.seed) ^ 0x9e3779b97f4a7c15ull));

    // Prime with seed text (if any) and echo it
    uint8_t tok = (uint8_t)' ';
    if (!args.seed.empty()){
        for(char c: args.seed){
            tok = (uint8_t)(unsigned char)c;
            lstm_forward(M, &tok, Hprev.data(), Cprev.data(), Hcur.data(), Ccur.data(), Z.data(), 1);
            std::memcpy(Hprev.data(), Hcur.data(), (size_t)H*sizeof(float));
            std::memcpy(Cprev.data(), Ccur.data(), (size_t)H*sizeof(float));
        }
        std::fwrite(args.seed.data(), 1, args.seed.size(), stdout);
        std::fflush(stdout);
    } else {
        // one warmup step from space
        tok = (uint8_t)' ';
        lstm_forward(M, &tok, Hprev.data(), Cprev.data(), Hcur.data(), Ccur.data(), Z.data(), 1);
        std::memcpy(Hprev.data(), Hcur.data(), (size_t)H*sizeof(float));
        std::memcpy(Cprev.data(), Ccur.data(), (size_t)H*sizeof(float));
    }

    // Generate
    for (int t=0; t<args.len; t++){
        lstm_forward(M, &tok, Hprev.data(), Cprev.data(), Hcur.data(), Ccur.data(), Z.data(), 1);
        int next = sample_from_logits(Z.data(), V, args.temp, args.topk, rng);
        std::putchar((char)next);
        std::fflush(stdout);
        tok = (uint8_t)next;
        std::memcpy(Hprev.data(), Hcur.data(), (size_t)H*sizeof(float));
        std::memcpy(Cprev.data(), Ccur.data(), (size_t)H*sizeof(float));
    }
    std::putchar('\n');

    free_model(M);
    return 0;
}
