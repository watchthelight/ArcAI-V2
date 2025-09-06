#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include "lightwatch_types.h"
#include "lightwatch_dataset.h"
#include "lightwatch_bptt.h"
#include "lightwatch_config.h"
#include "data_preprocess.h"

#ifdef _WIN32
#define fopen_s(pFile,filename,mode) (((*(pFile))=fopen((filename),(mode)))==NULL)
#endif

// Adam optimizer parameters
const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;

// How often to save
#ifndef SAVE_INTERVAL
#define SAVE_INTERVAL 10
#endif

// --- helpers to save/load weights ---
void save_checkpoint(const LSTM& M, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen save"); std::exit(1); }
    auto write = [&](const void* p, size_t n){ if(std::fwrite(p,1,n,f)!=n){std::perror("fwrite");std::exit(1);} };
    write(M.Wxi, VOCAB_SIZE*HIDDEN*sizeof(float));
    write(M.Whi, HIDDEN*HIDDEN*sizeof(float));
    write(M.bi, HIDDEN*sizeof(float));
    write(M.Wxf, VOCAB_SIZE*HIDDEN*sizeof(float));
    write(M.Whf, HIDDEN*HIDDEN*sizeof(float));
    write(M.bf, HIDDEN*sizeof(float));
    write(M.Wxo, VOCAB_SIZE*HIDDEN*sizeof(float));
    write(M.Who, HIDDEN*HIDDEN*sizeof(float));
    write(M.bo, HIDDEN*sizeof(float));
    write(M.Wxg, VOCAB_SIZE*HIDDEN*sizeof(float));
    write(M.Whg, HIDDEN*HIDDEN*sizeof(float));
    write(M.bg, HIDDEN*sizeof(float));
    write(M.Why, HIDDEN*VOCAB_SIZE*sizeof(float));
    write(M.by, VOCAB_SIZE*sizeof(float));
    std::fclose(f);
}

bool load_checkpoint(LSTM& M, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto read = [&](void* p, size_t n){ if(std::fread(p,1,n,f)!=n){std::perror("fread");std::exit(1);} };
    read(M.Wxi, VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whi, HIDDEN*HIDDEN*sizeof(float));
    read(M.bi, HIDDEN*sizeof(float));
    read(M.Wxf, VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whf, HIDDEN*HIDDEN*sizeof(float));
    read(M.bf, HIDDEN*sizeof(float));
    read(M.Wxo, VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Who, HIDDEN*HIDDEN*sizeof(float));
    read(M.bo, HIDDEN*sizeof(float));
    read(M.Wxg, VOCAB_SIZE*HIDDEN*sizeof(float));
    read(M.Whg, HIDDEN*HIDDEN*sizeof(float));
    read(M.bg, HIDDEN*sizeof(float));
    read(M.Why, HIDDEN*VOCAB_SIZE*sizeof(float));
    read(M.by, VOCAB_SIZE*sizeof(float));
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    // Get configuration first
    LightwatchConfig config = getLightwatchConfig(argc, argv);
    if (!validateLightwatchConfig(config)) {
        return 1;
    }

    // Initialize runtime parameters
    initializeLightwatchConfig(config.hidden_size, config.tbptt_len);

    // Parse command line arguments
    std::string data_file = "data.txt";
    std::string dataset_path = "dataset.bin";
    bool force_regen = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (std::strcmp(argv[i], "--regen") == 0) {
            force_regen = true;
        } else if (std::strcmp(argv[i], "--hidden-size") == 0) {
            ++i; // skip value (already handled by config)
        } else if (std::strcmp(argv[i], "--tbptt") == 0) {
            ++i; // skip value (already handled by config)
        } else if (argv[i][0] == '-') {
            // Unknown flag
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            std::fprintf(stderr, "usage: %s [--data file.txt] [--regen] [--hidden-size N] [--tbptt N]\n", argv[0]);
            return 1;
        } else {
            // Positional argument - dataset path
            dataset_path = argv[i];
        }
    }

    // Check if data preprocessing is needed
    bool needs_preprocessing = force_regen ||
                              !std::filesystem::exists(dataset_path) ||
                              !std::filesystem::exists("vocab.json");

    if (needs_preprocessing) {
        if (!std::filesystem::exists(data_file)) {
            std::fprintf(stderr, "Error: Data file '%s' not found. Please provide a text file for training.\n", data_file.c_str());
            std::fprintf(stderr, "You can:\n");
            std::fprintf(stderr, "  1. Place a 'data.txt' file in the current directory\n");
            std::fprintf(stderr, "  2. Specify a custom file with --data <file>\n");
            return 1;
        }

        std::printf("Preprocessing data from '%s'...\n", data_file.c_str());
        try {
            preprocess_data(data_file, dataset_path, "vocab.json");
            std::printf("Data preprocessing completed.\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Error during data preprocessing: %s\n", e.what());
            return 1;
        }
    } else {
        std::printf("Using existing dataset '%s'\n", dataset_path.c_str());
    }

    // --- load dataset ---
    FILE* f = std::fopen(dataset_path.c_str(), "rb");
    if (!f) { std::perror("dataset fopen"); return 1; }
    std::fseek(f, 0, SEEK_END);
    size_t fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> tokens(fsize);
    if (std::fread(tokens.data(),1,fsize,f)!=fsize) {
        std::perror("dataset fread"); return 1;
    }
    std::fclose(f);

    std::printf("LightwatchAI Training: BATCH=%d SEQ_LEN=%d HIDDEN=%d VOCAB=%d TBPTT=%d steps=10000\n",
        BATCH, SEQ_LEN, HIDDEN, VOCAB_SIZE, TBPTT_LEN);

    // --- init model ---
    LSTM model;
    model.Wxi = (float*)std::malloc(VOCAB_SIZE*HIDDEN*sizeof(float));
    model.Whi = (float*)std::malloc(HIDDEN*HIDDEN*sizeof(float));
    model.bi  = (float*)std::malloc(HIDDEN*sizeof(float));
    model.Wxf = (float*)std::malloc(VOCAB_SIZE*HIDDEN*sizeof(float));
    model.Whf = (float*)std::malloc(HIDDEN*HIDDEN*sizeof(float));
    model.bf  = (float*)std::malloc(HIDDEN*sizeof(float));
    model.Wxo = (float*)std::malloc(VOCAB_SIZE*HIDDEN*sizeof(float));
    model.Who = (float*)std::malloc(HIDDEN*HIDDEN*sizeof(float));
    model.bo  = (float*)std::malloc(HIDDEN*sizeof(float));
    model.Wxg = (float*)std::malloc(VOCAB_SIZE*HIDDEN*sizeof(float));
    model.Whg = (float*)std::malloc(HIDDEN*HIDDEN*sizeof(float));
    model.bg  = (float*)std::malloc(HIDDEN*sizeof(float));
    model.Why = (float*)std::malloc(HIDDEN*VOCAB_SIZE*sizeof(float));
    model.by  = (float*)std::malloc(VOCAB_SIZE*sizeof(float));

    // try loading checkpoint
    int step_start = 0;
    if (load_checkpoint(model, "checkpoint_latest.bin")) {
        std::printf("Resumed from checkpoint_latest.bin\n");
        // infer step number if numbered checkpoint exists
        step_start = 0;
    } else {
        std::printf("Starting from scratch\n");
        std::srand(42);
        for (size_t i=0;i<VOCAB_SIZE*HIDDEN;i++) model.Wxi[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN*HIDDEN;i++)    model.Whi[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN;i++) model.bi[i]=0;
        for (size_t i=0;i<VOCAB_SIZE*HIDDEN;i++) model.Wxf[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN*HIDDEN;i++)    model.Whf[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN;i++) model.bf[i]=0;
        for (size_t i=0;i<VOCAB_SIZE*HIDDEN;i++) model.Wxo[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN*HIDDEN;i++)    model.Who[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN;i++) model.bo[i]=0;
        for (size_t i=0;i<VOCAB_SIZE*HIDDEN;i++) model.Wxg[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN*HIDDEN;i++)    model.Whg[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<HIDDEN;i++) model.bg[i]=0;
        for (size_t i=0;i<HIDDEN*VOCAB_SIZE;i++)model.Why[i] = ((std::rand()%2000)/1000.f-1.f)*0.01f;
        for (size_t i=0;i<VOCAB_SIZE;i++) model.by[i]=0;
    }

    std::vector<uint8_t> batch(BATCH*SEQ_LEN);

    for (int step = step_start; step < 10000; ++step) {
        // fill batch (looping)
        static size_t pos=0;
        for (size_t i=0;i<batch.size();++i) {
            batch[i] = tokens[pos++];
            if (pos >= tokens.size()) pos=0;
        }

        float loss = lstm_train_batch(model, batch.data());
        std::printf("[step %d] loss=%.4f\n", step, loss);

        if (step % SAVE_INTERVAL == 0) {
            // numbered save
            std::ostringstream oss;
            oss << "checkpoint_step" << step << ".bin";
            save_checkpoint(model, oss.str());

            // update latest
            save_checkpoint(model, "checkpoint_latest.bin");

            std::printf("Saved checkpoint at step %d\n", step);
        }
    }

    return 0;
}
