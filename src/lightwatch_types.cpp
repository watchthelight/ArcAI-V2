#include "lightwatch_types.h"

// Runtime configuration variables (initialized by config system)
int VOCAB_SIZE = DEFAULT_VOCAB_SIZE;
int HIDDEN = DEFAULT_HIDDEN;
int BATCH = DEFAULT_BATCH;
int SEQ_LEN = DEFAULT_SEQ_LEN;
float LR = DEFAULT_LR;
int TBPTT_LEN = DEFAULT_TBPTT_LEN;

// Initialize runtime configuration
void initializeLightwatchConfig(int hidden_size, int tbptt_len) {
    VOCAB_SIZE = DEFAULT_VOCAB_SIZE;  // Keep vocab size constant
    HIDDEN = hidden_size;
    BATCH = DEFAULT_BATCH;            // Keep batch size constant
    SEQ_LEN = DEFAULT_SEQ_LEN;        // Keep sequence length constant
    LR = DEFAULT_LR;                  // Keep learning rate constant
    TBPTT_LEN = tbptt_len;
}
