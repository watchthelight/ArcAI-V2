#ifndef LIGHTWATCH_CONFIG_H
#define LIGHTWATCH_CONFIG_H

struct LightwatchConfig {
    int hidden_size;
    int tbptt_len;
};

// Main configuration function
LightwatchConfig getLightwatchConfig(int argc, char** argv);

// Validation function
bool validateLightwatchConfig(const LightwatchConfig& config);

#endif // LIGHTWATCH_CONFIG_H
