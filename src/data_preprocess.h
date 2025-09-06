#ifndef DATA_PREPROCESS_H
#define DATA_PREPROCESS_H

#include <string>

// Main preprocessing function
void preprocess_data(const std::string& input_file,
                     const std::string& output_bin,
                     const std::string& vocab_json);

#endif // DATA_PREPROCESS_H
