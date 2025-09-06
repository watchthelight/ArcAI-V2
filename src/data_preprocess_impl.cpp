#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

// Cross-platform JSON serialization (simple implementation)
std::string escape_json_string(const std::string& str) {
    std::string result;
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 32 || c > 126) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    result += buf;
                } else {
                    result += c;
                }
                break;
        }
    }
    return result;
}

void write_vocab_json(const std::string& vocab_json_path, const std::unordered_map<char, int>& char_to_id) {
    std::ofstream json_file(vocab_json_path);
    if (!json_file) {
        throw std::runtime_error("Failed to open vocab JSON file for writing: " + vocab_json_path);
    }

    json_file << "{\n";
    json_file << "  \"vocab_size\": " << char_to_id.size() << ",\n";
    json_file << "  \"char_to_id\": {\n";

    bool first = true;
    for (const auto& pair : char_to_id) {
        if (!first) json_file << ",\n";
        json_file << "    \"" << escape_json_string(std::string(1, pair.first)) << "\": " << pair.second;
        first = false;
    }
    json_file << "\n  }\n";
    json_file << "}\n";
}

void preprocess_data(const std::string& input_file,
                     const std::string& output_bin,
                     const std::string& vocab_json) {
    // Read the entire input file
    std::ifstream text_file(input_file, std::ios::binary);
    if (!text_file) {
        throw std::runtime_error("Failed to open input file: " + input_file);
    }

    std::string content((std::istreambuf_iterator<char>(text_file)),
                       std::istreambuf_iterator<char>());

    if (content.empty()) {
        throw std::runtime_error("Input file is empty: " + input_file);
    }

    std::cout << "Read " << content.size() << " characters from " << input_file << std::endl;

    // Build vocabulary (unique characters)
    std::unordered_set<char> unique_chars(content.begin(), content.end());
    std::vector<char> vocab(unique_chars.begin(), unique_chars.end());
    std::sort(vocab.begin(), vocab.end()); // Sort for consistent ordering

    // Create character to ID mapping
    std::unordered_map<char, int> char_to_id;
    for (size_t i = 0; i < vocab.size(); ++i) {
        char_to_id[vocab[i]] = static_cast<int>(i);
    }

    std::cout << "Vocabulary size: " << vocab.size() << " unique characters" << std::endl;

    // Convert text to token IDs
    std::vector<uint32_t> tokens;
    tokens.reserve(content.size());
    for (char c : content) {
        tokens.push_back(static_cast<uint32_t>(char_to_id[c]));
    }

    // Write binary file (32-bit little-endian)
    std::ofstream bin_file(output_bin, std::ios::binary);
    if (!bin_file) {
        throw std::runtime_error("Failed to open binary output file: " + output_bin);
    }

    // Write in little-endian format
    for (uint32_t token : tokens) {
        uint8_t bytes[4];
        bytes[0] = (token >> 0) & 0xFF;
        bytes[1] = (token >> 8) & 0xFF;
        bytes[2] = (token >> 16) & 0xFF;
        bytes[3] = (token >> 24) & 0xFF;
        bin_file.write(reinterpret_cast<char*>(bytes), 4);
    }

    bin_file.close();
    std::cout << "Wrote " << tokens.size() << " tokens to " << output_bin << std::endl;

    // Write vocabulary JSON
    write_vocab_json(vocab_json, char_to_id);
    std::cout << "Wrote vocabulary to " << vocab_json << std::endl;
}
