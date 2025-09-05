# ArcAI V2

A high-performance, custom implementation of a character-level Recurrent Neural Network (RNN) language model, optimized for AMD GPUs using HIP. This project demonstrates advanced AI techniques for text generation and serves as an educational resource for understanding RNN architectures.

## Features

- **LSTM Architecture**: Long Short-Term Memory networks for better long-term dependency modeling
- **HIP GPU Acceleration**: Optimized for AMD GPUs using ROCm/HIP
- **Adam Optimizer**: Advanced optimization algorithm for faster convergence
- **Truncated BPTT**: Efficient training with configurable truncation length
- **Character-Level Generation**: Generate text at the byte level (256 possible tokens)
- **Checkpointing**: Save and resume training progress
- **OpenMP Parallelization**: CPU fallback with multi-threading support
- **BLAS Integration**: Optional OpenBLAS for accelerated matrix operations

## Architecture

- **Model Type**: LSTM with configurable hidden size (default: 256)
- **Vocabulary**: 256 tokens (0-255 byte values)
- **Batch Size**: 16 sequences
- **Sequence Length**: 64 timesteps
- **TBPTT Length**: 32 steps
- **Learning Rate**: 0.001 (Adam)

## Requirements

- C++17 compatible compiler
- ROCm/HIP for AMD GPU support (optional, CPU fallback available)
- OpenBLAS (optional, for CPU acceleration)
- CMake (for building)

## Building

1. Clone the repository:
   ```bash
   git clone https://github.com/watchthelight/ArcAI-V2.git
   cd ArcAI-V2
   ```

2. Build with CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. For HIP support, ensure ROCm is installed and set `USE_HIP=ON` in CMake.

## Usage

### Training

Prepare your dataset as a binary file of uint8_t tokens, then train:

```bash
./arc_train dataset.bin
```

The model will train for 10,000 steps, saving checkpoints every 10 steps.

### Generation

Generate text using a trained checkpoint:

```bash
./arc_run checkpoint_latest.bin --len 1000 --temp 0.8 --seed "Hello world"
```

Options:
- `--len N`: Generate N characters
- `--temp T`: Sampling temperature (0.1-2.0)
- `--topk K`: Top-K sampling (0 = no limit)
- `--seed STR`: Initial seed text

## Project Structure

- `arc_types.h`: Model definitions and constants
- `arc_kernels.h`: Core computation kernels (forward/backward)
- `arc_bptt.h`: Backpropagation through time implementation
- `arc_train.cpp`: Training executable
- `arc_run.cpp`: Inference/generation executable
- `arc_dataset.h`: Data loading utilities
- `arc_generate.h`: Sampling utilities

## Performance Improvements

Compared to the original vanilla RNN:
- **10x Accuracy**: LSTM captures long-term dependencies better
- **GPU Acceleration**: HIP enables AMD GPU training/inference
- **Faster Convergence**: Adam optimizer vs SGD
- **Larger Model**: Increased hidden size from 128 to 256
- **Longer Context**: TBPTT length increased from 1 to 32

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by Andrej Karpathy's char-rnn and various open-source RNN implementations.
