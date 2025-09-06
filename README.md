# LightwatchAI

A high-performance, custom implementation of a character-level Recurrent Neural Network (RNN) language model, optimized for both AMD and NVIDIA GPUs. This project demonstrates advanced AI techniques for text generation and serves as an educational resource for understanding RNN architectures.

## Features

- **LSTM Architecture**: Long Short-Term Memory networks for better long-term dependency modeling
- **Interactive Model Configuration**: Terminal-based menu system for selecting model parameters
- **Multi-GPU Support**: 
  - **NVIDIA CUDA**: Optimized for NVIDIA GPUs using CUDA and cuBLAS
  - **AMD HIP**: Optimized for AMD GPUs using ROCm/HIP
- **Adam Optimizer**: Advanced optimization algorithm for faster convergence
- **Truncated BPTT**: Efficient training with configurable truncation length
- **Character-Level Generation**: Generate text at the byte level (256 possible tokens)
- **Checkpointing**: Save and resume training progress
- **OpenMP Parallelization**: CPU fallback with multi-threading support
- **BLAS Integration**: Optional OpenBLAS for accelerated matrix operations

## Interactive Configuration

LightwatchAI now features an interactive configuration system that allows you to select model parameters before training or running:

### Model Size Options
- **Minimum** → 64 hidden units
- **Mini** → 96 hidden units  
- **NotSoMini** → 128 hidden units (default)
- **Normaal** → 160 hidden units
- **Normalish** → 192 hidden units
- **Big** → 224 hidden units
- **Bigger** → 256 hidden units
- **Biggest** → 320 hidden units
- **LiterallyInsane** → 512 hidden units

### TBPTT Length Options
- **Shortest** → 1 step
- **Short** → 8 steps
- **NotShortButNotLong** → 16 steps
- **YesLong** → 32 steps (default)
- **Longer** → 64 steps
- **Longest** → 128 steps

## Architecture

- **Model Type**: LSTM with interactive configurable hidden size
- **Vocabulary**: 256 tokens (0-255 byte values)
- **Batch Size**: 16 sequences
- **Sequence Length**: 64 timesteps
- **TBPTT Length**: Interactive configurable (1-128 steps)
- **Learning Rate**: 0.001 (Adam)

## Requirements

- C++17 compatible compiler
- **For NVIDIA GPU support**: CUDA Toolkit 12.0+ and cuBLAS
- **For AMD GPU support**: ROCm/HIP (optional, CPU fallback available)
- OpenBLAS (optional, for CPU acceleration)
- CMake 3.16+ (for building)

## Building

1. Clone the repository:
   ```bash
   git clone https://github.com/watchthelight/LightwatchAI.git
   cd LightwatchAI
   ```

2. Build with CMake:

   **For CPU-only build:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

   **For NVIDIA CUDA support:**
   ```bash
   mkdir build_cuda
   cd build_cuda
   cmake .. -DUSE_CUDA=ON
   make
   ```

   **For AMD HIP support:**
   ```bash
   mkdir build_hip
   cd build_hip
   cmake .. -DUSE_HIP=ON
   make
   ```

3. **Testing CUDA Build:**
   ```bash
   # Test CUDA functionality
   ./simple_cuda_test
   ```

## Usage

### Interactive Training (New!)

Use the new interactive configuration system:

```bash
./lightwatch_train dataset.bin
```

This will launch an interactive menu where you can:
1. Select model size using arrow keys (↑/↓)
2. Select TBPTT length using arrow keys (↑/↓)
3. Press Enter to confirm selections

The model will train for 10,000 steps, saving checkpoints every 10 steps.

### Interactive Generation (New!)

Generate text with interactive configuration:

```bash
./lightwatch_run checkpoint_latest.bin --len 1000 --temp 0.8 --seed "Hello world"
```

### Command Line Override

You can override the interactive configuration with command line flags:

```bash
# Training with specific parameters
./lightwatch_train dataset.bin --hidden-size 256 --tbptt 64

# Generation with specific parameters  
./lightwatch_run checkpoint_latest.bin --hidden-size 256 --tbptt 64 --len 1000 --temp 0.8
```

### Legacy Usage

The original executables are still available for backward compatibility:

```bash
# Legacy training (uses compile-time constants)
./arc_train dataset.bin

# Legacy generation
./arc_run checkpoint_latest.bin --len 1000 --temp 0.8 --seed "Hello world"
```

### Configuration Options

**Interactive Configuration:**
- Use arrow keys (↑/↓) to navigate menu options
- Press Enter to select
- Press 'q' to quit
- Automatically falls back to defaults in non-interactive environments

**Command Line Flags:**
- `--hidden-size N`: Set hidden layer size (64-512)
- `--tbptt N`: Set TBPTT length (1-128)
- `--len N`: Generate N characters
- `--temp T`: Sampling temperature (0.1-2.0)
- `--topk K`: Top-K sampling (0 = no limit)
- `--seed STR`: Initial seed text

## Project Structure

### New LightwatchAI Files
- `lightwatch_types.h/cpp`: Runtime configurable model definitions
- `lightwatch_kernels.h`: Core computation kernels (forward/backward)
- `lightwatch_bptt.h`: Backpropagation through time implementation
- `lightwatch_config.h/cpp`: Interactive configuration system
- `lightwatch_train.cpp`: New training executable with interactive config
- `lightwatch_run.cpp`: New inference executable with interactive config
- `lightwatch_dataset.h`: Data loading utilities
- `lightwatch_generate.h`: Sampling utilities

### Legacy Files (Backward Compatibility)
- `arc_types.h`: Original model definitions and constants
- `arc_kernels.h`: Original computation kernels
- `arc_bptt.h`: Original BPTT implementation
- `arc_train.cpp`: Original training executable
- `arc_run.cpp`: Original inference executable
- `arc_dataset.h`: Original data loading utilities
- `arc_generate.h`: Original sampling utilities

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
