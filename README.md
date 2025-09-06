# LightwatchAI

A high-performance, custom implementation of a character-level LSTM language model with an interactive configuration system. Features cross-platform support for Windows, Linux, and macOS with GPU acceleration for both NVIDIA and AMD hardware.

## Features

- **🎯 Interactive Model Configuration**: Terminal-based menu system with arrow key navigation
- **🧠 LSTM Architecture**: Long Short-Term Memory networks for superior text generation
- **⚡ Multi-GPU Support**: 
  - **NVIDIA CUDA**: Optimized for NVIDIA GPUs using CUDA and cuBLAS
  - **AMD HIP**: Optimized for AMD GPUs using ROCm/HIP
- **🔧 Runtime Configuration**: Choose model size and sequence length interactively
- **💾 Smart Checkpointing**: Save and resume training progress automatically
- **🚀 Adam Optimizer**: Advanced optimization for faster convergence
- **🔄 Cross-Platform**: Windows, Linux, macOS with shell-specific instructions

## Interactive Configuration

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

## Requirements

- **C++17** compatible compiler
- **CMake 3.16+** for building
- **Optional**: CUDA Toolkit 12.0+ (NVIDIA GPU support)
- **Optional**: ROCm/HIP (AMD GPU support)
- **Optional**: OpenBLAS (CPU acceleration)

## Platform-Specific Setup

### Windows (PowerShell/Command Prompt)

**Prerequisites:**
- Visual Studio 2019/2022 with C++ tools
- CMake (install via Visual Studio Installer or download from cmake.org)
- Git for Windows

**Clone and Build:**
```powershell
# Clone repository
git clone https://github.com/watchthelight/LightwatchAI.git
cd LightwatchAI

# CPU-only build
mkdir build_cpu
cd build_cpu
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ..

# NVIDIA CUDA build (if CUDA is installed)
mkdir build_cuda
cd build_cuda
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
cmake --build . --config Release
cd ..
```

**Run Executables:**
```powershell
# Interactive training
.\Release\lightwatch_train.exe dataset.bin

# Interactive generation
.\Release\lightwatch_run.exe checkpoint_latest.bin

# With command line overrides
.\Release\lightwatch_train.exe dataset.bin --hidden-size 256 --tbptt 64
```

### Linux (Bash Shell)

**Prerequisites (Ubuntu/Debian):**
```bash
# Install build tools
sudo apt update
sudo apt install build-essential cmake git

# Optional: CUDA support (NVIDIA)
# Download and install CUDA Toolkit from developer.nvidia.com

# Optional: ROCm support (AMD)
# Follow ROCm installation guide for your distribution

# Optional: OpenBLAS
sudo apt install libopenblas-dev
```

**Clone and Build:**
```bash
# Clone repository
git clone https://github.com/watchthelight/LightwatchAI.git
cd LightwatchAI

# CPU-only build
mkdir build_cpu && cd build_cpu
cmake ..
make -j$(nproc)
cd ..

# NVIDIA CUDA build
mkdir build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON
make -j$(nproc)
cd ..

# AMD HIP build
mkdir build_hip && cd build_hip
cmake .. -DUSE_HIP=ON
make -j$(nproc)
cd ..

# OpenBLAS build
mkdir build_blas && cd build_blas
cmake .. -DUSE_OPENBLAS=ON
make -j$(nproc)
cd ..
```

**Run Executables:**
```bash
# Interactive training
./lightwatch_train dataset.bin

# Interactive generation
./lightwatch_run checkpoint_latest.bin

# With command line overrides
./lightwatch_train dataset.bin --hidden-size 256 --tbptt 64
```

### macOS (Bash/Zsh Shell)

**Prerequisites:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake
brew install cmake

# Optional: OpenBLAS
brew install openblas
```

**Clone and Build:**
```bash
# Clone repository
git clone https://github.com/watchthelight/LightwatchAI.git
cd LightwatchAI

# CPU-only build
mkdir build_cpu && cd build_cpu
cmake ..
make -j$(sysctl -n hw.ncpu)
cd ..

# OpenBLAS build (recommended for macOS)
mkdir build_blas && cd build_blas
cmake .. -DUSE_OPENBLAS=ON
make -j$(sysctl -n hw.ncpu)
cd ..
```

**Run Executables:**
```bash
# Interactive training
./lightwatch_train dataset.bin

# Interactive generation
./lightwatch_run checkpoint_latest.bin
```

### Fish Shell (Linux/macOS)

**Build Commands:**
```fish
# Clone repository
git clone https://github.com/watchthelight/LightwatchAI.git
cd LightwatchAI

# CPU build
mkdir build_cpu; and cd build_cpu
cmake ..
make -j(nproc)  # Linux
# make -j(sysctl -n hw.ncpu)  # macOS
cd ..

# CUDA build (Linux only)
mkdir build_cuda; and cd build_cuda
cmake .. -DUSE_CUDA=ON
make -j(nproc)
cd ..
```

**Run Executables:**
```fish
# Interactive training
./lightwatch_train dataset.bin

# Interactive generation  
./lightwatch_run checkpoint_latest.bin

# Command line override
./lightwatch_train dataset.bin --hidden-size 256 --tbptt 64
```

## Usage Guide

### Interactive Mode

When you run the executables without command line overrides, you'll see an interactive menu:

```
=== LightwatchAI Configuration ===
Select Model Size (Hidden Layer Size):

Use ↑/↓ arrow keys to navigate, Enter to select:

  Minimum (64)
  Mini (96)
→ NotSoMini (128)
  Normaal (160)
  Normalish (192)
  Big (224)
  Bigger (256)
  Biggest (320)
  LiterallyInsane (512)

Press 'q' to quit
```

**Navigation:**
- **↑/↓ Arrow Keys**: Navigate menu options
- **Enter**: Select current option
- **q**: Quit application

### Command Line Override

Skip the interactive menu by providing parameters directly:

```bash
# Training with specific configuration
./lightwatch_train dataset.bin --hidden-size 512 --tbptt 128

# Generation with specific configuration
./lightwatch_run checkpoint_latest.bin --hidden-size 512 --tbptt 128 --len 2000 --temp 0.9 --seed "Once upon a time"
```

### Non-Interactive Environments

The system automatically detects non-interactive environments (pipes, scripts, CI/CD) and uses default values:
- **Hidden Size**: 128
- **TBPTT Length**: 32

### Generation Options

```bash
./lightwatch_run checkpoint.bin [OPTIONS]

Options:
  --hidden-size N    Model hidden size (must match training)
  --tbptt N         TBPTT length (must match training)  
  --len N           Generate N characters (default: 100)
  --temp T          Sampling temperature 0.1-2.0 (default: 1.0)
  --topk K          Top-K sampling, 0=disabled (default: 50)
  --seed "text"     Initial seed text (default: empty)
```

## Preparing Training Data

Convert your text file to binary format:

```bash
# Create a simple Python script to convert text to binary
python3 -c "
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    text = f.read()
with open(sys.argv[2], 'wb') as f:
    f.write(text.encode('utf-8'))
" input.txt dataset.bin
```

## Troubleshooting

### Windows Issues
- **"MSVCR120.dll missing"**: Install Visual C++ Redistributable
- **CMake not found**: Add CMake to PATH or reinstall with "Add to PATH" option
- **CUDA not detected**: Ensure CUDA_PATH environment variable is set

### Linux Issues
- **"Permission denied"**: Make executables executable with `chmod +x lightwatch_*`
- **CUDA issues**: Check `nvidia-smi` and ensure CUDA_HOME is set
- **Missing libraries**: Install development packages (`-dev` suffix)

### macOS Issues
- **"Developer cannot be verified"**: Run `sudo spctl --master-disable` temporarily
- **OpenMP not found**: Install with `brew install libomp`

### General Issues
- **Interactive menu not working**: Ensure terminal supports ANSI escape sequences
- **Arrow keys not working**: Try using a different terminal emulator
- **Build fails**: Check CMake version is 3.16+ and compiler supports C++17

## Project Structure

```
LightwatchAI/
├── lightwatch_config.cpp/h     # Interactive configuration system
├── lightwatch_types.cpp/h      # Runtime configurable parameters  
├── lightwatch_train.cpp        # Training executable
├── lightwatch_run.cpp          # Generation executable
├── lightwatch_kernels.h        # LSTM computation kernels
├── lightwatch_bptt.h           # Backpropagation implementation
├── lightwatch_dataset.h        # Data loading utilities
├── lightwatch_generate.h       # Text generation utilities
├── simple_cuda_test.cpp        # CUDA functionality test
└── CMakeLists.txt              # Build configuration
```

## Performance Notes

- **CPU Performance**: Use OpenBLAS for significant speedup on CPU
- **GPU Memory**: Larger models (Biggest, LiterallyInsane) require more VRAM
- **Training Speed**: CUDA > HIP > OpenBLAS > Plain CPU
- **Model Quality**: Larger hidden sizes generally produce better text

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test on multiple platforms
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Andrej Karpathy's char-rnn
- Built with modern C++17 and cross-platform compatibility in mind
- Community contributions for platform-specific optimizations
