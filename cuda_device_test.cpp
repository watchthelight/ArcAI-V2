#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }

    return 0;
}
