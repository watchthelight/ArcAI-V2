#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        std::cout << "CUDA device not found. Running on CPU." << std::endl;
        return 1;
    } else {
        std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device " << i << ": " << prop.name << std::endl;
        }
        return 0;
    }
}
