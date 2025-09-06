#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "arc_types.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "arc_kernels.h"
#endif

int main() {
    std::cout << "=== ArcAI V2 CUDA Functionality Test ===" << std::endl;
    
    // Test 1: Check if CUDA is enabled at compile time
#ifdef USE_CUDA
    std::cout << "✓ CUDA support is ENABLED at compile time" << std::endl;
    
    // Test 2: Check CUDA device availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cout << "✗ No CUDA devices found. Error: " << cudaGetErrorString(err) << std::endl;
        std::cout << "  Falling back to CPU implementation" << std::endl;
    } else {
        std::cout << "✓ Found " << deviceCount << " CUDA device(s)" << std::endl;
        
        // Get device properties
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  Device " << i << ": " << prop.name 
                      << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
        }
    }
    
    // Test 3: Test basic CUDA memory allocation
    std::cout << "\n--- Testing CUDA Memory Operations ---" << std::endl;
    const int test_size = 1024;
    float* host_data = (float*)malloc(test_size * sizeof(float));
    float* device_data = nullptr;
    
    // Initialize host data
    for (int i = 0; i < test_size; i++) {
        host_data[i] = (float)i;
    }
    
    err = cudaMalloc(&device_data, test_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cout << "✗ CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "✓ CUDA malloc successful" << std::endl;
        
        // Test memory copy
        err = cudaMemcpy(device_data, host_data, test_size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cout << "✗ CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "✓ CUDA memcpy Host to Device successful" << std::endl;
            
            // Copy back
            float* result_data = (float*)malloc(test_size * sizeof(float));
            err = cudaMemcpy(result_data, device_data, test_size * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cout << "✗ CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "✓ CUDA memcpy Device to Host successful" << std::endl;
                
                // Verify data
                bool data_correct = true;
                for (int i = 0; i < test_size; i++) {
                    if (result_data[i] != host_data[i]) {
                        data_correct = false;
                        break;
                    }
                }
                
                if (data_correct) {
                    std::cout << "✓ Data integrity verified" << std::endl;
                } else {
                    std::cout << "✗ Data corruption detected" << std::endl;
                }
            }
            free(result_data);
        }
        
        cudaFree(device_data);
    }
    
    free(host_data);
    
    // Test 4: Test cuBLAS availability
    std::cout << "\n--- Testing cuBLAS ---" << std::endl;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "✗ cuBLAS initialization failed" << std::endl;
    } else {
        std::cout << "✓ cuBLAS initialized successfully" << std::endl;
        cublasDestroy(handle);
    }
    
    // Test 5: Test LSTM functions with CUDA
    std::cout << "\n--- Testing LSTM CUDA Functions ---" << std::endl;
    const int B = 2, H = 4, V = 8;
    
    // Allocate test data
    uint8_t* xt = (uint8_t*)malloc(B * sizeof(uint8_t));
    float* Hprev = (float*)arc_aligned_malloc(B * H * sizeof(float));
    float* Cprev = (float*)arc_aligned_malloc(B * H * sizeof(float));
    float* Hcur = (float*)arc_aligned_malloc(B * H * sizeof(float));
    float* Ccur = (float*)arc_aligned_malloc(B * H * sizeof(float));
    float* Z = (float*)arc_aligned_malloc(B * V * sizeof(float));
    
    // Initialize test data
    for (int i = 0; i < B; i++) xt[i] = i % V;
    memset(Hprev, 0, B * H * sizeof(float));
    memset(Cprev, 0, B * H * sizeof(float));
    
    // Create a minimal LSTM model for testing
    LSTM test_model;
    test_model.Wxi = (float*)arc_aligned_malloc(V * H * sizeof(float));
    test_model.Whi = (float*)arc_aligned_malloc(H * H * sizeof(float));
    test_model.bi = (float*)arc_aligned_malloc(H * sizeof(float));
    test_model.Wxf = (float*)arc_aligned_malloc(V * H * sizeof(float));
    test_model.Whf = (float*)arc_aligned_malloc(H * H * sizeof(float));
    test_model.bf = (float*)arc_aligned_malloc(H * sizeof(float));
    test_model.Wxo = (float*)arc_aligned_malloc(V * H * sizeof(float));
    test_model.Who = (float*)arc_aligned_malloc(H * H * sizeof(float));
    test_model.bo = (float*)arc_aligned_malloc(H * sizeof(float));
    test_model.Wxg = (float*)arc_aligned_malloc(V * H * sizeof(float));
    test_model.Whg = (float*)arc_aligned_malloc(H * H * sizeof(float));
    test_model.bg = (float*)arc_aligned_malloc(H * sizeof(float));
    test_model.Why = (float*)arc_aligned_malloc(H * V * sizeof(float));
    test_model.by = (float*)arc_aligned_malloc(V * sizeof(float));
    
    // Initialize weights with small random values
    srand(42);
    for (int i = 0; i < V * H; i++) {
        test_model.Wxi[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Wxf[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Wxo[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Wxg[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
    }
    for (int i = 0; i < H * H; i++) {
        test_model.Whi[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Whf[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Who[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
        test_model.Whg[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
    }
    for (int i = 0; i < H * V; i++) {
        test_model.Why[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.01f;
    }
    memset(test_model.bi, 0, H * sizeof(float));
    memset(test_model.bf, 0, H * sizeof(float));
    memset(test_model.bo, 0, H * sizeof(float));
    memset(test_model.bg, 0, H * sizeof(float));
    memset(test_model.by, 0, V * sizeof(float));
    
    try {
        lstm_forward(test_model, xt, Hprev, Cprev, Hcur, Ccur, Z, B);
        std::cout << "✓ LSTM forward pass completed successfully" << std::endl;
        
        // Check if output is reasonable (not all zeros or NaN)
        bool output_valid = false;
        for (int i = 0; i < B * V; i++) {
            if (Z[i] != 0.0f && !isnan(Z[i]) && !isinf(Z[i])) {
                output_valid = true;
                break;
            }
        }
        
        if (output_valid) {
            std::cout << "✓ LSTM output appears valid" << std::endl;
        } else {
            std::cout << "⚠ LSTM output may be invalid (all zeros or NaN)" << std::endl;
        }
        
    } catch (...) {
        std::cout << "✗ LSTM forward pass failed with exception" << std::endl;
    }
    
    // Cleanup
    free(xt);
    arc_aligned_free(Hprev);
    arc_aligned_free(Cprev);
    arc_aligned_free(Hcur);
    arc_aligned_free(Ccur);
    arc_aligned_free(Z);
    arc_aligned_free(test_model.Wxi);
    arc_aligned_free(test_model.Whi);
    arc_aligned_free(test_model.bi);
    arc_aligned_free(test_model.Wxf);
    arc_aligned_free(test_model.Whf);
    arc_aligned_free(test_model.bf);
    arc_aligned_free(test_model.Wxo);
    arc_aligned_free(test_model.Who);
    arc_aligned_free(test_model.bo);
    arc_aligned_free(test_model.Wxg);
    arc_aligned_free(test_model.Whg);
    arc_aligned_free(test_model.bg);
    arc_aligned_free(test_model.Why);
    arc_aligned_free(test_model.by);
    
#else
    std::cout << "✗ CUDA support is DISABLED at compile time" << std::endl;
    std::cout << "  The project was built without -DUSE_CUDA flag" << std::endl;
#endif
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
