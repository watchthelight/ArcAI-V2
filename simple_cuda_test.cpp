#include <iostream>
#include <cstdio>

int main() {
    std::cout << "=== Simple CUDA Build Test ===" << std::endl;
    
#ifdef USE_CUDA
    std::cout << "✓ CUDA support is ENABLED at compile time" << std::endl;
    std::cout << "✓ Project successfully built with CUDA support" << std::endl;
#else
    std::cout << "✗ CUDA support is DISABLED at compile time" << std::endl;
#endif
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}
