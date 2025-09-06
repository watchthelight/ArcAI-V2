cmake_minimum_required(VERSION 3.16)
project(LightwatchAI VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_HIP "Enable HIP for AMD GPU support" OFF)
option(USE_CUDA "Enable CUDA for NVIDIA GPU support" OFF)
option(USE_OPENBLAS "Use OpenBLAS for matrix operations" OFF)

# Find packages
if(USE_HIP)
    find_package(HIP REQUIRED)
    enable_language(HIP)
endif()

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(USE_OPENBLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# Source files
set(SOURCES
    arc_train.cpp
    arc_run.cpp
)

set(HEADERS
    arc_types.h
    arc_kernels.h
    arc_bptt.h
    arc_dataset.h
    arc_generate.h
)

# Executables
add_executable(arc_train arc_train.cpp ${HEADERS})
add_executable(arc_run arc_run.cpp ${HEADERS})

# Compiler flags
if(MSVC)
    # Remove /RTC1 option globally to avoid conflict with /O2
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    target_compile_options(arc_train PRIVATE /W4 /O2)
    target_compile_options(arc_run PRIVATE /W4 /O2)
    # Disable runtime checks for optimized builds to avoid /RTC1 incompatibility
    target_compile_options(arc_train PRIVATE $<$<CONFIG:Release>:/RTC- >)
    target_compile_options(arc_run PRIVATE $<$<CONFIG:Release>:/RTC- >)
else()
    target_compile_options(arc_train PRIVATE -Wall -Wextra -O3 -fopenmp)
    target_compile_options(arc_run PRIVATE -Wall -Wextra -O3 -fopenmp)
endif()

# HIP support
if(USE_HIP)
    target_compile_definitions(arc_train PRIVATE USE_HIP)
    target_compile_definitions(arc_run PRIVATE USE_HIP)
    target_link_libraries(arc_train hip::device)
    target_link_libraries(arc_run hip::device)
endif()

# CUDA support
if(USE_CUDA)
    target_compile_definitions(arc_train PRIVATE USE_CUDA)
    target_compile_definitions(arc_run PRIVATE USE_CUDA)
    target_link_libraries(arc_train CUDA::cudart CUDA::cublas)
    target_link_libraries(arc_run CUDA::cudart CUDA::cublas)
endif()

# OpenBLAS support
if(USE_OPENBLAS)
    target_compile_definitions(arc_train PRIVATE USE_OPENBLAS)
    target_compile_definitions(arc_run PRIVATE USE_OPENBLAS)
    target_link_libraries(arc_train ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_link_libraries(arc_run ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Simple CUDA test executable
add_executable(simple_cuda_test simple_cuda_test.cpp)

# Apply same compiler flags as other executables
if(MSVC)
    target_compile_options(simple_cuda_test PRIVATE /W4 /O2)
    target_compile_options(simple_cuda_test PRIVATE $<$<CONFIG:Release>:/RTC- >)
    target_compile_definitions(simple_cuda_test PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(simple_cuda_test PRIVATE -Wall -Wextra -O3)
endif()

# CUDA support for simple test executable
if(USE_CUDA)
    target_compile_definitions(simple_cuda_test PRIVATE USE_CUDA)
endif()

# Install
install(TARGETS arc_train arc_run simple_cuda_test
    RUNTIME DESTINATION bin
)
