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

# Find ncurses for interactive configuration (optional)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NCURSES ncurses)
endif()

# Source files
set(LIGHTWATCH_SOURCES
    lightwatch_types.cpp
    lightwatch_config.cpp
)

set(LIGHTWATCH_HEADERS
    lightwatch_types.h
    lightwatch_kernels.h
    lightwatch_bptt.h
    lightwatch_dataset.h
    lightwatch_generate.h
    lightwatch_config.h
)

# LightwatchAI executables
add_executable(lightwatch_train lightwatch_train.cpp ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})
add_executable(lightwatch_run lightwatch_run.cpp ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})

# Compiler flags
if(MSVC)
    # Remove /RTC1 option globally to avoid conflict with /O2
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    
    # LightwatchAI executables
    target_compile_options(lightwatch_train PRIVATE /W4 /O2)
    target_compile_options(lightwatch_run PRIVATE /W4 /O2)
    target_compile_options(lightwatch_train PRIVATE $<$<CONFIG:Release>:/RTC- >)
    target_compile_options(lightwatch_run PRIVATE $<$<CONFIG:Release>:/RTC- >)
else()
    # LightwatchAI executables
    target_compile_options(lightwatch_train PRIVATE -Wall -Wextra -O3 -fopenmp)
    target_compile_options(lightwatch_run PRIVATE -Wall -Wextra -O3 -fopenmp)
endif()

# HIP support
if(USE_HIP)
    target_compile_definitions(lightwatch_train PRIVATE USE_HIP)
    target_compile_definitions(lightwatch_run PRIVATE USE_HIP)
    target_link_libraries(lightwatch_train hip::device)
    target_link_libraries(lightwatch_run hip::device)
endif()

# CUDA support
if(USE_CUDA)
    target_compile_definitions(lightwatch_train PRIVATE USE_CUDA)
    target_compile_definitions(lightwatch_run PRIVATE USE_CUDA)
    target_link_libraries(lightwatch_train CUDA::cudart CUDA::cublas)
    target_link_libraries(lightwatch_run CUDA::cudart CUDA::cublas)
endif()

# OpenBLAS support
if(USE_OPENBLAS)
    target_compile_definitions(lightwatch_train PRIVATE USE_OPENBLAS)
    target_compile_definitions(lightwatch_run PRIVATE USE_OPENBLAS)
    target_link_libraries(lightwatch_train ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_link_libraries(lightwatch_run ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
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
install(TARGETS lightwatch_train lightwatch_run simple_cuda_test
    RUNTIME DESTINATION bin
)
