cmake_minimum_required(VERSION 3.16)
project(ArcAI VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_HIP "Enable HIP for AMD GPU support" OFF)
option(USE_OPENBLAS "Use OpenBLAS for matrix operations" OFF)

# Find packages
if(USE_HIP)
    find_package(HIP REQUIRED)
    enable_language(HIP)
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
    target_compile_options(arc_train PRIVATE /W4 /O2)
    target_compile_options(arc_run PRIVATE /W4 /O2)
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

# OpenBLAS support
if(USE_OPENBLAS)
    target_compile_definitions(arc_train PRIVATE USE_OPENBLAS)
    target_compile_definitions(arc_run PRIVATE USE_OPENBLAS)
    target_link_libraries(arc_train ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_link_libraries(arc_run ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Install
install(TARGETS arc_train arc_run
    RUNTIME DESTINATION bin
)
