cmake_minimum_required(VERSION 3.16)
project(LightwatchAI VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_HIP "Enable HIP for AMD GPU support" OFF)
option(USE_CUDA "Enable CUDA for NVIDIA GPU support" OFF)
option(USE_OPENBLAS "Use OpenBLAS for matrix operations" OFF)
option(ENABLE_GUI "Enable GUI build with Qt" ON)

# Set global runtime output directory to build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

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

# Find Qt for GUI (optional)
if(ENABLE_GUI)
    find_package(Qt6 QUIET COMPONENTS Widgets)
    if(NOT Qt6_FOUND)
        find_package(Qt5 QUIET COMPONENTS Widgets)
        if(Qt5_FOUND)
            set(QT_VERSION 5)
        endif()
    else()
        set(QT_VERSION 6)
    endif()
    if(Qt6_FOUND OR Qt5_FOUND)
        set(QT_FOUND TRUE)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)
    endif()
endif()

# Source files
set(LIGHTWATCH_SOURCES
    src/lightwatch_types.cpp
    src/lightwatch_config.cpp
    src/data_preprocess_impl.cpp
)

set(LIGHTWATCH_HEADERS
    src/lightwatch_types.h
    src/lightwatch_kernels.h
    src/lightwatch_bptt.h
    src/lightwatch_dataset.h
    src/lightwatch_generate.h
    src/lightwatch_config.h
    src/data_preprocess.h
)

# LightwatchAI executables
add_executable(lightwatch_train src/lightwatch_train.cpp ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})
set_target_properties(lightwatch_train PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(lightwatch_run src/lightwatch_run.cpp ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})
set_target_properties(lightwatch_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler flags with platform-specific OpenMP detection
if(MSVC)
    # Remove /RTC1 option globally to avoid conflict with /O2
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

    # LightwatchAI executables
    target_compile_options(lightwatch_train PRIVATE /W4 /O2 /openmp)
    target_compile_options(lightwatch_run PRIVATE /W4 /O2 /openmp)
else()
    # Check for OpenMP support
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(OPENMP_FLAGS -fopenmp)
    else()
        set(OPENMP_FLAGS "")
    endif()

    # LightwatchAI executables
    target_compile_options(lightwatch_train PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
    target_compile_options(lightwatch_run PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
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

# Data preprocessing executable
add_executable(data_preprocess src/data_preprocess.cpp src/data_preprocess.h)
set_target_properties(data_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Apply same compiler flags as other executables
if(MSVC)
    target_compile_options(data_preprocess PRIVATE /W4 /O2 /openmp)
    target_compile_definitions(data_preprocess PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(data_preprocess PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
endif()

# Simple CUDA test executable
add_executable(simple_cuda_test src/simple_cuda_test.cpp)
set_target_properties(simple_cuda_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Apply same compiler flags as other executables
if(MSVC)
    target_compile_options(simple_cuda_test PRIVATE /W4 /O2 /openmp)
    target_compile_definitions(simple_cuda_test PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(simple_cuda_test PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
endif()

# CUDA support for simple test executable
if(USE_CUDA)
    target_compile_definitions(simple_cuda_test PRIVATE USE_CUDA)
endif()

# CLI executable (generation)
add_executable(lightwatch_cli src/lightwatch_run.cpp ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})
set_target_properties(lightwatch_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Apply same compiler flags as other executables
if(MSVC)
    target_compile_options(lightwatch_cli PRIVATE /W4 /O2 /openmp)
    target_compile_definitions(lightwatch_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(lightwatch_cli PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
endif()



# GUI executable
if(QT_FOUND)
    add_executable(lightwatch_gui src/lightwatch_gui.cpp src/gui_mainwindow.cpp src/gui_mainwindow.h ${LIGHTWATCH_SOURCES} ${LIGHTWATCH_HEADERS})
    set_target_properties(lightwatch_gui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_link_libraries(lightwatch_gui Qt${QT_VERSION}::Widgets)

    # Apply same compiler flags as other executables
    if(MSVC)
        target_compile_options(lightwatch_gui PRIVATE /W4 /O2 /openmp)
        target_compile_definitions(lightwatch_gui PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(lightwatch_gui PRIVATE -Wall -Wextra -O3 ${OPENMP_FLAGS})
    endif()



    # Bundle Qt dependencies for Windows standalone executable
    if(WIN32)
        find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS ${QT_BIN_DIR})
        if(WINDEPLOYQT_EXECUTABLE)
            add_custom_command(TARGET lightwatch_gui POST_BUILD
                COMMAND ${WINDEPLOYQT_EXECUTABLE} $<TARGET_FILE:lightwatch_gui>
                COMMENT "Bundling Qt runtime libraries"
            )
        endif()
    endif()
endif()



# Install
set(INSTALL_TARGETS lightwatch_train lightwatch_run data_preprocess simple_cuda_test)
if(QT_FOUND AND TARGET lightwatch_gui)
    list(APPEND INSTALL_TARGETS lightwatch_gui)
endif()
install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION bin
)
