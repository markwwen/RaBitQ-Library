cmake_minimum_required (VERSION 3.10)

project(RaBitQLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RABITQ_ENABLE_SVE "Enable Arm SVE/SVE2 kernels" ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Ofast -Wextra -lrt -fpic -fopenmp -ftree-vectorize -fexceptions")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(RABITQ_ENABLE_SVE)
        add_compile_definitions(RABITQ_USE_SVE)
        check_cxx_compiler_flag("-mcpu=native" RABITQ_HAS_MCPU_NATIVE)
        if(RABITQ_HAS_MCPU_NATIVE)
            add_compile_options(-mcpu=native)
        else()
            add_compile_options(-march=armv8.6-a+sve+sve2)
        endif()
    else()
        add_compile_options(-march=native)
    endif()
else()
    add_compile_options(-march=native)
endif()

add_subdirectory(sample/cpp)

option(RABITQ_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(RABITQ_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

add_library(rabitq_headers INTERFACE)

target_include_directories(rabitq_headers INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Testing
option(RABITQ_BUILD_TESTS "Build tests" OFF)

if(RABITQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()