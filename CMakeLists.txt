cmake_minimum_required(VERSION 3.25)
project(itch_aggregator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 4080 (Ada). 89-real emits SASS, 89-virtual embeds PTX for forward compat.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ZLIB REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
# -lineinfo keeps source correlation for ncu/nsys without disabling optimization.
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")

# CPU-side core: parser, order hash, batch assembler. Deliberately CUDA-free so
# the parser and --cpu-only paths build and test without a GPU.
add_library(itch_core STATIC
  src/parser/itch_parser.cpp
  src/pipeline/batch_assembler.cpp
  src/pipeline/cpu_reference.cpp
)
target_include_directories(itch_core PUBLIC src)
target_link_libraries(itch_core PUBLIC ZLIB::ZLIB)

add_library(itch_kernels STATIC
  src/cuda/decode_kernel.cu
  src/cuda/aggregate_kernel.cu
  src/cuda/snapshot_kernel.cu
)
target_include_directories(itch_kernels PUBLIC src)
target_link_libraries(itch_kernels PUBLIC CUDA::cudart)

add_executable(itch_pipeline src/main.cpp)
target_link_libraries(itch_pipeline PRIVATE itch_core itch_kernels)

enable_testing()
add_executable(itch_tests
  tests/test_main.cpp
  tests/test_parser.cpp
  tests/test_order_hash.cpp
  tests/test_assembler.cpp
  tests/test_kernels.cpp
)
target_include_directories(itch_tests PRIVATE third_party)
target_link_libraries(itch_tests PRIVATE itch_core itch_kernels)
add_test(NAME unit COMMAND itch_tests)

add_executable(parser_bench benchmarks/parser_bench.cpp)
target_include_directories(parser_bench PRIVATE tests)  # itch_builder.hpp
target_link_libraries(parser_bench PRIVATE itch_core)

add_executable(kernel_bench benchmarks/kernel_bench.cu)
target_include_directories(kernel_bench PRIVATE tests)
target_link_libraries(kernel_bench PRIVATE itch_core itch_kernels)
