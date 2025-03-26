cmake_minimum_required(VERSION 3.31)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json" FORCE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(RNS-Tests LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE GPU_SOURCES "src/*.cu")
add_library(matrix_cuda STATIC ${GPU_SOURCES})
set_target_properties(matrix_cuda PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(matrix_cuda PRIVATE ${CUDA_LIBRARIES})
target_include_directories(matrix_cuda PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
)

file(GLOB_RECURSE MAIN_SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE matrix_cuda)

include(CTest)
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(googlebenchmark)

# Tests
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
  add_executable(tests ${TEST_SOURCES})
  set_target_properties(tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  target_link_libraries(tests PRIVATE gtest gtest_main matrix_cuda ${CUDA_LIBRARIES})
  include(GoogleTest)
  gtest_discover_tests(tests)
endif()

# Benchmarks
file(GLOB_RECURSE BENCHMARK_SOURCES "benchmarks/*.cpp")
if(BENCHMARK_SOURCES)
  add_executable(matrix_benchmark ${BENCHMARK_SOURCES})
  set_target_properties(matrix_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  target_link_libraries(matrix_benchmark PRIVATE benchmark::benchmark matrix_cuda ${CUDA_LIBRARIES})
  add_test(NAME matrix_benchmark COMMAND matrix_benchmark)
endif()
