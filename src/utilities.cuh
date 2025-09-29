#pragma once

#include <cuda_runtime.h>
#include <thrust/random.h>

#include <filesystem>
#include <format>
#include <iostream>
#include <source_location>
#include <string>

#define EPSILON 0.00001f

constexpr bool CHECK_ERRORS = true;

inline void check_cuda_error(const char* message,
                             std::source_location loc = std::source_location::current()) {
  if constexpr (CHECK_ERRORS) {
    cudaError_t error = cudaDeviceSynchronize();

    if (cudaSuccess == error) {
      return;
    }

    std::string file_name = std::filesystem::path(loc.file_name()).filename().string();
    std::cerr << std::format("[CUDA] Error at {}({}): {}: {}", file_name, loc.line(), message,
                             cudaGetErrorString(error))
              << std::endl;
  }
}

/// Helper for usage in `cuda::std::visit`. Taken from
/// https://en.cppreference.com/w/cpp/utility/variant/visit2.html#Example
template <typename... Ts>
struct Match : Ts... {
  using Ts::operator()...;
};

/// Handy-dandy hash function that provides seeds for random number generation.
__host__ __device__ constexpr unsigned int generate_hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);

  return a;
}

__host__ __device__ inline thrust::default_random_engine make_seeded_random_engine(int iteration,
                                                                                   int index,
                                                                                   int depth) {
  int input = (1 << 31) | (depth << 22) | iteration;
  int seed = generate_hash(input) ^ generate_hash(index);

  return thrust::default_random_engine(seed);
}

/// Divides `a` by `b` and rounds it up to the nearest integer.
__host__ __device__ constexpr int divide_ceil(int a, int b) {
  return (a + b - 1) / b;
};
