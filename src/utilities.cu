#include "utilities.cuh"

__host__ __device__ thrust::default_random_engine
make_seeded_random_engine(int iteration, int index, int depth) {
  int input = (1 << 31) | (depth << 22) | iteration;
  int seed = generate_hash(input) ^ generate_hash(index);

  return thrust::default_random_engine(seed);
}

__host__ __device__ int divide_ceil(int a, int b) {
  return (a + b - 1) / b;
}