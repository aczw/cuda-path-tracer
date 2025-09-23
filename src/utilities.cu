#include "utilities.cuh"

__host__ __device__ thrust::default_random_engine make_seeded_random_engine(int iteration,
                                                                            int index,
                                                                            int depth) {
  int seed = generate_hash((1 << 31) | (depth << 22) | iteration) ^ generate_hash(index);
  return thrust::default_random_engine(seed);
}
