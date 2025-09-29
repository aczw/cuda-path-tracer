#pragma once

#include "intersection.cuh"
#include "material.hpp"
#include "path_segment.hpp"

#include <thrust/random.h>

#include <glm/glm.hpp>

/// Computes a cosine-weighted random direction in a hemisphere.
__host__ __device__ glm::vec3 calculate_random_direction_in_hemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);

__host__ __device__ void sample_material(int index,
                                         int curr_iter,
                                         int curr_depth,
                                         Material material,
                                         Hit hit,
                                         PathSegment* segments);
