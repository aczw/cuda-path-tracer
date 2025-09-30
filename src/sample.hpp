#pragma once

#include "intersection.hpp"
#include "material.hpp"
#include "path_segment.hpp"

#include <thrust/random.h>

#include <glm/glm.hpp>

/// Computes a cosine-weighted random direction in a hemisphere.
__device__ glm::vec3 calculate_random_direction_in_hemisphere(glm::vec3 normal,
                                                              thrust::default_random_engine& rng);

/// Given a pair of values between [0, 1), maps them to a 2D unit disk
/// centered at the origin i.e. (0, 0).
__device__ glm::vec2 sample_uniform_disk_concentric(glm::vec2 u);

/// Given a pair of values between [0, 1), maps them to a 2D unit disk
/// centered at the origin i.e. (0, 0).
__device__ glm::vec2 sample_uniform_disk_concentric(float u0, float u1);

namespace kernel {

/// Given a list of intersections, samples the material at the point and adds its contribution
/// to the overall image. Then, it determines the next ray.
__global__ void sample(int num_paths,
                       int curr_iter,
                       int curr_depth,
                       Material* material_list,
                       Intersection* intersections,
                       PathSegment* segments);

}  // namespace kernel
