#pragma once

#include "intersection.hpp"
#include "material.hpp"
#include "path_segment.hpp"

#include <thrust/random.h>

#include <glm/glm.hpp>

/// Computes a cosine-weighted random direction in a hemisphere.
__device__ glm::vec3 calculate_random_direction_in_hemisphere(glm::vec3 normal,
                                                              thrust::default_random_engine& rng);

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
