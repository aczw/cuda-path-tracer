#pragma once

#include "scene.hpp"

#include <thrust/random.h>

#include <glm/glm.hpp>

/**
 * Computes a cosine-weighted random direction in a hemisphere. Used for diffuse
 * lighting.
 */
__host__ __device__ glm::vec3 calculate_random_direction_in_hemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);
