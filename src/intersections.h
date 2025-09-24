#pragma once

#include "ray.cuh"
#include "scene_structs.h"

#include <cuda/std/optional>

#include <glm/glm.hpp>

/**
 * Test intersection between a ray and a transformed cube. Untransformed, the
 * cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 */
__host__ __device__ cuda::std::optional<Intersection> cube_intersection_test(
    Geometry box,
    Ray r);

/**
 * Test intersection between a ray and a transformed sphere. Untransformed, the
 * sphere always has radius 0.5 and is centered at the origin.
 */
__host__ __device__ cuda::std::optional<Intersection> sphere_intersection_test(
    Geometry sphere,
    Ray r);
