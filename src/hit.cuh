#pragma once

#include "ray.cuh"
#include "scene.hpp"

#include <cuda/std/optional>

#include <glm/glm.hpp>

struct Hit {
  float t;
  glm::vec3 point;
  glm::vec3 surface_normal;
  bool is_outside;
};

/// An intersection test either resulted in a hit, or nothing at all.
using HitResult = cuda::std::optional<Hit>;

/**
 * Test intersection between a ray and a transformed cube. Untransformed, the
 * cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 */
__host__ __device__ HitResult test_cube_hit(Geometry box, Ray r);

/**
 * Test intersection between a ray and a transformed sphere. Untransformed, the
 * sphere always has radius 0.5 and is centered at the origin.
 */
__host__ __device__ HitResult test_sphere_hit(Geometry sphere, Ray r);
