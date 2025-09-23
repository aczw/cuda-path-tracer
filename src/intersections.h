#pragma once

#include "scene_structs.h"

#include <cuda/std/optional>

#include <glm/glm.hpp>

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`. Falls slightly short so that it doesn't
 * intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 get_point_on_ray(Ray r, float t) {
  return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiply_mat4_vec4(glm::mat4 m, glm::vec4 v) {
  return glm::vec3(m * v);
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed, the cube ranges from -0.5
 * to 0.5 in each axis and is centered at the origin.
 */
__host__ __device__ cuda::std::optional<Intersection> cube_intersection_test(Geometry box, Ray r);

/**
 * Test intersection between a ray and a transformed sphere. Untransformed, the sphere always has
 * radius 0.5 and is centered at the origin.
 */
__host__ __device__ cuda::std::optional<Intersection> sphere_intersection_test(Geometry sphere,
                                                                               Ray r);
