#pragma once

#include "ray.cuh"
#include "scene.hpp"
#include "utilities.cuh"

#include <cuda/std/optional>
#include <cuda/std/variant>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

struct OutOfBounds {
  /// The last material this path intersected with before going OOB.
  char prev_material_id;
};

enum class Surface : char { Inside, Outside };

struct Hit {
  float t;
  glm::vec3 point;
  glm::vec3 normal;
  Surface surface;
  char material_id;
};

/// An intersection either resulted in a hit on a geometry surface, or went out of bounds.
using Intersection = cuda::std::variant<OutOfBounds, Hit>;

/// Returns the material ID from the intersection.
__host__ __device__ inline char get_material_id(Intersection isect) {
  return cuda::std::visit(Match{
                              [](OutOfBounds oob) { return oob.prev_material_id; },
                              [](Hit hit) { return hit.material_id; },
                          },
                          isect);
}

/// Test intersection between a ray and a transformed cube. Untransformed, the
/// cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
__host__ __device__ cuda::std::optional<Hit> test_cube_hit(Geometry cube, Ray ray);

/// Test intersection between a ray and a transformed sphere. Untransformed, the
/// sphere always has radius 0.5 and is centered at the origin.
__host__ __device__ cuda::std::optional<Hit> test_sphere_hit(Geometry sphere, Ray ray);
