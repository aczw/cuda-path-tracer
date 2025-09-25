#pragma once

#include "utilities.cuh"

#include <cuda/std/optional>
#include <cuda/std/variant>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

struct OutOfBounds {};

struct HitLight {
  char material_id;
  float emittance;
};

struct Intermediate {
  char material_id;
  float t;
  glm::vec3 surface_normal;
};

/// An intersection can only exist in three possible states:
///
/// - A "regular" intersection occurred with some geometry in the scene.
/// - An intersection occurred with a light (i.e. emissive material).
/// - No intersection occurred because the ray went out of bounds.
using Intersection = cuda::std::variant<Intermediate, HitLight, OutOfBounds>;

/// Returns the material ID from the intersection.
__host__ __device__ inline cuda::std::optional<char> get_material_id(Intersection isect) {
  return cuda::std::visit<cuda::std::optional<char>>(
      Match{
          [](OutOfBounds) { return cuda::std::nullopt; },
          [](HitLight light) { return light.material_id; },
          [](Intermediate intm) { return intm.material_id; },
      },
      isect);
}
