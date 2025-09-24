#pragma once

#include <cuda/std/variant>

#include <glm/glm.hpp>

struct OutOfBounds {};

struct HitLight {
  float material_emittance;
};

struct Intermediate {
  int material_id;
  float t;
  glm::vec3 surface_normal;
};

/// An intersection can only exist in three possible states:
///
/// - No intersection occurred because the ray went out of bounds.
/// - An intersection occurred with a light (i.e. emissive material).
/// - A "regular" intersection occurred with some geometry in the scene.
using Intersection = cuda::std::variant<Intermediate, HitLight, OutOfBounds>;

/// Helper for usage in `cuda::std::visit`. Taken from
/// https://en.cppreference.com/w/cpp/utility/variant/visit2.html#Example
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
