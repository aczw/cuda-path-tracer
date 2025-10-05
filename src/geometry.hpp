#pragma once

#include "aabb.hpp"

#include <cuda/std/array>

#include <glm/glm.hpp>

struct Vertex {
  int pos_idx;
  int nor_idx;
};

/// A triangle is simply a trio of vertices containing the indices
/// pointing at various vertex attributes.
using Triangle = cuda::std::array<Vertex, 3>;

struct Geometry {
  enum class Type { Sphere, Cube, Gltf } type;

  char material_id;
  int tri_begin;
  int tri_end;

  /// World-space bounding box for this geometry.
  Aabb bbox;
  int bvh_root_idx;

  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inv_transform;
  glm::mat4 inv_transpose;
};
