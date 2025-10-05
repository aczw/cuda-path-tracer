#pragma once

#include "aabb.hpp"

#include <glm/glm.hpp>

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
