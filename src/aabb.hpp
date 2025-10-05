#pragma once

#include <glm/glm.hpp>

/// Axis-aligned bounding box.
struct Aabb {
  glm::vec3 min;
  glm::vec3 max;

  inline glm::vec3 get_center() const { return (min + max) / 2.f; }

  /// Adds `point` to the bounding box, growing the bounds if necessary.
  inline void include(glm::vec3 point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
  }
};
