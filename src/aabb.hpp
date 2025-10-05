#pragma once

#include "ray.hpp"
#include "scene.hpp"

#include <cuda/std/limits>
#include <cuda_runtime_api.h>

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

  /// Checks whether a ray intersected with this box. Adapted from
  /// https://tavianator.com/2022/ray_box_boundary.html.
  __device__ inline bool intersect(Ray ray) const {
    float t_min = 0.f;
    float t_max = cuda::std::numeric_limits<float>::infinity();
    glm::vec3 inv_direction = 1.f / ray.direction;

    for (int dir = 0; dir < 3; ++dir) {
      float t0 = (min[dir] - ray.origin[dir]) * inv_direction[dir];
      float t1 = (max[dir] - ray.origin[dir]) * inv_direction[dir];

      t_min = glm::max(t_min, glm::min(glm::min(t0, t1), t_max));
      t_max = glm::min(t_max, glm::max(glm::max(t0, t1), t_min));
    }

    return t_min <= t_max;
  }
};
