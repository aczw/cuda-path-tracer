#pragma once

#include "mesh.hpp"
#include "ray.hpp"

#include <cuda/std/limits>
#include <cuda_runtime_api.h>

#include <glm/glm.hpp>

/// Axis-aligned bounding box.
struct Aabb {
  glm::vec3 min = glm::vec3(cuda::std::numeric_limits<float>::infinity());
  glm::vec3 max = glm::vec3(-cuda::std::numeric_limits<float>::infinity());

  inline glm::vec3 get_center() const { return (min + max) / 2.f; }

  inline glm::vec3 get_size() const { return glm::abs(min - max); }

  /// Adds `point` to the bounding box, growing the bounds if necessary.
  inline void include(glm::vec3 point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
  }

  /// Adds all the points of a triangle to the bounding box, growing the bounds if necessary.
  inline void include(const Triangle& triangle,
                      const std::vector<glm::vec3>& positions,
                      const glm::mat4& transform) {
    include(glm::vec3(transform * glm::vec4(positions[triangle[0].pos_idx], 1.f)));
    include(glm::vec3(transform * glm::vec4(positions[triangle[1].pos_idx], 1.f)));
    include(glm::vec3(transform * glm::vec4(positions[triangle[2].pos_idx], 1.f)));
  }

  /// Checks whether a ray intersected with this box. Adapted from
  /// https://tavianator.com/2022/ray_box_boundary.html.
  __device__ inline bool intersect(Ray ray) const {
    float t_min = -cuda::std::numeric_limits<float>::infinity();
    float t_max = cuda::std::numeric_limits<float>::infinity();
    glm::vec3 inv_dir = 1.f / ray.direction;

    for (int axis = 0; axis < 3; ++axis) {
      if (ray.direction[axis] != 0.f) {
        float t0 = (min[axis] - ray.origin[axis]) * inv_dir[axis];
        float t1 = (max[axis] - ray.origin[axis]) * inv_dir[axis];

        t_min = glm::max(t_min, glm::min(t0, t1));
        t_max = glm::min(t_max, glm::max(t0, t1));
      } else if (ray.origin[axis] <= min[axis] || ray.origin[axis] >= max[axis]) {
        return false;
      }
    }

    return t_max > t_min && t_max > 0.f;
  }
};
