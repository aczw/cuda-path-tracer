#pragma once

#include "mesh.hpp"
#include "ray.hpp"

#include <cuda/std/limits>
#include <cuda_runtime_api.h>

#include <glm/glm.hpp>

/// Axis-aligned bounding box.
struct Aabb {
  glm::vec3 min = glm::vec3(cuda::std::numeric_limits<float>::max());
  glm::vec3 max = glm::vec3(cuda::std::numeric_limits<float>::lowest());

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
    glm::vec3 t_min = (min - ray.origin) / ray.direction;
    glm::vec3 t_max = (max - ray.origin) / ray.direction;

    glm::vec3 t0 = glm::min(t_min, t_max);
    glm::vec3 t1 = glm::max(t_min, t_max);

    float far = glm::min(glm::min(t1.x, t1.y), t1.z);
    float near = glm::max(glm::max(t0.x, t0.y), t0.z);

    return far >= near && far > 0.f;
  }
};
