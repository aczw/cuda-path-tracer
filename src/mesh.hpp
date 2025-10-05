#pragma once

#include <cuda/std/array>
#include <cuda_runtime_api.h>

#include <glm/glm.hpp>

#include <vector>

struct Vertex {
  int pos_idx;
  int nor_idx;
};

/// A triangle is simply a trio of vertices containing the indices
/// pointing at various vertex attributes.
struct Triangle {
  cuda::std::array<Vertex, 3> verts;

  /// Convenience overload for member access operator.
  __host__ __device__ Vertex& operator[](cuda::std::array<Vertex, 3>::size_type idx) {
    return verts[idx];
  }

  /// Convenience overload for read-only member access operator.
  __host__ __device__ const Vertex& operator[](cuda::std::array<Vertex, 3>::size_type idx) const {
    return verts[idx];
  }

  glm::vec3 get_center(const std::vector<glm::vec3>& positions) const {
    return (positions[verts[0].pos_idx] + positions[verts[1].pos_idx] +
            positions[verts[2].pos_idx]) /
           3.f;
  }
};
