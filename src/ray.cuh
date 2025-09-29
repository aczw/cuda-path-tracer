#pragma once

#include "utilities.cuh"

#include <cuda_runtime.h>

#include <glm/glm.hpp>

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;

  /// Assumes that direction is normalized.
  __host__ __device__ inline glm::vec3 at(float t) const {
    return origin + (t - EPSILON) * direction;
  }
};
