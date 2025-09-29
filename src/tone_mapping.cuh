#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

/// Expects values in the range [0.0, 1.0].
__device__ inline glm::vec3 apply_reinhard(glm::vec3 hdr) {
  return hdr / (glm::vec3(1.f) + hdr);
}

/// Expects values in the range [0.0, 1.0].
__device__ inline glm::vec3 gamma_correct(glm::vec3 color) {
  return glm::pow(color, glm::vec3(1.f / 2.2f));
}
