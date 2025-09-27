#pragma once

#include "ray.cuh"

#include <glm/glm.hpp>

struct PathSegment {
  Ray ray;
  glm::vec3 throughput;
  float radiance;
  int pixel_index;
  int remaining_bounces;
};
