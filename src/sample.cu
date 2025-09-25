#include "sample.cuh"
#include "utilities.cuh"

#include <thrust/random.h>

#include <numbers>

#define SQRT_ONE_THIRD 0.5773502691896257645091487805019574556476f

__host__ __device__ glm::vec3 calculate_random_direction_in_hemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng) {
  thrust::uniform_real_distribution<float> uniform_01;

  float up = std::sqrt(uniform_01(rng));  // cos(theta)
  float over = std::sqrt(1 - up * up);    // sin(theta)
  float around = uniform_01(rng) * 2.f * std::numbers::pi;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at least
  // one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
  glm::vec3 direction_not_normal;
  if (std::abs(normal.x) < SQRT_ONE_THIRD) {
    direction_not_normal = glm::vec3(1, 0, 0);
  } else if (std::abs(normal.y) < SQRT_ONE_THIRD) {
    direction_not_normal = glm::vec3(0, 1, 0);
  } else {
    direction_not_normal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perp_dir_1 = glm::normalize(glm::cross(normal, direction_not_normal));
  glm::vec3 perp_dir_2 = glm::normalize(glm::cross(normal, perp_dir_1));

  return up * normal + std::cos(around) * over * perp_dir_1 + std::sin(around) * over * perp_dir_2;
}
