#include "interactions.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculate_random_direction_in_hemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng));       // cos(theta)
  float over = sqrt(1 - up * up);  // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the normal's components are
  // all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned
  // this trick from Peter Kutz.
  glm::vec3 direction_not_normal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    direction_not_normal = glm::vec3(1, 0, 0);
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    direction_not_normal = glm::vec3(0, 1, 0);
  } else {
    direction_not_normal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicular_direction_1 = glm::normalize(glm::cross(normal, direction_not_normal));
  glm::vec3 perpendicular_direction_2 =
      glm::normalize(glm::cross(normal, perpendicular_direction_1));

  return up * normal + cos(around) * over * perpendicular_direction_1 +
         sin(around) * over * perpendicular_direction_2;
}

__host__ __device__ void scatter_ray(PathSegment& path_segment,
                                     glm::vec3 intersection_point,
                                     glm::vec3 normal,
                                     const Material& mat,
                                     thrust::default_random_engine& rng) {
  Ray& ray = path_segment.ray;

  // Offset a little from the intersection surface
  ray.direction = calculate_random_direction_in_hemisphere(normal, rng);
  ray.origin = intersection_point + EPSILON * ray.direction;

  // TODO(aczw): figure out color contribution calculation
  path_segment.color *= mat.color;

  path_segment.remaining_bounces -= 1;
}
