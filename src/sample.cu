#include "sample.cuh"
#include "utilities.cuh"

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

__host__ __device__ inline void sample_material(int index,
                                                int curr_iter,
                                                int curr_depth,
                                                Material material,
                                                Hit hit,
                                                PathSegment* segments) {
  PathSegment og_segment = segments[index];

  cuda::std::visit(
      Match{
          [=](UnknownMat) {
            segments[index].radiance = 1.f;
            segments[index].throughput = glm::vec3(1.f, 0.f, 1.f);
          },

          [=](Light light) {
            segments[index].radiance = light.emission;
            segments[index].throughput *= light.color;
          },

          [=](Diffuse diffuse) {
            Ray og_ray = og_segment.ray;
            glm::vec3 omega_o = -og_ray.direction;

            // Calculate Lambertian term, which is also is cos(theta)
            float lambert = glm::abs(glm::dot(hit.normal, omega_o));

            // BSDF for perfectly diffuse materials is given by (albedo / pi)
            glm::vec3 bsdf = diffuse.color * static_cast<float>(std::numbers::inv_pi);

            // PDF for cosine-weighted hemisphere sampling
            float pdf = lambert * std::numbers::inv_pi;

            segments[index].throughput *= bsdf * lambert / pdf;

            auto rng = make_seeded_random_engine(curr_iter, index, curr_depth);

            // Determine next ray
            segments[index].ray = {
                .origin = og_ray.get_point(hit.t),
                .direction = calculate_random_direction_in_hemisphere(hit.normal, rng),
            };
          },

          [=](PureReflection specular) {
            Ray og_ray = og_segment.ray;

            segments[index].throughput *= specular.color;
            segments[index].ray = {
                .origin = og_ray.get_point(hit.t),
                .direction = glm::normalize(glm::reflect(og_ray.direction, hit.normal)),
            };
          },

          [=](PureTransmission transmissive) {
            Ray og_ray = og_segment.ray;

            // GLSL/GLM refract expects the IOR ratio to be incident over target, so
            // we treat the default as us starting from inside the material
            float eta = transmissive.eta;
            if (hit.surface == Surface::Outside) {
              eta = 1.f / eta;
            }

            glm::vec3 result = glm::refract(og_ray.direction, hit.normal, eta);

            // Handle total internal reflection
            if (result == glm::vec3()) {
              return;
            }

            segments[index].throughput *= transmissive.color;
            segments[index].ray = {
                // Need to offset origin by an additional factor. Otherwise, it appears that
                // the new origin isn't fully inside the material yet.
                .origin = og_ray.get_point(hit.t) + 0.0001f * og_ray.direction,
                .direction = glm::normalize(result),
            };
          },
      },
      material);
}
