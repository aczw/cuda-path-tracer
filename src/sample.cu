#include "sample.hpp"
#include "utilities.cuh"

#include <cuda/std/optional>

#include <numbers>

#define SQRT_ONE_THIRD 0.5773502691896257645091487805019574556476f

__device__ glm::vec3 calculate_random_direction_in_hemisphere(glm::vec3 normal,
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

__device__ Ray find_pure_reflection(Ray og_ray, Intersection isect) {
  return {
      .origin = og_ray.at(isect.t),
      .direction = glm::normalize(glm::reflect(og_ray.direction, isect.normal)),
  };
}

/// Assumes `eta` parameter ratio is target over incident.
__device__ cuda::std::optional<Ray> find_pure_transmission(Ray og_ray,
                                                           Intersection isect,
                                                           float eta) {
  // GLSL/GLM refract expects the IOR ratio to be incident over target, so
  // we treat the default as us starting from inside the material
  if (isect.surface == Surface::Outside) {
    eta = 1.f / eta;
  }

  glm::vec3 omega_i = glm::normalize(glm::refract(og_ray.direction, isect.normal, eta));

  // Handle total internal reflection
  if (omega_i == glm::vec3()) {
    return {};
  }

  return Ray{
      // Need to offset origin by an additional factor
      .origin = og_ray.at(isect.t) + 0.0001f * og_ray.direction,
      .direction = omega_i,
  };
}

/// Assumes that the other medium we're leaving/entering is always the vacuum.
__device__ float fresnel_schlick(float cos_theta, float eta) {
  float sqrt_r_0 = (eta - 1.f) / (eta + 1.f);
  float r_0 = sqrt_r_0 * sqrt_r_0;
  float term = 1.f - cos_theta;

  return r_0 + (1.f - r_0) * term * term * term * term * term;
}

__device__ float fresnel_unpolarized(float cos_theta_i, float eta) {
  cos_theta_i = glm::clamp(cos_theta_i, -1.f, 1.f);

  if (cos_theta_i < 0.f) {
    eta = 1.f / eta;
    cos_theta_i = -cos_theta_i;
  }

  float sin_2_theta_i = 1.f - (cos_theta_i * cos_theta_i);
  float sin_2_theta_t = sin_2_theta_i / (eta * eta);

  // Total internal reflection. As a result, we should only do a reflection
  if (sin_2_theta_t >= 1.f) {
    return 1.f;
  }

  // Calculate cos_theta_t using Snell's law
  float cos_theta_t = glm::sqrt(glm::max(0.f, 1.f - sin_2_theta_t));

  float r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
  float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

  return (r_parallel * r_parallel + r_perp * r_perp) / 2.f;
}

__device__ void sample_material(int index,
                                int curr_iter,
                                int curr_depth,
                                Material material,
                                Intersection isect,
                                PathSegment* segments) {
  PathSegment og_segment = segments[index];
  Ray og_ray = og_segment.ray;

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
            glm::vec3 omega_o = -og_ray.direction;

            // Calculate Lambertian term, which is also is cos(theta)
            float lambert = glm::abs(glm::dot(isect.normal, omega_o));

            // BSDF for perfectly diffuse materials is given by (albedo / pi)
            glm::vec3 bsdf = diffuse.color * static_cast<float>(std::numbers::inv_pi);

            // PDF for cosine-weighted hemisphere sampling
            float pdf = lambert * std::numbers::inv_pi;

            segments[index].throughput *= bsdf * lambert / pdf;

            auto rng = make_seeded_random_engine(curr_iter, index, curr_depth);

            // Determine next ray
            segments[index].ray = {
                .origin = og_ray.at(isect.t),
                .direction = calculate_random_direction_in_hemisphere(isect.normal, rng),
            };
          },

          [=](PureReflection specular) {
            segments[index].throughput *= specular.color;
            segments[index].ray = find_pure_reflection(og_ray, isect);
          },

          [=](PureTransmission transmissive) {
            cuda::std::optional<Ray> new_ray_opt =
                find_pure_transmission(og_ray, isect, transmissive.eta);

            // Total internal reflection
            if (!new_ray_opt) {
              return;
            }

            segments[index].throughput *= transmissive.color;
            segments[index].ray = new_ray_opt.value();
          },

          [=](PerfectSpecular perf_spec) {
            glm::vec3 omega_o = -og_ray.direction;
            float cos_theta = glm::abs(glm::dot(isect.normal, omega_o));

            auto rng = make_seeded_random_engine(curr_iter, index, curr_depth);
            thrust::uniform_real_distribution<float> uniform_01;

            float eta = perf_spec.eta;
            float refl_term = fresnel_schlick(glm::dot(isect.normal, omega_o), eta);

            // Either reflection or transmission
            if (uniform_01(rng) < refl_term) {
              glm::vec3 bsdf = glm::vec3(refl_term / cos_theta);
              segments[index].throughput *= perf_spec.color;
              segments[index].ray = find_pure_reflection(og_ray, isect);
            } else {
              float trans_term = 1.f - refl_term;

              cuda::std::optional<Ray> new_ray_opt = find_pure_transmission(og_ray, isect, eta);

              // Total internal reflection
              if (!new_ray_opt) {
                return;
              }

              glm::vec3 bsdf = glm::vec3(trans_term / cos_theta);
              segments[index].throughput *= perf_spec.color;
              segments[index].ray = new_ray_opt.value();
            }
          },
      },
      material);
}

namespace kernel {

__global__ void sample(int num_paths,
                       int curr_iter,
                       int curr_depth,
                       Material* material_list,
                       Intersection* intersections,
                       PathSegment* segments) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  Intersection isect = intersections[index];

  if (isect.t > 0.f) {
    sample_material(index, curr_iter, curr_depth, material_list[isect.material_id], isect,
                    segments);
  }
}

}  // namespace kernel
