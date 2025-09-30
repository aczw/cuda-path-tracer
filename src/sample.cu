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
__device__ Opt<Ray> find_pure_transmission(Ray og_ray, Intersection isect, float eta) {
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

  PathSegment segment = segments[index];
  Intersection isect = intersections[index];

  if (segment.remaining_bounces == 0 || isect.t < 0.f) {
    return;
  }

  Material material = material_list[isect.material_id];
  Ray og_ray = segment.ray;

  using enum Material::Type;

  switch (material.type) {
    case Unknown: {
      segment.radiance = 1.f;
      segment.throughput = glm::vec3(1.f, 0.f, 1.f);
      segment.remaining_bounces = 0;
      break;
    }

    case Light: {
      segment.radiance = material.emission;
      segment.throughput *= material.color;
      segment.remaining_bounces = 0;
      break;
    }

    case Diffuse: {
      glm::vec3 omega_o = -og_ray.direction;

      // Calculate Lambertian term, which is also is cos(theta)
      float lambert = glm::abs(cos_theta(isect.normal, omega_o));

      // BSDF for perfectly diffuse materials is given by (albedo / pi)
      glm::vec3 bsdf = material.color * static_cast<float>(std::numbers::inv_pi);

      // PDF for cosine-weighted hemisphere sampling
      float pdf = lambert * std::numbers::inv_pi;

      segment.throughput *= bsdf * lambert / pdf;

      auto rng = make_seeded_random_engine(curr_iter, index, curr_depth);

      // Determine next ray
      segment.ray = {
          .origin = og_ray.at(isect.t),
          .direction = calculate_random_direction_in_hemisphere(isect.normal, rng),
      };

      break;
    }

    case PureReflection: {
      segment.throughput *= material.color;
      segment.ray = find_pure_reflection(og_ray, isect);
      break;
    }

    case PureTransmission: {
      if (Opt<Ray> new_ray_opt = find_pure_transmission(og_ray, isect, material.eta); new_ray_opt) {
        segment.throughput *= material.color;
        segment.ray = new_ray_opt.value();
      } else {
        // Total internal reflection
        segment.remaining_bounces = 0;
      }

      break;
    }

    case PerfectSpecular: {
      auto rng = make_seeded_random_engine(curr_iter, index, curr_depth);
      thrust::uniform_real_distribution<float> uniform_01;

      glm::vec3 omega_o = -og_ray.direction;
      float eta = material.eta;

      float refl_term = fresnel_schlick(cos_theta(isect.normal, omega_o), eta);
      float trans_term = 1.f - refl_term;

      // Either reflection or transmission
      if (uniform_01(rng) < refl_term) {
        Ray new_ray = find_pure_reflection(og_ray, isect);
        glm::vec3 omega_i = new_ray.direction;
        glm::vec3 bsdf = glm::vec3(refl_term / abs_cos_theta(isect.normal, omega_i));

        // segment.throughput *= bsdf * material.color / refl_term;
        // segment.throughput *= abs_cos_theta(isect.normal, omega_i) / refl_term;
        segment.throughput *= material.color;
        segment.ray = std::move(new_ray);
      } else {
        if (Opt<Ray> new_ray_opt = find_pure_transmission(og_ray, isect, eta); new_ray_opt) {
          Ray new_ray = new_ray_opt.value();
          glm::vec3 omega_i = new_ray.direction;
          glm::vec3 bsdf = glm::vec3(trans_term / abs_cos_theta(isect.normal, omega_o));

          // segment.throughput *= bsdf * material.color / trans_term;
          // segment.throughput *= abs_cos_theta(isect.normal, omega_i) / trans_term;
          segment.throughput *= material.color;
          segment.ray = std::move(new_ray);
        } else {
          // Total internal reflection
          segment.remaining_bounces = 0;
        }
      }

      break;
    }

    default:
      // Unreachable
      return;
  }

  segments[index] = std::move(segment);
}

}  // namespace kernel
