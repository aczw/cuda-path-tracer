#pragma once

#include "intersection.cuh"
#include "material.hpp"
#include "path_segment.hpp"

#include <thrust/random.h>

#include <glm/glm.hpp>

#include <numbers>

/// Computes a cosine-weighted random direction in a hemisphere.
__host__ __device__ glm::vec3 calculate_random_direction_in_hemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);

__host__ __device__ inline void sample_material(int index,
                                                int curr_iter,
                                                int curr_depth,
                                                Material material,
                                                Hit hit,
                                                PathSegment* segments) {
  PathSegment og_segment = segments[index];

  cuda::std::visit(
      Match{[=](UnknownMat) {
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

            [=](Specular specular) {
              Ray og_ray = og_segment.ray;

              segments[index].throughput *= specular.color;
              segments[index].ray = {
                  .origin = og_ray.get_point(hit.t),
                  .direction = glm::normalize(glm::reflect(og_ray.direction, hit.normal)),
              };
            }},
      material);
}
