#include "hit.cuh"
#include "intersection.cuh"
#include "path_segment.hpp"
#include "path_tracer.h"
#include "sample.cuh"
#include "tone_mapping.cuh"

#include <cuda.h>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/variant>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/zip_function.h>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

/// Kernel that writes the image to the OpenGL PBO directly.
__global__ void kern_send_to_pbo(int num_pixels,
                                 uchar4* pbo,
                                 int curr_iter,
                                 glm::vec3* image,
                                 bool apply_tone_mapping) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  glm::vec3 pixel = image[index] / static_cast<float>(curr_iter);

  if (apply_tone_mapping) {
    pixel = glm::clamp(gamma_correct(apply_reinhard(pixel)), glm::vec3(), glm::vec3(1.f));
  }

  glm::ivec3 color;
  color.x = glm::clamp(static_cast<int>(pixel.x * 255.f), 0, 255);
  color.y = glm::clamp(static_cast<int>(pixel.y * 255.f), 0, 255);
  color.z = glm::clamp(static_cast<int>(pixel.z * 255.f), 0, 255);

  // Each thread writes one pixel location in the texture (textel)
  pbo[index].w = 0;
  pbo[index].x = color.x;
  pbo[index].y = color.y;
  pbo[index].z = color.z;
}

/**
 * Generate `PathSegment`s with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void kern_gen_rays_from_cam(int num_pixels,
                                       Camera camera,
                                       int curr_iter,
                                       int max_depth,
                                       PathSegment* path_segments,
                                       bool perform_stochastic_sampling) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  int cam_res_x = camera.resolution.x;

  // Derive image x-coord and y-coord from index
  float y = glm::ceil((static_cast<float>(index) + 1.0) / cam_res_x) - 1.0;
  float x = static_cast<float>(index - y * cam_res_x);

  // Reduce aliasing via stochastic sampling
  if (perform_stochastic_sampling) {
    thrust::default_random_engine rng = make_seeded_random_engine(curr_iter, index, max_depth);
    thrust::uniform_real_distribution<float> uniform_01;

    y += uniform_01(rng);
    x += uniform_01(rng);
  }

  Ray ray = {
      .origin = camera.position,
      .direction = glm::normalize(
          camera.view -
          camera.right * camera.pixel_length.x * (x - static_cast<float>(cam_res_x) * 0.5f) -
          camera.up * camera.pixel_length.y * (y - static_cast<float>(camera.resolution.y) * 0.5f)),
  };

  // Initialize path segment
  path_segments[index] = {
      .ray = ray,
      .radiance = glm::vec3(),
      .throughput = glm::vec3(1.f),
      .pixel_index = index,
      .remaining_bounces = max_depth,
  };
}

/**
 * Generates intersections only. Sampling new rays from this intersection is handled in
 * another kernel down the line.
 */
__global__ void kern_find_isects(int num_paths,
                                 Geometry* geometry_list,
                                 int geometry_list_size,
                                 Material* material_list,
                                 PathSegment* segments,
                                 Intersection* intersections) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index >= num_paths) {
    return;
  }

  Ray path_ray = segments[path_index].ray;

  float t_min = cuda::std::numeric_limits<float>::max();
  int hit_geometry_index = -1;
  glm::vec3 surface_normal;

  // Naively parse through global geometry
  // TODO(aczw): use better intersection algorithm i.e. acceleration structures
  for (int geometry_index = 0; geometry_index < geometry_list_size; ++geometry_index) {
    Geometry geom = geometry_list[geometry_index];
    HitResult result;

    switch (geom.type) {
      case Geometry::Type::Cube:
        result = test_cube_hit(geom, path_ray);
        break;

      case Geometry::Type::Sphere:
        result = test_sphere_hit(geom, path_ray);
        break;

      default:
        break;
    }

    // Discovered a closer object, record it
    if (result.has_value() && t_min > result->t) {
      t_min = result->t;
      hit_geometry_index = geometry_index;
      surface_normal = result->surface_normal;
    }
  }

  // Check whether we hit any geometry at all
  if (hit_geometry_index == -1) {
    intersections[path_index] = OutOfBounds{};
  } else {
    char material_id = geometry_list[hit_geometry_index].material_id;

    if (const Material material = material_list[material_id]; material.emittance > 0.f) {
      intersections[path_index] = HitLight{
          .material_id = material_id,
          .emittance = material.emittance,
      };
    } else {
      intersections[path_index] = Intermediate{
          .material_id = geometry_list[hit_geometry_index].material_id,
          .t = t_min,
          .surface_normal = surface_normal,
      };
    }
  }
}

__global__ void kern_sample(int num_paths,
                            int curr_iter,
                            int curr_depth,
                            Material* material_list,
                            Intersection* intersections,
                            PathSegment* segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::visit(
      Match{
          [=](OutOfBounds) {},

          [=](HitLight light) {
            segments[index].radiance = light.emittance * segments[index].throughput;
          },

          [=](Intermediate intm) {
            Material material = material_list[intm.material_id];
            PathSegment segment = segments[index];
            Ray ray = segment.ray;

            glm::vec3 omega_o = -ray.direction;

            // Lambertian term is cos_theta in this case
            float lambert = glm::abs(glm::dot(intm.surface_normal, omega_o));

            // Calculate simple Lambertian lighting
            glm::vec3 bsdf = material.color * static_cast<float>(std::numbers::inv_pi);

            // Cosine-weighted hemisphere sampling
            float pdf = lambert / (glm::length(intm.surface_normal) * glm::length(omega_o)) /
                        std::numbers::pi;

            segments[index].throughput *= bsdf * lambert / pdf;

            thrust::default_random_engine rng =
                make_seeded_random_engine(curr_iter, index, curr_depth);

            // Determine next ray
            segments[index].ray = {
                .origin = ray.origin + (intm.t * ray.direction) + (EPSILON * intm.surface_normal),
                .direction = calculate_random_direction_in_hemisphere(intm.surface_normal, rng),
            };
          },
      },
      intersections[index]);
}

/**
 * Add the current iteration's output to the overall image.
 */
__global__ void kern_final_gather(int num_pixels, glm::vec3* image, PathSegment* segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  PathSegment segment = segments[index];
  image[segment.pixel_index] += segment.radiance;
}

struct IsNotOutOfBounds {
  __host__ __device__ bool operator()(Intersection isect, PathSegment) {
    return !cuda::std::holds_alternative<OutOfBounds>(isect);
  }
};

struct IsNotLightIsect {
  __host__ __device__ bool operator()(Intersection isect, PathSegment) {
    return !cuda::std::holds_alternative<HitLight>(isect);
  }
};

struct SortByMaterialId {
  __host__ __device__ bool operator()(auto zip_1, auto zip_2) {
    return get_material_id(thrust::get<0>(zip_1)) < get_material_id(thrust::get<0>(zip_2));
  }
};

PathTracer::PathTracer(RenderContext* ctx)
    : ctx(ctx),
      dev_image(nullptr),
      dev_geometry_list(nullptr),
      dev_material_list(nullptr),
      dev_segments(nullptr),
      dev_intersections(nullptr),
      tdp_segments(nullptr),
      tdp_intersections(nullptr),
      num_blocks_64(divide_ceil(ctx->get_width() * ctx->get_height(), BLOCK_SIZE_64)),
      num_blocks_128(divide_ceil(ctx->get_width() * ctx->get_height(), BLOCK_SIZE_128)) {}

void PathTracer::initialize() {
  const int num_pixels = ctx->get_width() * ctx->get_height();

  cudaMalloc(&dev_image, num_pixels * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, num_pixels * sizeof(glm::vec3));

  const std::vector<Geometry>& geometry_list = ctx->scene.geometry_list;
  cudaMalloc(&dev_geometry_list, geometry_list.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry_list, geometry_list.data(), geometry_list.size() * sizeof(Geometry),
             cudaMemcpyHostToDevice);

  const std::vector<Material>& material_list = ctx->scene.material_list;
  cudaMalloc(&dev_material_list, material_list.size() * sizeof(Material));
  cudaMemcpy(dev_material_list, material_list.data(), material_list.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_segments, num_pixels * sizeof(PathSegment));
  tdp_segments = thrust::device_ptr<PathSegment>(dev_segments);

  cudaMalloc(&dev_intersections, num_pixels * sizeof(Intersection));
  tdp_intersections = thrust::device_ptr<Intersection>(dev_intersections);

  zip_begin = thrust::make_zip_iterator(tdp_intersections, tdp_segments);
  zip_end = thrust::make_zip_iterator(tdp_intersections + num_pixels, tdp_segments + num_pixels);

  check_cuda_error("PathTracer::initialize");
}

void PathTracer::free() {
  // No-op if dev_image is null
  cudaFree(dev_image);
  cudaFree(dev_segments);
  cudaFree(dev_geometry_list);
  cudaFree(dev_material_list);
  cudaFree(dev_intersections);

  check_cuda_error("PathTracer::free");
}

void PathTracer::run_iteration(uchar4* pbo, int curr_iter) {
  static const thrust::zip_function not_oob = thrust::make_zip_function(IsNotOutOfBounds{});
  static const thrust::zip_function not_light_isect = thrust::make_zip_function(IsNotLightIsect{});

  const int max_depth = ctx->settings.max_depth;
  const Camera& camera = ctx->scene.camera;
  const int num_pixels = ctx->get_width() * ctx->get_height();

  GuiData* gui_data = ctx->get_gui_data();

  // Initialize first batch of path segments
  kern_gen_rays_from_cam<<<num_blocks_64, BLOCK_SIZE_64>>>(
      num_pixels, camera, curr_iter, max_depth, dev_segments, gui_data->stochastic_sampling);
  check_cuda_error("kern_gen_rays_from_cam");

  int curr_depth = 0;
  int num_paths = num_pixels;

  while (true) {
    int num_blocks_isects = divide_ceil(num_paths, BLOCK_SIZE_128);
    kern_find_isects<<<num_blocks_isects, BLOCK_SIZE_128>>>(
        num_paths, dev_geometry_list, ctx->scene.geometry_list.size(), dev_material_list,
        dev_segments, dev_intersections);
    check_cuda_error("kern_find_isects");
    curr_depth++;

    // Discard out of bounds intersections. While the goal is to remove "complete" paths,
    // we don't discard intersections with lights yet because it's required below
    if (gui_data->discard_oob_paths) {
      zip_end = thrust::partition(zip_begin, zip_begin + num_paths, not_oob);
      num_paths = thrust::distance(zip_begin, zip_end);
    }

    if (gui_data->sort_paths_by_material) {
      thrust::sort(zip_begin, zip_end, SortByMaterialId{});
    }

    const int num_blocks_sample = divide_ceil(num_paths, BLOCK_SIZE_128);
    kern_sample<<<num_blocks_sample, BLOCK_SIZE_128>>>(
        num_paths, curr_iter, curr_depth, dev_material_list, dev_intersections, dev_segments);
    check_cuda_error("kern_sample");

    // TODO(aczw): stream compact away all of the following:
    // - russian roulette
    // - too many bounces within glass
    if (gui_data->discard_light_isect_paths) {
      zip_end = thrust::partition(zip_begin, zip_end, not_light_isect);
      num_paths = thrust::distance(zip_begin, zip_end);
    }

    if (curr_depth == max_depth || num_paths == 0) {
      break;
    }

    gui_data->max_depth = curr_depth;
  }

  // Assemble this iteration and apply it to the image
  kern_final_gather<<<num_blocks_128, BLOCK_SIZE_128>>>(num_pixels, dev_image, dev_segments);

  // Send results to OpenGL buffer for rendering
  kern_send_to_pbo<<<num_blocks_64, BLOCK_SIZE_64>>>(num_pixels, pbo, curr_iter, dev_image,
                                                     gui_data->apply_tone_mapping);

  // Retrieve image from GPU
  cudaMemcpy(ctx->image.data(), dev_image, num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  check_cuda_error("PathTracer::run");
}
