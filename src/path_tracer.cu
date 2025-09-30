#include "intersection.hpp"
#include "path_segment.hpp"
#include "path_tracer.hpp"
#include "sample.hpp"
#include "tone_mapping.cuh"

#include <cuda.h>
#include <thrust/partition.h>

#include <cmath>
#include <cstdio>

namespace kernel {

/// Writes the image to the OpenGL PBO directly.
__global__ void send_to_pbo(int num_pixels,
                            uchar4* pbo,
                            int curr_iter,
                            glm::vec3* image,
                            bool apply_tone_mapping) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

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

/// Construct first batch of path segments with rays pointing from the camera into the scene.
///
/// @note Intersections don't need to be initialized here because they'll be set to a default
/// and valid value before performing intersection tests.
__global__ void initialize_segments(int num_pixels,
                                    int curr_iter,
                                    int max_depth,
                                    Camera camera,
                                    PathSegment* path_segments,
                                    bool perform_stochastic_sampling) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

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

  path_segments[index] = {
      .ray = ray,
      .throughput = glm::vec3(1.f),
      .radiance = 0.f,
      .pixel_index = index,
      .remaining_bounces = max_depth,
  };
}

/// Add the current iteration's output to the overall image.
__global__ void final_gather(int num_pixels, glm::vec3* image, PathSegment* segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  PathSegment segment = segments[index];

  if (segment.radiance <= 0.f) {
    return;
  }

  image[segment.pixel_index] += segment.radiance * segment.throughput;
}

}  // namespace kernel

/// Operations.
namespace op {

struct is_not_oob {
  __host__ __device__ bool operator()(const PathTracer::ZipTuple& tuple) const {
    return thrust::get<0>(tuple).t > 0.f;
  }
};

struct is_not_light_isect {
  __host__ __device__ bool operator()(const PathTracer::ZipTuple& tuple) const {
    return thrust::get<1>(tuple).radiance == 0.f;
  }
};

struct sort_by_material_id {
  __host__ __device__ bool operator()(const PathTracer::ZipTuple& zip_1,
                                      const PathTracer::ZipTuple& zip_2) const {
    return thrust::get<0>(zip_1).material_id < thrust::get<0>(zip_2).material_id;
  }
};

}  // namespace op

PathTracer::PathTracer(RenderContext* ctx)
    : ctx(ctx),
      dev_image(nullptr),
      dev_geometry_list(nullptr),
      dev_material_list(nullptr),
      dev_segments(nullptr),
      dev_intersections(nullptr),
      tdp_segments(nullptr),
      tdp_intersections(nullptr),
      max_depth(ctx->settings.max_depth),
      num_pixels(ctx->get_width() * ctx->get_height()),
      num_blocks_64(divide_ceil(num_pixels, BLOCK_SIZE_64)),
      num_blocks_128(divide_ceil(num_pixels, BLOCK_SIZE_128)) {}

void PathTracer::initialize() {
  const int num_pixels = ctx->get_width() * ctx->get_height();

  cudaMalloc(&dev_image, num_pixels * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, num_pixels * sizeof(glm::vec3));

  const std::vector<Geometry>& geometry = ctx->scene.geometry_list;
  cudaMalloc(&dev_geometry_list, geometry.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry_list, geometry.data(), geometry.size() * sizeof(Geometry),
             cudaMemcpyHostToDevice);

  const std::vector<Material>& materials = ctx->scene.material_list;
  cudaMalloc(&dev_material_list, materials.size() * sizeof(Material));
  cudaMemcpy(dev_material_list, materials.data(), materials.size() * sizeof(Material),
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
  cudaFree(dev_image);
  cudaFree(dev_segments);
  cudaFree(dev_geometry_list);
  cudaFree(dev_material_list);
  cudaFree(dev_intersections);

  check_cuda_error("PathTracer::free");
}

void PathTracer::run_iteration(uchar4* pbo, int curr_iter) {
  const Camera& camera = ctx->scene.camera;
  GuiData* gui_data = ctx->get_gui_data();
  int geometry_list_size = ctx->scene.geometry_list.size();

  kernel::initialize_segments<<<num_blocks_64, BLOCK_SIZE_64>>>(
      num_pixels, curr_iter, max_depth, camera, dev_segments, gui_data->stochastic_sampling);
  check_cuda_error("kern_init_segments_isects");

  int curr_depth = 0;
  int num_paths = num_pixels;

  while (true) {
    kernel::find_intersections<<<divide_ceil(num_paths, BLOCK_SIZE_128), BLOCK_SIZE_128>>>(
        num_paths, dev_geometry_list, geometry_list_size, dev_material_list, dev_segments,
        dev_intersections);
    check_cuda_error("kern_find_isects");
    curr_depth++;

    // Discard out of bounds intersections. While the goal is to remove "complete" paths,
    // we don't discard intersections with lights yet because it's required below
    if (gui_data->discard_oob_paths) {
      // int old = num_paths;
      // std::cout << std::format("[not_oob] before: {}", num_paths);
      zip_end = thrust::partition(zip_begin, zip_begin + num_paths, op::is_not_oob{});
      num_paths = thrust::distance(zip_begin, zip_end);
      // std::cout << std::format(", after: {}, diff: {}\n", num_paths, old - num_paths);
    }

    if (gui_data->sort_paths_by_material) {
      thrust::sort(zip_begin, zip_end, op::sort_by_material_id{});
    }

    kernel::sample<<<divide_ceil(num_paths, BLOCK_SIZE_128), BLOCK_SIZE_128>>>(
        num_paths, curr_iter, curr_depth, dev_material_list, dev_intersections, dev_segments);
    check_cuda_error("kern_sample");

    // TODO(aczw): stream compact away all of the following:
    // - russian roulette
    // - too many bounces within glass
    if (gui_data->discard_light_isect_paths) {
      // TODO(aczw): overhead of partitioning zip iterator vs. just path segments? Intersections
      // get reset at the end of this while loop anyway.
      zip_end = thrust::partition(zip_begin, zip_begin + num_paths, op::is_not_light_isect{});
      num_paths = thrust::distance(zip_begin, zip_end);
    }

    if (curr_depth == max_depth || num_paths == 0) {
      break;
    }
  }

  // Assemble this iteration and apply it to the image
  kernel::final_gather<<<num_blocks_128, BLOCK_SIZE_128>>>(num_pixels, dev_image, dev_segments);

  // Send results to OpenGL buffer for rendering
  kernel::send_to_pbo<<<num_blocks_64, BLOCK_SIZE_64>>>(num_pixels, pbo, curr_iter, dev_image,
                                                        gui_data->apply_tone_mapping);

  // Retrieve image from GPU
  cudaMemcpy(ctx->image.data(), dev_image, num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  check_cuda_error("PathTracer::run");
}
