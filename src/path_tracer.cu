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
                                    CameraSettings settings,
                                    PathSegment* path_segments) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  int cam_res_x = camera.resolution.x;

  // Derive image x-coord and y-coord from index
  float y = glm::ceil((static_cast<float>(index) + 1.0) / cam_res_x) - 1.0;
  float x = static_cast<float>(index - y * cam_res_x);

  // Reduce aliasing via stochastic sampling
  if (settings.stochastic_sampling) {
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

  if (settings.depth_of_field && settings.lens_radius > 0.f && settings.focal_distance > 0.f) {
    thrust::default_random_engine rng = make_seeded_random_engine(curr_iter, index, max_depth);
    thrust::uniform_real_distribution<float> uniform_01;

    // Sample point on lens
    glm::vec2 sample = sample_uniform_disk_concentric(uniform_01(rng), uniform_01(rng));
    glm::vec2 lens_point = settings.lens_radius * sample;

    // We want the relative distance from the camera to the plane of focus, so it
    // doesn't matter what sign  the ray direction is
    float t = settings.focal_distance / glm::abs(ray.direction.z);
    glm::vec3 focus = ray.at(t);

    // Offset ray origin by lens sample point and adjust direction such that the ray still
    // intersects with the same point on the plane of focus
    ray.origin += glm::vec3(lens_point.x, lens_point.y, 0.f);
    ray.direction = glm::normalize(focus - ray.origin);
  }

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
  int index = blockIdx.x * blockDim.x + threadIdx.x;

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
      dev_triangle_list(nullptr),
      dev_bvh_node_list(nullptr),
      dev_bvh_tri_list(nullptr),
      dev_normal_list(nullptr),
      dev_position_list(nullptr),
      dev_segments(nullptr),
      dev_intersections(nullptr),
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

  const std::vector<Triangle>& triangles = ctx->scene.triangle_list;
  cudaMalloc(&dev_triangle_list, triangles.size() * sizeof(Triangle));
  cudaMemcpy(dev_triangle_list, triangles.data(), triangles.size() * sizeof(Triangle),
             cudaMemcpyHostToDevice);
  check_cuda_error("PathTracer::initialize: cudaMalloc(dev_triangle_list)");

  const std::vector<bvh::Node>& bvh_nodes = ctx->scene.bvh_node_list;
  cudaMalloc(&dev_bvh_node_list, bvh_nodes.size() * sizeof(bvh::Node));
  cudaMemcpy(dev_bvh_node_list, bvh_nodes.data(), bvh_nodes.size() * sizeof(bvh::Node),
             cudaMemcpyHostToDevice);
  check_cuda_error("PathTracer::initialize: cudaMalloc(dev_bvh_node_list)");

  const std::vector<Triangle>& bvh_tris = ctx->scene.bvh_tri_list;
  cudaMalloc(&dev_bvh_tri_list, bvh_tris.size() * sizeof(Triangle));
  cudaMemcpy(dev_bvh_tri_list, bvh_tris.data(), bvh_tris.size() * sizeof(Triangle),
             cudaMemcpyHostToDevice);
  check_cuda_error("PathTracer::initialize: cudaMalloc(dev_bvh_tri_list)");

  const std::vector<glm::vec3>& positions = ctx->scene.position_list;
  cudaMalloc(&dev_position_list, positions.size() * sizeof(glm::vec3));
  cudaMemcpy(dev_position_list, positions.data(), positions.size() * sizeof(glm::vec3),
             cudaMemcpyHostToDevice);
  check_cuda_error("PathTracer::initialize: cudaMalloc(dev_position_list)");

  const std::vector<glm::vec3>& normals = ctx->scene.normal_list;
  cudaMalloc(&dev_normal_list, normals.size() * sizeof(glm::vec3));
  cudaMemcpy(dev_normal_list, normals.data(), normals.size() * sizeof(glm::vec3),
             cudaMemcpyHostToDevice);
  check_cuda_error("PathTracer::initialize: cudaMalloc(dev_position_list)");

  cudaMalloc(&dev_segments, num_pixels * sizeof(PathSegment));
  cudaMalloc(&dev_intersections, num_pixels * sizeof(Intersection));

  // Wrap in thrust::device_ptr first to make Thrust happy
  begin = thrust::make_zip_iterator(IntersectionThrustPtr(dev_intersections),
                                    PathSegmentThrustPtr(dev_segments));

  check_cuda_error("PathTracer::initialize");
}

void PathTracer::free() {
  cudaFree(dev_image);
  cudaFree(dev_geometry_list);
  cudaFree(dev_material_list);
  cudaFree(dev_triangle_list);
  cudaFree(dev_bvh_node_list);
  cudaFree(dev_bvh_tri_list);
  cudaFree(dev_position_list);
  cudaFree(dev_normal_list);
  cudaFree(dev_segments);
  cudaFree(dev_intersections);

  check_cuda_error("PathTracer::free");
}

void PathTracer::run_iteration(uchar4* pbo, int curr_iter) {
  const Camera& camera = ctx->scene.camera;
  GuiData* gui_data = ctx->get_gui_data();
  const int geometry_list_size = ctx->scene.geometry_list.size();

  kernel::initialize_segments<<<num_blocks_64, BLOCK_SIZE_64>>>(
      num_pixels, curr_iter, max_depth, camera, gui_data->camera, dev_segments);
  check_cuda_error("kernel::initialize_segments");

  int curr_depth = 0;
  int num_paths = num_pixels;
  thrust::zip_iterator end = begin + num_pixels;

  while (true) {
    kernel::find_intersections<<<divide_ceil(num_paths, BLOCK_SIZE_128), BLOCK_SIZE_128>>>(
        num_paths, dev_geometry_list, geometry_list_size, dev_material_list, dev_triangle_list,
        dev_position_list, dev_normal_list, dev_bvh_node_list, dev_bvh_tri_list, dev_segments,
        dev_intersections, gui_data->bbox_isect_culling, gui_data->bvh_isect_culling);
    check_cuda_error("kernel::find_intersections");
    curr_depth++;

    // Discard out of bounds intersections. While the goal is to remove "complete" paths,
    // we don't discard intersections with lights yet because we need to sample them first below
    if (gui_data->discard_oob_paths) {
      end = thrust::partition(begin, end, op::is_not_oob{});
      num_paths = thrust::distance(begin, end);
      check_cuda_error("thrust::partition: op::is_not_oob");
    }

    if (gui_data->sort_paths_by_material) {
      thrust::sort(begin, end, op::sort_by_material_id{});
      check_cuda_error("thrust::sort: op::sort_by_material_id");
    }

    kernel::sample<<<divide_ceil(num_paths, BLOCK_SIZE_128), BLOCK_SIZE_128>>>(
        num_paths, curr_iter, curr_depth, dev_material_list, dev_intersections, dev_segments);
    check_cuda_error("kernel::sample");

    // TODO(aczw): stream compact away all of the following:
    // - russian roulette
    // - too many bounces within glass
    if (gui_data->discard_light_isect_paths) {
      end = thrust::partition(begin, end, op::is_not_light_isect{});
      num_paths = thrust::distance(begin, end);
      check_cuda_error("thrust::partition: op::is_not_light_isect");
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
