#include "hit.cuh"
#include "intersection.cuh"
#include "path_tracer.h"
#include "sample.cuh"
#include "scene_structs.h"

#include <cuda.h>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/variant>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

#define ERRORCHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define check_cuda_error(msg) check_cuda_error_function(msg, FILENAME, __LINE__)

void check_cuda_error_function(const char* msg, const char* file, int line) {
#if ERRORCHECK
  cudaError_t err = cudaDeviceSynchronize();

  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");

  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif  // _WIN32
  exit(EXIT_FAILURE);
#endif  // ERRORCHECK
}

/// Kernel that writes the image to the OpenGL PBO directly.
__global__ void kern_send_to_pbo(int num_pixels, uchar4* pbo, int curr_iter, glm::vec3* image) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  glm::vec3 pixel = image[index];

  glm::ivec3 color;
  color.x = glm::clamp(static_cast<int>(pixel.x / curr_iter * 255.0), 0, 255);
  color.y = glm::clamp(static_cast<int>(pixel.y / curr_iter * 255.0), 0, 255);
  color.z = glm::clamp(static_cast<int>(pixel.z / curr_iter * 255.0), 0, 255);

  // Each thread writes one pixel location in the texture (textel)
  pbo[index].w = 0;
  pbo[index].x = color.x;
  pbo[index].y = color.y;
  pbo[index].z = color.z;
}

// TODO(aczw): convert to thrust::device_ptr? would need to use
// thrust::raw_pointer_cast when submitting these to kernels
static Scene* hst_scene = nullptr;
static GuiDataContainer* gui_data = nullptr;
static glm::vec3* dev_image = nullptr;

static Geometry* dev_geometry_list = nullptr;
static Material* dev_material_list = nullptr;

static PathSegment* dev_segments = nullptr;
static Intersection* dev_intersections = nullptr;

void init_data_container(GuiDataContainer* imgui_data) {
  gui_data = imgui_data;
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
                                       PathSegment* path_segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  int cam_res_x = camera.resolution.x;

  // Derive image x-coord and y-coord from index
  float y = glm::ceil((static_cast<float>(index) + 1.0) / cam_res_x) - 1.0;
  float x = static_cast<float>(index - (y * cam_res_x));

  Ray ray = {
      .origin = camera.position,
      // TODO(aczw): implement antialiasing by jittering the ray
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

/**
 * "Fake" shader demonstrating what you might do with the info in a `Intersection`, as well as how
 * to use Thrust's random number generator. Observe that since the Thrust random number generator
 * basically adds "noise" to the iteration, the image should start off noisy and get cleaner as more
 * iterations are computed.
 *
 * Note that this shader does NOT do a BSDF evaluation! Your shaders should handle that - this can
 * allow techniques such as bump mapping.
 */
__global__ void kern_shade_fake_material(int num_paths,
                                         int curr_iter,
                                         Material* material_list,
                                         Intersection* intersections,
                                         PathSegment* segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::visit(
      Match{
          [=](OutOfBounds) {
            // If there was no intersection, color the ray black. Lots of
            // renderers use 4 channel color, RGBA, where A = alpha, often used
            // for opacity, in which case they can indicate no opacity. This can
            // be useful for post-processing and image compositing.
            segments[index].throughput = glm::vec3();
          },

          [=](HitLight light) { segments[index].radiance = glm::vec3(light.emittance); },

          [=](Intermediate intm) {
            Material material = material_list[intm.material_id];

            // Do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            float light_term = glm::dot(intm.surface_normal, glm::vec3(0.f, 1.f, 0.f));
            segments[index].radiance = (material.color * light_term) * 0.3f +
                                       ((1.0f - intm.t * 0.02f) * material.color) * 0.7f;

            // Apply some noise because why not
            thrust::default_random_engine rng = make_seeded_random_engine(curr_iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);
            segments[index].radiance *= u01(rng);
          },
      },
      intersections[index]);
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

void PathTracer::initialize(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixel_count = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixel_count * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixel_count * sizeof(glm::vec3));

  cudaMalloc(&dev_segments, pixel_count * sizeof(PathSegment));

  cudaMalloc(&dev_geometry_list, scene->geoms.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry_list, scene->geoms.data(), scene->geoms.size() * sizeof(Geometry),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_material_list, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_material_list, scene->materials.data(), scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixel_count * sizeof(Intersection));

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
  const int max_depth = hst_scene->state.trace_depth;
  const Camera& camera = hst_scene->state.camera;
  const int geometry_size = hst_scene->geoms.size();

  const int num_pixels = camera.resolution.x * camera.resolution.y;

  const int block_size_64 = 64;
  const int num_blocks_64 = divide_ceil(num_pixels, block_size_64);
  const int block_size_128 = 128;
  const int num_blocks_128 = divide_ceil(num_pixels, block_size_128);

  const thrust::device_ptr<PathSegment> thrust_segments(dev_segments);
  const thrust::device_ptr<Intersection> thrust_intersections(dev_intersections);
  const thrust::zip_iterator zip_begin =
      thrust::make_zip_iterator(thrust_intersections, thrust_segments);

  const thrust::zip_function zip_not_oob = thrust::make_zip_function(IsNotOutOfBounds{});
  const thrust::zip_function zip_not_light_isect = thrust::make_zip_function(IsNotLightIsect{});

  // Initialize first batch of path segments
  kern_gen_rays_from_cam<<<num_blocks_64, block_size_64>>>(num_pixels, camera, curr_iter, max_depth,
                                                           dev_segments);
  check_cuda_error("kern_gen_rays_from_cam");

  int curr_depth = 0;
  int num_paths = num_pixels;

  while (true) {
    int num_blocks_isects = divide_ceil(num_paths, block_size_128);
    kern_find_isects<<<num_blocks_isects, block_size_128>>>(num_paths, dev_geometry_list,
                                                            geometry_size, dev_material_list,
                                                            dev_segments, dev_intersections);
    check_cuda_error("kern_find_isects");
    curr_depth++;

    // Discard out of bounds intersections. While the goal is to remove "complete" paths,
    // we don't discard intersections with lights yet because it's required below
    const thrust::zip_iterator zip_oob_begin =
        thrust::partition(zip_begin, zip_begin + num_paths, zip_not_oob);
    num_paths = thrust::distance(zip_begin, zip_oob_begin);

    // TODO(aczw): sort intersections by material_id, make it toggleable via UI
    thrust::sort(zip_begin, zip_oob_begin, SortByMaterialId{});

    const int num_blocks_sample = divide_ceil(num_paths, block_size_128);
    kern_sample<<<num_blocks_sample, block_size_128>>>(
        num_paths, curr_iter, curr_depth, dev_material_list, dev_intersections, dev_segments);
    check_cuda_error("kern_sample");

    // TODO(aczw): stream compact away all of the following:
    // - russian roulette
    // - too many bounces within glass
    const thrust::zip_iterator zip_light_begin =
        thrust::partition(zip_begin, zip_oob_begin, zip_not_light_isect);
    num_paths = thrust::distance(zip_begin, zip_light_begin);

    if (curr_depth == max_depth || num_paths == 0) {
      break;
    }

    if (gui_data) {
      gui_data->traced_depth = curr_depth;
    }
  }

  // Assemble this iteration and apply it to the image
  kern_final_gather<<<num_blocks_128, block_size_128>>>(num_pixels, dev_image, dev_segments);

  // Send results to OpenGL buffer for rendering
  kern_send_to_pbo<<<num_blocks_64, block_size_64>>>(num_pixels, pbo, curr_iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image, num_pixels * sizeof(glm::vec3),
             cudaMemcpyDeviceToHost);

  check_cuda_error("PathTracer::run");
}
