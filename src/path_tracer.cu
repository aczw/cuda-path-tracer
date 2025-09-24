#include "hit.cuh"
#include "path_tracer.h"
#include "sample.cuh"
#include "scene_structs.h"

#include <cuda.h>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <thrust/execution_policy.h>
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
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
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
static Geometry* dev_geometry = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_path_segments = nullptr;
static cuda::std::optional<Intersection>* dev_shading_data = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...

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
                                       int trace_depth,
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
      .remaining_bounces = trace_depth,
  };
}

/**
 * Generates shading data only. Generating new rays from this data is handled in
 * the shaders.
 */
__global__ void kern_find_isects(int depth,
                                 int num_paths,
                                 PathSegment* path_segments,
                                 Geometry* geometry,
                                 int geometry_size,
                                 cuda::std::optional<Intersection>* shading_data) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index >= num_paths) {
    return;
  }

  Ray path_ray = path_segments[path_index].ray;

  float t_min = cuda::std::numeric_limits<float>::max();
  int hit_geometry_index = -1;
  glm::vec3 surface_normal;

  // Naively parse through global geometry
  // TODO(aczw): use better intersection algorithm i.e. acceleration structures
  for (int geometry_index = 0; geometry_index < geometry_size; ++geometry_index) {
    Geometry geom = geometry[geometry_index];
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
    shading_data[path_index] = cuda::std::nullopt;
  } else {
    shading_data[path_index] = {
        .t = t_min,
        .surface_normal = surface_normal,
        .material_id = geometry[hit_geometry_index].material_id,
    };
  }
}

/**
 * "Fake" shader demonstrating what you might do with the info in a
 * `Intersection`, as well as how to use Thrust's random number generator.
 * Observe that since the Thrust random number generator basically adds "noise"
 * to the iteration, the image should start off noisy and get cleaner as more
 * iterations are computed.
 *
 * Note that this shader does NOT do a BSDF evaluation! Your shaders should
 * handle that - this can allow techniques such as bump mapping.
 */
__global__ void shade_fake_material(int curr_iteration,
                                    int num_paths,
                                    cuda::std::optional<Intersection>* shading_data,
                                    PathSegment* path_segments,
                                    Material* materials) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::optional<Intersection> data_opt = shading_data[index];

  // If there was no intersection, color the ray black. Lots of renderers use 4
  // channel color, RGBA, where A = alpha, often used for opacity, in which case
  // they can indicate no opacity. This can be useful for post-processing and
  // image compositing.
  if (!data_opt) {
    path_segments[index].throughput = glm::vec3();
    return;
  }

  const Intersection& data = data_opt.value();

  // Set up the RNG. LOOK: this is how you use thrust's RNG! Please look at
  // make_seeded_random_engine as well.
  thrust::default_random_engine rng = make_seeded_random_engine(curr_iteration, index, 0);
  thrust::uniform_real_distribution<float> u01(0, 1);

  Material material = materials[data.material_id];
  glm::vec3 material_color = material.color;

  // If the material indicates that the object was a light, "light" the ray
  if (material.emittance > 0.f) {
    path_segments[index].throughput *= material_color * material.emittance;
  } else {
    // Otherwise, do some pseudo-lighting computation. This is actually more
    // like what you would expect from shading in a rasterizer like OpenGL.
    // TODO: replace this! you should be able to start with basically a
    // one-liner
    float lightTerm = glm::dot(data.surface_normal, glm::vec3(0.0f, 1.0f, 0.0f));
    path_segments[index].throughput *=
        (material_color * lightTerm) * 0.3f + ((1.0f - data.t * 0.02f) * material_color) * 0.7f;
    path_segments[index].throughput *= u01(rng);  // apply some noise because why not
  }
}

__global__ void kern_sample(int curr_iter,
                            int num_pixels,
                            int curr_depth,
                            cuda::std::optional<Intersection>* shading_data,
                            PathSegment* path_segments,
                            Material* materials) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  cuda::std::optional<Intersection> data_opt = shading_data[index];

  if (!data_opt) {
    return;
  }

  const Intersection& data = data_opt.value();

  // Set up the RNG. LOOK: this is how you use thrust's RNG! Please look at
  // make_seeded_random_engine as well.
  thrust::default_random_engine rng = make_seeded_random_engine(curr_iter, index, curr_depth);
  thrust::uniform_real_distribution<float> u01(0, 1);

  Material material = materials[data.material_id];
  glm::vec3 material_color = material.color;

  if (material.emittance > 0.f) {
    // If the material indicates that the object was a light, "light" the ray.
    // This also indicates that this path is complete
    path_segments[index].radiance = material.emittance * path_segments[index].throughput;
  } else {
    glm::vec3 omega_o = -path_segments[index].ray.direction;

    // Calculate simple Lambertian lighting
    glm::vec3 bsdf = material_color * static_cast<float>(std::numbers::inv_pi);

    // Cosine-weighted hemisphere sampling
    float pdf = glm::abs(glm::dot(data.surface_normal, omega_o) /
                         (glm::length(data.surface_normal) * glm::length(omega_o))) /
                std::numbers::pi;

    float lambert = glm::abs(glm::dot(data.surface_normal, omega_o));

    path_segments[index].throughput *= bsdf * lambert / pdf;
  }

  const Ray& original_ray = path_segments[index].ray;

  // Determine next ray
  glm::vec3 new_direction = calculate_random_direction_in_hemisphere(data.surface_normal, rng);
  path_segments[index].ray.origin =
      original_ray.origin + (data.t * original_ray.direction) + (EPSILON * data.surface_normal);
  path_segments[index].ray.direction = new_direction;
}

/**
 * Add the current iteration's output to the overall image.
 */
__global__ void kern_final_gather(int num_pixels, glm::vec3* image, PathSegment* path_segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  const PathSegment& segment = path_segments[index];
  image[segment.pixel_index] += segment.radiance;
}

PathTracer::PathTracer(glm::ivec2 resolution) {}

void PathTracer::initialize(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixel_count = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixel_count * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixel_count * sizeof(glm::vec3));

  cudaMalloc(&dev_path_segments, pixel_count * sizeof(PathSegment));

  cudaMalloc(&dev_geometry, scene->geoms.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry, scene->geoms.data(), scene->geoms.size() * sizeof(Geometry),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_shading_data, pixel_count * sizeof(cuda::std::optional<Intersection>));
  cudaMemset(dev_shading_data, 0, pixel_count * sizeof(cuda::std::optional<Intersection>));

  check_cuda_error("PathTracer::initialize");
}

void PathTracer::free() {
  // No-op if dev_image is null
  cudaFree(dev_image);
  cudaFree(dev_path_segments);
  cudaFree(dev_geometry);
  cudaFree(dev_materials);
  cudaFree(dev_shading_data);

  check_cuda_error("PathTracer::free");
}

void PathTracer::run(uchar4* pbo, int curr_iter) {
  const int trace_depth = hst_scene->state.trace_depth;
  const Camera& camera = hst_scene->state.camera;
  const int geometry_size = hst_scene->geoms.size();
  const int num_pixels = camera.resolution.x * camera.resolution.y;

  const int block_size_64 = 64;
  const int num_blocks_64 = divide_ceil(num_pixels, block_size_64);
  const int block_size_128 = 128;
  const int num_blocks_128 = divide_ceil(num_pixels, block_size_128);

  // Initialize first batch of path segments
  kern_gen_rays_from_cam<<<num_blocks_64, block_size_64>>>(num_pixels, camera, curr_iter,
                                                           trace_depth, dev_path_segments);
  check_cuda_error("kern_gen_rays_from_cam");

  int curr_depth = 0;
  int num_paths = num_pixels;

  // Shoot ray into scene, bounce between objects, push shading chunks
  while (true) {
    // Clean shading chunks
    cudaMemset(dev_shading_data, 0, num_pixels * sizeof(cuda::std::optional<Intersection>));

    int num_blocks_isects = divide_ceil(num_paths, block_size_128);
    kern_find_isects<<<num_blocks_isects, block_size_128>>>(
        curr_depth, num_paths, dev_path_segments, dev_geometry, geometry_size, dev_shading_data);
    check_cuda_error("kern_find_isects");

    curr_depth++;

    // TODO(aczw): stream compact away out of bounds intersections

    // TODO(aczw): sort intersections by material_id, make it toggleable via UI

    const int num_blocks_sample = divide_ceil(num_paths, block_size_128);
    kern_sample<<<num_blocks_sample, block_size_128>>>(
        curr_iter, num_paths, curr_depth, dev_shading_data, dev_path_segments, dev_materials);

    // TODO(aczw): stream compact away all of the following:
    // - intersection with lights (do this first)
    // - russian roulette
    // - too many bounces within glass

    // TODO(aczw): should be based off of stream compaction results (i.e. all
    // paths have been stream compacted away)
    if (curr_depth > 7) {
      break;
    }

    if (gui_data) {
      gui_data->traced_depth = curr_depth;
    }
  }

  // Assemble this iteration and apply it to the image
  kern_final_gather<<<num_blocks_128, block_size_128>>>(num_pixels, dev_image, dev_path_segments);

  // Send results to OpenGL buffer for rendering
  kern_send_to_pbo<<<num_blocks_64, block_size_64>>>(num_pixels, pbo, curr_iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image, num_pixels * sizeof(glm::vec3),
             cudaMemcpyDeviceToHost);

  check_cuda_error("PathTracer::run");
}
