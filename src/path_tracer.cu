#include "interactions.h"
#include "intersections.h"
#include "path_tracer.h"
#include "scene.h"
#include "scene_structs.h"
#include "utilities.cuh"

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
#define FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
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

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void send_image_to_pbo(uchar4* pbo,
                                  glm::ivec2 resolution,
                                  int iter,
                                  glm::vec3* image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];

    glm::ivec3 color;
    color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

// TODO(aczw): convert to thrust::device_ptr? would need to use
// thrust::raw_pointer_cast when submitting these to kernels
static Scene* hst_scene = nullptr;
static GuiDataContainer* gui_data = nullptr;
static glm::vec3* dev_image = nullptr;
static Geometry* dev_geometry = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_path_segments = nullptr;
static cuda::std::optional<ShadingData>* dev_shading_data = nullptr;

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
__global__ void generate_ray_from_camera(Camera cam,
                                         int iter,
                                         int trace_depth,
                                         PathSegment* path_segments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    PathSegment& segment = path_segments[index];

    segment.ray.origin = cam.position;
    segment.radiance = glm::vec3();
    segment.throughput = glm::vec3(1.0f);

    // TODO: implement antialiasing by jittering the ray
    segment.ray.direction =
        glm::normalize(cam.view -
                       cam.right * cam.pixelLength.x *
                           ((float)x - (float)cam.resolution.x * 0.5f) -
                       cam.up * cam.pixelLength.y *
                           ((float)y - (float)cam.resolution.y * 0.5f));

    segment.pixel_index = index;
    segment.remaining_bounces = trace_depth;
  }
}

/**
 * Generates shading data only. Generating new rays from this data is handled in
 * the shaders.
 */
__global__ void compute_intersections(
    int depth,
    int num_paths,
    PathSegment* path_segments,
    Geometry* geometry,
    int geometry_size,
    cuda::std::optional<ShadingData>* shading_data) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index >= num_paths) {
    return;
  }

  Ray path_ray = path_segments[path_index].ray;

  float t_min = cuda::std::numeric_limits<float>::max();
  int hit_geometry_index = -1;
  glm::vec3 surface_normal;

  // TODO(aczw): do something with this value
  bool is_outside = true;

  // Naively parse through global geometry
  // TODO(aczw): use better intersection algorithm i.e. acceleration structures
  for (int geometry_index = 0; geometry_index < geometry_size;
       ++geometry_index) {
    Geometry& geom = geometry[geometry_index];
    cuda::std::optional<Intersection> curr_intersection_opt;

    switch (geom.type) {
      case Geometry::Type::Cube:
        curr_intersection_opt = cube_intersection_test(geom, path_ray);
        break;

      case Geometry::Type::Sphere:
        curr_intersection_opt = sphere_intersection_test(geom, path_ray);
        break;

      default:
        break;
    }

    // Compute the minimum t to determine what scene geometry object is the
    // closest
    if (curr_intersection_opt && t_min > curr_intersection_opt->t) {
      const Intersection& intersection = curr_intersection_opt.value();

      t_min = intersection.t;
      hit_geometry_index = geometry_index;
      surface_normal = intersection.surface_normal;
    }
  }

  if (hit_geometry_index == -1) {
    // Intersection calculation went out of bounds, path ends here
    shading_data[path_index] = cuda::std::nullopt;
  } else {
    ShadingData data;
    data.t = t_min;
    data.material_id = geometry[hit_geometry_index].material_id;
    data.surface_normal = surface_normal;

    shading_data[path_index] = data;
  }
}

/**
 * "Fake" shader demonstrating what you might do with the info in a
 * `ShadingData`, as well as how to use Thrust's random number generator.
 * Observe that since the Thrust random number generator basically adds "noise"
 * to the iteration, the image should start off noisy and get cleaner as more
 * iterations are computed.
 *
 * Note that this shader does NOT do a BSDF evaluation! Your shaders should
 * handle that - this can allow techniques such as bump mapping.
 */
__global__ void shade_fake_material(
    int curr_iteration,
    int num_paths,
    cuda::std::optional<ShadingData>* shading_data,
    PathSegment* path_segments,
    Material* materials) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::optional<ShadingData> data_opt = shading_data[index];

  // If there was no intersection, color the ray black. Lots of renderers use 4
  // channel color, RGBA, where A = alpha, often used for opacity, in which case
  // they can indicate no opacity. This can be useful for post-processing and
  // image compositing.
  if (!data_opt) {
    path_segments[index].throughput = glm::vec3();
    return;
  }

  const ShadingData& data = data_opt.value();

  // Set up the RNG. LOOK: this is how you use thrust's RNG! Please look at
  // make_seeded_random_engine as well.
  thrust::default_random_engine rng =
      make_seeded_random_engine(curr_iteration, index, 0);
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
    float lightTerm =
        glm::dot(data.surface_normal, glm::vec3(0.0f, 1.0f, 0.0f));
    path_segments[index].throughput *=
        (material_color * lightTerm) * 0.3f +
        ((1.0f - data.t * 0.02f) * material_color) * 0.7f;
    path_segments[index].throughput *=
        u01(rng);  // apply some noise because why not
  }
}

__global__ void shade_material(int curr_iteration,
                               int num_paths,
                               int curr_depth,
                               cuda::std::optional<ShadingData>* shading_data,
                               PathSegment* path_segments,
                               Material* materials) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::optional<ShadingData> data_opt = shading_data[index];

  if (!data_opt) {
    return;
  }

  const ShadingData& data = data_opt.value();

  // Set up the RNG. LOOK: this is how you use thrust's RNG! Please look at
  // make_seeded_random_engine as well.
  thrust::default_random_engine rng =
      make_seeded_random_engine(curr_iteration, index, curr_depth);
  thrust::uniform_real_distribution<float> u01(0, 1);

  Material material = materials[data.material_id];
  glm::vec3 material_color = material.color;

  if (material.emittance > 0.f) {
    // If the material indicates that the object was a light, "light" the ray.
    // This also indicates that this path is complete
    path_segments[index].radiance =
        material.emittance * path_segments[index].throughput;
  } else {
    glm::vec3 omega_o = -path_segments[index].ray.direction;

    // Calculate simple Lambertian lighting
    glm::vec3 bsdf = material_color * static_cast<float>(std::numbers::inv_pi);

    // Cosine-weighted hemisphere sampling
    float pdf =
        glm::abs(glm::dot(data.surface_normal, omega_o) /
                 (glm::length(data.surface_normal) * glm::length(omega_o))) /
        std::numbers::pi;

    float lambert = glm::abs(glm::dot(data.surface_normal, omega_o));

    path_segments[index].throughput *= bsdf * lambert / pdf;
  }

  const Ray& original_ray = path_segments[index].ray;

  // Determine next ray
  glm::vec3 new_direction =
      calculate_random_direction_in_hemisphere(data.surface_normal, rng);
  path_segments[index].ray.origin = original_ray.origin +
                                    (data.t * original_ray.direction) +
                                    (EPSILON * data.surface_normal);
  path_segments[index].ray.direction = new_direction;
}

/**
 * Add the current iteration's output to the overall image
 */
__global__ void final_gather(int num_paths,
                             glm::vec3* image,
                             PathSegment* path_segments) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  const PathSegment& segment = path_segments[index];
  image[segment.pixel_index] += segment.radiance;
}

namespace path_tracer {

void initialize(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixel_count = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixel_count * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixel_count * sizeof(glm::vec3));

  cudaMalloc(&dev_path_segments, pixel_count * sizeof(PathSegment));

  cudaMalloc(&dev_geometry, scene->geoms.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry, scene->geoms.data(),
             scene->geoms.size() * sizeof(Geometry), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(),
             scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_shading_data,
             pixel_count * sizeof(cuda::std::optional<ShadingData>));
  cudaMemset(dev_shading_data, 0,
             pixel_count * sizeof(cuda::std::optional<ShadingData>));

  check_cuda_error("path_trace_init");
}

void free() {
  // No-op if dev_image is null
  cudaFree(dev_image);
  cudaFree(dev_path_segments);
  cudaFree(dev_geometry);
  cudaFree(dev_materials);
  cudaFree(dev_shading_data);

  check_cuda_error("path_trace_free");
}

void run(uchar4* pbo, int curr_iteration) {
  const int trace_depth = hst_scene->state.trace_depth;
  const Camera& camera = hst_scene->state.camera;
  const int num_pixels = camera.resolution.x * camera.resolution.y;

  // 2D block for generating ray from camera
  const dim3 block_size_2d(8, 8);
  const dim3 blocks_per_grid_2d(
      (camera.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
      (camera.resolution.y + block_size_2d.y - 1) / block_size_2d.y);

  // 1D block for path tracing
  const int block_size_1d = 128;

  // Initialize `dev_path_segments` by using rays that come out of the camera.
  generate_ray_from_camera<<<blocks_per_grid_2d, block_size_2d>>>(
      camera, curr_iteration, trace_depth, dev_path_segments);
  check_cuda_error("generate_ray_from_camera");

  int curr_depth = 0;
  PathSegment* dev_path_segments_end = dev_path_segments + num_pixels;
  int num_paths = dev_path_segments_end - dev_path_segments;

  // Shoot ray into scene, bounce between objects, push shading chunks
  while (true) {
    // Clean shading chunks
    cudaMemset(dev_shading_data, 0,
               num_pixels * sizeof(cuda::std::optional<ShadingData>));

    // Tracing
    dim3 num_blocks_path_segment_tracing =
        (num_paths + block_size_1d - 1) / block_size_1d;
    compute_intersections<<<num_blocks_path_segment_tracing, block_size_1d>>>(
        curr_depth, num_paths, dev_path_segments, dev_geometry,
        hst_scene->geoms.size(), dev_shading_data);
    check_cuda_error("compute_intersections: trace one bounce");
    cudaDeviceSynchronize();
    curr_depth++;

    // TODO(aczw): stream compaction away dead paths here (for now, this means
    // `shading_data` contains `cuda::std::nullopt` instead of actual data)
    //
    // Note that you can't really use a 2D kernel launch any more - switch to
    // 1D.

    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on shading_data and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.
    shade_material<<<num_blocks_path_segment_tracing, block_size_1d>>>(
        curr_iteration, num_paths, curr_depth, dev_shading_data,
        dev_path_segments, dev_materials);

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
  dim3 num_blocks_pixels = (num_pixels + block_size_1d - 1) / block_size_1d;
  final_gather<<<num_blocks_pixels, block_size_1d>>>(num_paths, dev_image,
                                                     dev_path_segments);

  // Send results to OpenGL buffer for rendering
  send_image_to_pbo<<<blocks_per_grid_2d, block_size_2d>>>(
      pbo, camera.resolution, curr_iteration, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  check_cuda_error("path_trace");
}

}  // namespace path_tracer
