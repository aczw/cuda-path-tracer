#include "interactions.h"
#include "intersections.h"
#include "path_trace.h"
#include "scene.h"
#include "scene_structs.h"
#include "utilities.h"

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

__host__ __device__ thrust::default_random_engine make_seeded_random_engine(int iter, int index, int depth) {
  int h = util_hash((1 << 31) | (depth << 22) | iter) ^ util_hash(index);
  return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void send_image_to_pbo(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
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

// TODO(aczw): convert to thrust::device_ptr? would need to use thrust::raw_pointer_cast
// when submitting these to kernels
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

void path_trace_init(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixel_count = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixel_count * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixel_count * sizeof(glm::vec3));

  cudaMalloc(&dev_path_segments, pixel_count * sizeof(PathSegment));

  cudaMalloc(&dev_geometry, scene->geoms.size() * sizeof(Geometry));
  cudaMemcpy(dev_geometry, scene->geoms.data(), scene->geoms.size() * sizeof(Geometry), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_shading_data, pixel_count * sizeof(cuda::std::optional<ShadingData>));
  cudaMemset(dev_shading_data, 0, pixel_count * sizeof(cuda::std::optional<ShadingData>));

  check_cuda_error("path_trace_init");
}

void path_trace_free() {
  // No-op if dev_image is null
  cudaFree(dev_image);
  cudaFree(dev_path_segments);
  cudaFree(dev_geometry);
  cudaFree(dev_materials);
  cudaFree(dev_shading_data);

  check_cuda_error("path_trace_free");
}

/**
 * Generate `PathSegment`s with rays from the camera through the screen into the scene, which is the first bounce of
 * rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generate_ray_from_camera(Camera cam, int iter, int trace_depth, PathSegment* path_segments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    PathSegment& segment = path_segments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
    segment.radiance = glm::vec3();

    // TODO: implement antialiasing by jittering the ray
    segment.ray.direction =
        glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f) -
                       cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));

    segment.pixel_index = index;
    segment.remaining_bounces = trace_depth;
  }
}

/**
 * Generates shading data only. Generating new rays from this data is handled in the shaders.
 */
__global__ void compute_intersections(int depth,
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
  for (int geometry_index = 0; geometry_index < geometry_size; ++geometry_index) {
    Geometry& geom = geometry[geometry_index];
    cuda::std::optional<Intersection> curr_intersection_opt;

    // TODO(aczw): add more intersection tests here... Triangle? Metaball? CSG?
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

    // Compute the minimum t from the intersection tests to determine what scene geometry object is the closest.
    if (curr_intersection_opt && t_min > curr_intersection_opt->t) {
      const Intersection& intersection = curr_intersection_opt.value();

      t_min = intersection.t;
      hit_geometry_index = geometry_index;
      surface_normal = intersection.surface_normal;
    }
  }

  if (hit_geometry_index == -1) {
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
 * "Fake" shader demonstrating what you might do with the info in a `ShadingData`, as well as how to use Thrust's random
 * number generator. Observe that since the Thrust random number generator basically adds "noise" to the iteration, the
 * image should start off noisy and get cleaner as more iterations are computed.
 *
 * Note that this shader does NOT do a BSDF evaluation! Your shaders should handle that - this can allow techniques such
 * as bump mapping.
 */
__global__ void shade_fake_material(int iter,
                                    int num_paths,
                                    cuda::std::optional<ShadingData>* shading_data,
                                    PathSegment* path_segments,
                                    Material* materials) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_paths) {
    return;
  }

  cuda::std::optional<ShadingData> data_opt = shading_data[index];

  // If there was no intersection, color the ray black. Lots of renderers use 4 channel color, RGBA, where A = alpha,
  // often used for opacity, in which case they can indicate no opacity. This can be useful for post-processing and
  // image compositing.
  if (!data_opt) {
    path_segments[index].color = glm::vec3();
    return;
  }

  const ShadingData& data = data_opt.value();

  // Set up the RNG. LOOK: this is how you use thrust's RNG! Please look at make_seeded_random_engine as well.
  thrust::default_random_engine rng = make_seeded_random_engine(iter, index, 0);
  thrust::uniform_real_distribution<float> u01(0, 1);

  Material material = materials[data.material_id];
  glm::vec3 materialColor = material.color;

  // If the material indicates that the object was a light, "light" the ray
  if (material.emittance > 0.0f) {
    path_segments[index].color *= (materialColor * material.emittance);
  } else {
    // Otherwise, do some pseudo-lighting computation. This is actually more
    // like what you would expect from shading in a rasterizer like OpenGL.
    // TODO: replace this! you should be able to start with basically a one-liner
    float lightTerm = glm::dot(data.surface_normal, glm::vec3(0.0f, 1.0f, 0.0f));
    path_segments[index].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - data.t * 0.02f) * materialColor) * 0.7f;
    path_segments[index].color *= u01(rng);  // apply some noise because why not
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixel_index] += iterationPath.color;
  }
}

/**
 * Wrapper for the `__global__` call that sets up the kernel calls and does a ton of memory management
 */
void path_trace(uchar4* pbo, int frame, int iter) {
  const int trace_depth = hst_scene->state.trace_depth;
  const Camera& camera = hst_scene->state.camera;
  const int num_pixels = camera.resolution.x * camera.resolution.y;

  // 2D block for generating ray from camera
  const dim3 block_size_2d(8, 8);
  const dim3 blocks_per_grid_2d((camera.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
                                (camera.resolution.y + block_size_2d.y - 1) / block_size_2d.y);

  // 1D block for path tracing
  const int block_size_1d = 128;

  // Recap:
  // * For each depth:
  //   * Compute an intersection in the scene for each path ray.
  //     A very naive version of this has been implemented for you, but feel
  //     free to add more primitives and/or a better algorithm.
  //     Currently, intersection distance is recorded as a parametric distance,
  //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
  //     * Color is attenuated (multiplied) by reflections off of any object
  //   * TODO: Stream compact away all of the terminated paths.
  //     You may use either your implementation or `thrust::remove_if` or its
  //     cousins.
  //     * Note that you can't really use a 2D kernel launch any more - switch
  //       to 1D.
  //   * TODO: Shade the rays that intersected something or didn't bottom out.
  //     That is, color the ray by performing a color computation according
  //     to the shader, then generate a new ray to continue the ray path.
  //     We recommend just updating the ray's PathSegment in place.
  //     Note that this step may come before or after stream compaction,
  //     since some shaders you write may also cause a path to terminate.
  // * Finally, add this iteration's results to the image. This has been done
  //   for you.

  // TODO: perform one iteration of path tracing

  // Initialize `dev_path_segments` by using rays that come out of the camera.
  generate_ray_from_camera<<<blocks_per_grid_2d, block_size_2d>>>(camera, iter, trace_depth, dev_path_segments);
  check_cuda_error("generate_ray_from_camera");

  int depth = 0;
  PathSegment* dev_path_segments_end = dev_path_segments + num_pixels;
  int num_paths = dev_path_segments_end - dev_path_segments;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iteration_done = false;

  while (!iteration_done) {
    // Clean shading chunks
    cudaMemset(dev_shading_data, 0, num_pixels * sizeof(cuda::std::optional<ShadingData>));

    // Tracing
    dim3 num_blocks_path_segment_tracing = (num_paths + block_size_1d - 1) / block_size_1d;
    compute_intersections<<<num_blocks_path_segment_tracing, block_size_1d>>>(
        depth, num_paths, dev_path_segments, dev_geometry, hst_scene->geoms.size(), dev_shading_data);
    check_cuda_error("compute_intersections: trace one bounce");
    cudaDeviceSynchronize();
    depth++;

    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on shading_data and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    shade_fake_material<<<num_blocks_path_segment_tracing, block_size_1d>>>(iter, num_paths, dev_shading_data,
                                                                            dev_path_segments, dev_materials);
    iteration_done = true;  // TODO: should be based off stream compaction results.

    if (gui_data != NULL) {
      gui_data->traced_depth = depth;
    }
  }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (num_pixels + block_size_1d - 1) / block_size_1d;
  finalGather<<<numBlocksPixels, block_size_1d>>>(num_paths, dev_image, dev_path_segments);

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  send_image_to_pbo<<<blocks_per_grid_2d, block_size_2d>>>(pbo, camera.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image, num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  check_cuda_error("path_trace");
}
