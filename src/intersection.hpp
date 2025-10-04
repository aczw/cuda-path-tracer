#pragma once

#include "material.hpp"
#include "path_segment.hpp"
#include "ray.cuh"
#include "scene.hpp"

#include <cuda_runtime.h>

#include <glm/glm.hpp>

/// Tracks which side of the geometry the intersection is at.
enum class Surface : char { Inside, Outside };

/// An intersection either resulted in a hit on a geometry surface,
/// or went out of bounds, in which case `t` is set to a negative value.
struct Intersection {
  float t;
  glm::vec3 point;
  glm::vec3 normal;
  Surface surface;
  char material_id;
};

/// Test intersection between a ray and a transformed cube. Untransformed, the
/// cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
__device__ Intersection test_cube_isect(Geometry cube, Ray ray);

/// Test intersection between a ray and a transformed sphere. Untransformed, the
/// sphere always has radius 0.5 and is centered at the origin.
__device__ Intersection test_sphere_isect(Geometry sphere, Ray ray);

__device__ Intersection test_gltf_isect(Geometry gltf,
                                        Ray ray,
                                        Triangle* triangle_list,
                                        glm::vec3* position_list);

namespace kernel {

/// Finds the intersection with scene geometry, if any.
__global__ void find_intersections(int num_paths,
                                   Geometry* geometry_list,
                                   int geometry_list_size,
                                   Material* material_list,
                                   Triangle* triangle_list,
                                   glm::vec3* position_list,
                                   PathSegment* segments,
                                   Intersection* intersections);

}  // namespace kernel
