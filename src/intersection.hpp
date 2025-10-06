#pragma once

#include "bvh.hpp"
#include "geometry.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "path_segment.hpp"
#include "ray.hpp"

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
__device__ Intersection test_cube_isect(const Geometry& cube, Ray ray);

/// Test intersection between a ray and a transformed sphere. Untransformed, the
/// sphere always has radius 0.5 and is centered at the origin.
__device__ Intersection test_sphere_isect(const Geometry& sphere, Ray ray);

/// Performs a series of ray-triangle intersection tests on a list of triangles. They are
/// assumed to be stored contiguously, hence the begin and end parameters.
///
/// @param obj_ray Ray to test intersection for. Must be in local object space.
/// @return The intersection. If successful, the intersection point and normal are in local object
/// space, and must be converted back. The material ID also needs to be set.
__device__ Intersection test_tri_list_isect(int tri_idx_begin,
                                            int tri_idx_end,
                                            Ray obj_ray,
                                            const Triangle* triangle_list,
                                            const glm::vec3* position_list,
                                            const glm::vec3* normal_list);

__device__ Intersection test_gltf_isect(const Geometry& gltf,
                                        Ray ray,
                                        Triangle* triangle_list,
                                        glm::vec3* position_list,
                                        glm::vec3* normal_list);

__device__ Intersection test_bvh_isect(int root_node_idx,
                                       Ray world_ray,
                                       const Geometry& geometry,
                                       const bvh::Node* node_list,
                                       const Triangle* triangle_list,
                                       const glm::vec3* position_list,
                                       const glm::vec3* normal_list);

namespace kernel {

/// Finds the intersection with scene geometry, if any.
__global__ void find_intersections(int num_paths,
                                   Geometry* geometry_list,
                                   int geometry_list_size,
                                   Material* material_list,
                                   Triangle* triangle_list,
                                   glm::vec3* position_list,
                                   glm::vec3* normal_list,
                                   bvh::Node* bvh_node_list,
                                   Triangle* bvh_tri_list,
                                   PathSegment* segments,
                                   Intersection* intersections,
                                   bool bbox_isect_culling,
                                   bool bvh_isect_culling);

}  // namespace kernel
