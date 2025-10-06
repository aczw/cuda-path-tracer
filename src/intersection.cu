#include "intersection.hpp"

#include <cuda/std/cmath>
#include <cuda/std/limits>

#include <glm/gtx/intersect.hpp>

__device__ Intersection test_cube_isect(const Geometry& cube, Ray ray) {
  float t_min = cuda::std::numeric_limits<float>::lowest();
  float t_max = cuda::std::numeric_limits<float>::max();
  glm::vec3 t_min_n;
  glm::vec3 t_max_n;

  Ray obj_ray = {
      .origin = glm::vec3(cube.inv_transform * glm::vec4(ray.origin, 1.f)),
      .direction = glm::vec3(cube.inv_transform * glm::vec4(ray.direction, 0.f)),
  };

  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = obj_ray.direction[xyz];

    /*if (glm::abs(qdxyz) > 0.00001f)*/
    {
      float t1 = (-0.5f - obj_ray.origin[xyz]) / qdxyz;
      float t2 = (0.5f - obj_ray.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);

      glm::vec3 n;
      n[xyz] = t2 < t1 ? 1.f : -1.f;

      if (ta > 0.f && ta > t_min) {
        t_min = ta;
        t_min_n = n;
      }

      if (tb < t_max) {
        t_max = tb;
        t_max_n = n;
      }
    }
  }

  Intersection isect;

  if (t_max >= t_min && t_max > 0.f) {
    isect.surface = Surface::Outside;

    if (t_min <= 0.f) {
      t_min = t_max;
      t_min_n = t_max_n;
      isect.surface = Surface::Inside;
    }

    isect.point = glm::vec3(cube.transform * glm::vec4(obj_ray.at(t_min), 1.f));
    isect.normal = glm::normalize(glm::vec3(cube.inv_transpose * glm::vec4(t_min_n, 0.f)));
    isect.t = glm::length(ray.origin - isect.point);
    isect.material_id = cube.material_id;
  } else {
    isect.t = -1.f;
  }

  return isect;
}

__device__ Intersection test_sphere_isect(const Geometry& sphere, Ray ray) {
  static const float radius = 0.5f;

  Intersection isect;
  isect.t = -1.f;

  Ray obj_ray = {
      .origin = glm::vec3(sphere.inv_transform * glm::vec4(ray.origin, 1.f)),
      .direction = glm::normalize(glm::vec3(sphere.inv_transform * glm::vec4(ray.direction, 0.f))),
  };

  float vector_dot_direction = glm::dot(obj_ray.origin, obj_ray.direction);
  float radicand = vector_dot_direction * vector_dot_direction -
                   (glm::dot(obj_ray.origin, obj_ray.origin) - (radius * radius));

  if (radicand < 0.f) {
    return isect;
  }

  float square_root = std::sqrt(radicand);
  float first_term = -vector_dot_direction;
  float t1 = first_term + square_root;
  float t2 = first_term - square_root;

  float t = 0.f;
  if (t1 < 0.f && t2 < 0.f) {
    return isect;
  } else if (t1 > 0.f && t2 > 0.f) {
    t = glm::min(t1, t2);
    isect.surface = Surface::Outside;
  } else {
    // Not sure if this takes into account intersections w.r.t. the tangent
    // of the sphere. Can't just assume we're inside the sphere?
    t = glm::max(t1, t2);
    isect.surface = Surface::Inside;
  }

  glm::vec3 obj_point = obj_ray.at(t);

  isect.point = glm::vec3(sphere.transform * glm::vec4(obj_point, 1.f));
  isect.t = glm::length(ray.origin - isect.point);
  isect.normal = glm::normalize(glm::vec3(sphere.inv_transpose * glm::vec4(obj_point, 0.f)));
  isect.material_id = sphere.material_id;

  if (isect.surface == Surface::Inside) {
    isect.normal = -isect.normal;
  }

  return isect;
}

__device__ Intersection test_tri_list_isect(int tri_idx_begin,
                                            int tri_idx_end,
                                            Ray obj_ray,
                                            const Triangle* triangle_list,
                                            const glm::vec3* position_list,
                                            const glm::vec3* normal_list) {
  Intersection isect;
  isect.t = -1.f;
  float t_min = cuda::std::numeric_limits<float>::max();

  for (int tri_idx = tri_idx_begin; tri_idx < tri_idx_end; ++tri_idx) {
    const Triangle& triangle = triangle_list[tri_idx];
    const glm::vec3 v0 = position_list[triangle[0].pos_idx];
    const glm::vec3 v1 = position_list[triangle[1].pos_idx];
    const glm::vec3 v2 = position_list[triangle[2].pos_idx];

    glm::vec3 bary;
    if (!glm::intersectRayTriangle(obj_ray.origin, obj_ray.direction, v0, v1, v2, bary)) {
      continue;
    }

    if (t_min > bary.z) {
      float u = bary.x;
      float v = bary.y;
      float w = 1.f - u - v;

      const glm::vec3 normal = normal_list[triangle[1].nor_idx];
      const glm::vec3 point = w * v0 + u * v1 + v * v2;

      if (glm::dot(normal, obj_ray.direction) < 0.f) {
        isect.surface = Surface::Outside;
      } else {
        isect.surface = Surface::Inside;
      }

      // Keep in local space, perform transform outside
      isect.normal = normal;
      isect.point = point;
      isect.t = bary.z;

      t_min = isect.t;
    }
  }

  return isect;
}

__device__ Intersection test_gltf_isect(const Geometry& gltf,
                                        Ray ray,
                                        Triangle* triangle_list,
                                        glm::vec3* position_list,
                                        glm::vec3* normal_list) {
  Ray obj_ray = {
      .origin = glm::vec3(gltf.inv_transform * glm::vec4(ray.origin, 1.f)),
      .direction = glm::vec3(gltf.inv_transform * glm::vec4(ray.direction, 0.f)),
  };

  Intersection isect = test_tri_list_isect(gltf.tri_begin, gltf.tri_end, obj_ray, triangle_list,
                                           position_list, normal_list);

  if (isect.t < 0.f) return isect;

  // Transform back to world space
  isect.normal = glm::normalize(glm::vec3(gltf.inv_transpose * glm::vec4(isect.normal, 0.f)));
  isect.point = glm::vec3(gltf.transform * glm::vec4(isect.point, 1.f));
  isect.material_id = gltf.material_id;

  return isect;
}

__device__ Intersection test_bvh_isect(int root_node_idx,
                                       Ray world_ray,
                                       const Geometry& geometry,
                                       const bvh::Node* node_list,
                                       const Triangle* triangle_list,
                                       const glm::vec3* position_list,
                                       const glm::vec3* normal_list) {
  Intersection isect;
  isect.t = -1.f;
  float t_min = cuda::std::numeric_limits<float>::max();

  int to_check[10];
  int idx = 0;
  to_check[idx++] = root_node_idx;

  glm::vec3 inv_direction = 1.f / world_ray.direction;
  Ray obj_ray = {
      .origin = glm::vec3(geometry.inv_transform * glm::vec4(world_ray.origin, 1.f)),
      .direction = glm::vec3(geometry.inv_transform * glm::vec4(world_ray.direction, 0.f)),
  };

  while (idx > 0) {
    const bvh::Node& node = node_list[to_check[--idx]];

    // Hit leaf
    if (node.child_idx == -1) {
      Intersection curr = test_tri_list_isect(node.tri_idx, node.tri_idx + node.tri_count, obj_ray,
                                              triangle_list, position_list, normal_list);

      if (curr.t > 0.f && curr.t < t_min) {
        isect = std::move(curr);
        t_min = isect.t;
      }
    } else {
      const bvh::Node& c0 = node_list[node.child_idx];
      const bvh::Node& c1 = node_list[node.child_idx + 1];

      float t_result_c0 = c0.bbox.intersect(world_ray, inv_direction);
      float t_result_c1 = c1.bbox.intersect(world_ray, inv_direction);

      // Two things: first, we want to always look at the closer child BVH node first, because
      // it may mean we get to skip checking farther BVH nodes later. Secondly, we only
      // check the child nodes if it's closer than our current intersection.
      bool is_c0_nearer = t_result_c0 < t_result_c1;
      float near = is_c0_nearer ? t_result_c0 : t_result_c1;
      float far = is_c0_nearer ? t_result_c1 : t_result_c0;
      int child_idx_near = is_c0_nearer ? node.child_idx : node.child_idx + 1;
      int child_idx_far = is_c0_nearer ? node.child_idx + 1 : node.child_idx;

      if (far < t_min) to_check[idx++] = child_idx_far;
      if (near < t_min) to_check[idx++] = child_idx_near;
    }
  }

  if (isect.t > 0.f) {
    isect.normal = glm::normalize(glm::vec3(geometry.inv_transpose * glm::vec4(isect.normal, 0.f)));
    isect.point = glm::vec3(geometry.transform * glm::vec4(isect.point, 1.f));
    isect.material_id = geometry.material_id;
  }

  return isect;
}

namespace kernel {

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
                                   bool bvh_isect_culling) {
  int segment_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (segment_index >= num_paths) {
    return;
  }

  PathSegment segment = segments[segment_index];

  if (segment.remaining_bounces == 0) {
    return;
  }

  Ray segment_ray = segment.ray;
  glm::vec3 inv_direction = 1.f / segment_ray.direction;
  float t_min = cuda::std::numeric_limits<float>::max();

  Intersection isect;
  isect.t = -1.f;

  for (int geometry_index = 0; geometry_index < geometry_list_size; ++geometry_index) {
    const Geometry& geometry = geometry_list[geometry_index];

    if (bbox_isect_culling) {
      float t_result = geometry.bbox.intersect(segment_ray, inv_direction);

      if (cuda::std::isinf(t_result) || t_result >= t_min) continue;
    }

    Intersection curr_isect;

    switch (geometry.type) {
      case Geometry::Type::Cube:
        curr_isect = test_cube_isect(geometry, segment_ray);
        break;

      case Geometry::Type::Sphere:
        curr_isect = test_sphere_isect(geometry, segment_ray);
        break;

      case Geometry::Type::Gltf: {
        if (bvh_isect_culling && geometry.bvh_root_idx >= 0) {
          curr_isect = test_bvh_isect(geometry.bvh_root_idx, segment_ray, geometry, bvh_node_list,
                                      bvh_tri_list, position_list, normal_list);
        } else {
          curr_isect =
              test_gltf_isect(geometry, segment_ray, triangle_list, position_list, normal_list);
        }

        break;
      }

      default:
        // Unreachable
        return;
    }

    // Ray did not hit any geometry
    if (curr_isect.t < 0.f) {
      continue;
    }

    // Discovered a closer object, save it
    if (t_min > curr_isect.t) {
      t_min = curr_isect.t;
      isect = curr_isect;
    }
  }

  intersections[segment_index] = std::move(isect);
}

}  // namespace kernel
