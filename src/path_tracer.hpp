#pragma once

#include "gui_data.hpp"
#include "intersection.hpp"
#include "path_segment.hpp"
#include "render_context.hpp"

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <glm/glm.hpp>

#include <vector_types.h>

class PathTracer {
 public:
  using ZipTuple = thrust::tuple<Intersection, PathSegment>;
  using PathSegmentThrustPtr = thrust::device_ptr<PathSegment>;
  using IntersectionThrustPtr = thrust::device_ptr<Intersection>;

  explicit PathTracer(RenderContext* ctx);

  void initialize();
  void free();

  /// Runs the path tracer for one iteration i.e. sample.
  void run_iteration(uchar4* pbo, int curr_iter);

 private:
  RenderContext* ctx;

  glm::vec3* dev_image;

  Geometry* dev_geometry_list;
  Material* dev_material_list;

  Triangle* dev_triangle_list;
  bvh::Node* dev_bvh_node_list;
  Triangle* dev_bvh_tri_list;

  glm::vec3* dev_position_list;
  glm::vec3* dev_normal_list;

  PathSegment* dev_segments;
  Intersection* dev_intersections;

  /// This will always point to the front of the iterator and can therefore be safely stored.
  thrust::zip_iterator<thrust::tuple<IntersectionThrustPtr, PathSegmentThrustPtr>> begin;

  const int max_depth;
  const int num_pixels;
  const int num_blocks_64;
  const int num_blocks_128;

  static constexpr int BLOCK_SIZE_64 = 64;
  static constexpr int BLOCK_SIZE_128 = 128;
};
