#pragma once

#include "gui_data.hpp"
#include "intersection.cuh"
#include "path_segment.hpp"
#include "render_context.hpp"

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <glm/glm.hpp>

#include <vector_types.h>

class PathTracer {
 public:
  using PathSegmentPtr = thrust::device_ptr<PathSegment>;
  using IntersectionPtr = thrust::device_ptr<Intersection>;

  using ZipIteratorTuple = thrust::tuple<IntersectionPtr, PathSegmentPtr>;
  using ZipTuple = thrust::tuple<Intersection, PathSegment>;

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
  PathSegment* dev_segments;
  Intersection* dev_intersections;

  PathSegmentPtr tdp_segments;
  IntersectionPtr tdp_intersections;

  thrust::zip_iterator<ZipIteratorTuple> zip_begin;
  thrust::zip_iterator<ZipIteratorTuple> zip_end;

  int num_blocks_64;
  int num_blocks_128;

  static constexpr int BLOCK_SIZE_64 = 64;
  static constexpr int BLOCK_SIZE_128 = 128;
};
