#pragma once

#include "aabb.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <vector>

namespace bvh {

struct Node {
  Aabb bbox;

  /// Index into the global triangle list to access the triangles held by this node.
  int tri_idx;

  /// Triangles for a node are contiguous in the list, so we simply keep track of the count.
  int tri_count;

  /// Index into the global BVH node list to access the children nodes. Note that we only need to
  /// store one child index because the other is simply located at `child_idx + 1`.
  int child_idx;
};

constexpr int MAX_DEPTH = 10;

inline void split(int parent_node_idx,
                  int depth,
                  std::vector<Node>& node_list,
                  std::vector<Triangle>& tri_list,
                  const std::vector<glm::vec3>& positions,
                  const glm::mat4 transform) {
  if (depth == MAX_DEPTH) return;

  // Find axis to split along. Pick the axis by choosing the one with the biggest length
  glm::vec3 size = node_list[parent_node_idx].bbox.get_size();
  int axis = size.x > glm::max(size.y, size.z) ? 0 : size.y > size.z ? 1 : 2;
  float center_pos = node_list[parent_node_idx].bbox.get_center()[axis];

  // Since we haven't reached the max depth, we further split this node
  node_list[parent_node_idx].child_idx = node_list.size();
  int parent_tri_idx = node_list[parent_node_idx].tri_idx;

  node_list.emplace_back(Node{.tri_idx = parent_tri_idx, .child_idx = -1});
  node_list.emplace_back(Node{.tri_idx = parent_tri_idx, .child_idx = -1});

  // Iterators may have been invalidated, so get a new reference to parent
  Node& c0 = node_list[node_list[parent_node_idx].child_idx];
  Node& c1 = node_list[node_list[parent_node_idx].child_idx + 1];

  for (int tri_idx = parent_tri_idx;
       tri_idx < parent_tri_idx + node_list[parent_node_idx].tri_count; ++tri_idx) {
    const Triangle& triangle = tri_list[tri_idx];

    // Use center of triangle to determine which side of the parent it should be on
    float tri_axis_center = triangle.get_center(positions)[axis];
    bool use_side0 = tri_axis_center < center_pos;
    Node& child = use_side0 ? c0 : c1;

    // Add this triangle to the child's group
    child.bbox.include(triangle, positions, transform);
    child.tri_count++;

    if (use_side0) {
      // First, get the triangle index of the most recently added triangle. We want to add
      // the current tri_idx's triangle data to this index instead.
      int tri_idx_to_use = child.tri_idx + child.tri_count - 1;

      // Then, we make sure that this index is pointing at the current triangle data that we
      // added. This can easily be done by swapping the contents of the index we want, and
      // the current index. This will also correctly place side1's triangles.
      std::swap(tri_list[tri_idx_to_use], tri_list[tri_idx]);

      // Triangles belonging to side1 are stored "after" the triangles on side0. This means
      // that everytime we add a triangle to side0, the starting triangle index for side1
      // must be incremented to account for the additional offset.
      c1.tri_idx++;
    }
  }

  int parent_child_idx = node_list[parent_node_idx].child_idx;
  split(parent_child_idx, depth + 1, node_list, tri_list, positions, transform);
  split(parent_child_idx + 1, depth + 1, node_list, tri_list, positions, transform);
}

}  // namespace bvh
