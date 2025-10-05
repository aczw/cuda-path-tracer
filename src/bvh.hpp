#pragma once

#include "geometry.hpp"

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
                  std::vector<Triangle>& tri_list) {
  if (depth == MAX_DEPTH) return;
}

}  // namespace bvh
