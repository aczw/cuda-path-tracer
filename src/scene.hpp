#pragma once

#include "bvh.hpp"
#include "camera.hpp"
#include "geometry.hpp"
#include "json.hpp"
#include "material.hpp"
#include "utilities.cuh"

#include <glm/glm.hpp>
#include <tiny_gltf.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

/// Helper for converting glTF vertex indices of any component type to a vector of integers.
template <class T>
std::vector<int> reinterpret_indices_as(const tinygltf::Model& model,
                                        const tinygltf::Accessor& idx_accessor) {
  using namespace tinygltf;

  const BufferView& idx_bv = model.bufferViews[idx_accessor.bufferView];
  const Buffer& idx_buffer = model.buffers[idx_bv.buffer];

  std::vector<int> indices;
  const unsigned char* begin = &idx_buffer.data[idx_bv.byteOffset + idx_accessor.byteOffset];
  const T* temp = reinterpret_cast<const T*>(begin);

  for (int i = 0; i < idx_accessor.count; ++i) {
    indices.push_back(static_cast<int>(temp[i]));
  }

  return indices;
}

struct Settings {
  int max_iterations;
  int max_depth;
  Camera original_camera;
  std::string scene_name;
  bool built_bvh;
};

class Scene {
 public:
  using MatNameIdMap = std::unordered_map<std::string, char>;

  Opt<Settings> load_from_json(std::filesystem::path scene_file);

  Camera camera;

  std::vector<Geometry> geometry_list;
  std::vector<Material> material_list;

  /// There is one single global list containing all triangles in the scene. Only relevant for
  /// geometry imported from models such as glTF.
  std::vector<Triangle> triangle_list;

  /// Global list containing all nodes in the BVH.
  std::vector<bvh::Node> bvh_node_list;
  std::vector<Triangle> bvh_tri_list;

  /// Global list of position data. Accessed via indices contained in triangles.
  std::vector<glm::vec3> position_list;

  /// Global list of normal data. Accessed via indices contained in triangles.
  std::vector<glm::vec3> normal_list;

 private:
  MatNameIdMap parse_materials(const nlohmann::json& root);
  Opt<bool> parse_geometry(const nlohmann::json& root, const MatNameIdMap& mat_name_to_id);

  /// Attempt to load and parse a glTF model data into the geometry.
  bool parse_gltf(Geometry& geometry, std::filesystem::path gltf_file);

  /// Collect unique positions into global geometry list. We will reference their indices later.
  const float* collect_unique_positions(const tinygltf::Model& model,
                                        const tinygltf::Accessor& pos_accessor);

  // Collect unique normals into global geometry list. We will reference their indices later.
  const float* collect_unique_normals(const tinygltf::Model& model,
                                      const tinygltf::Accessor& nor_accessor);

  /// Builds the bounding box for the corresponding geometry, and stores it.
  void build_bounding_box(Geometry& geometry);

  /// Builds a BVH for the corresponding geometry, and stores it.
  void build_bvh_tree(Geometry& geometry);
};
