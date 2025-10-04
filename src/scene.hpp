#pragma once

#include "camera.hpp"
#include "material.hpp"
#include "utilities.cuh"

#include <cuda/std/array>

#include <glm/glm.hpp>
#include <tiny_gltf.h>

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

struct Vertex {
  int pos_idx;
  int nor_idx;
};

/// A triangle is simply a trio of vertices containing the indices
/// pointing at various vertex attributes.
using Triangle = cuda::std::array<Vertex, 3>;

struct Geometry {
  enum class Type { Sphere, Cube, Gltf } type;

  char material_id;
  int tri_begin;
  int tri_end;

  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inv_transform;
  glm::mat4 inv_transpose;
};

struct Settings {
  int max_iterations;
  int max_depth;
  Camera original_camera;
  std::string scene_name;
};

class Scene {
 public:
  Opt<Settings> load_from_json(std::filesystem::path scene_file);

  Camera camera;

  std::vector<Geometry> geometry_list;
  std::vector<Material> material_list;

  /// There is one single global list containing all triangles in the scene.
  std::vector<Triangle> triangle_list;

  /// Global list of position data. Accessed via indices contained in triangles.
  std::vector<glm::vec3> position_list;
  std::vector<glm::vec3> normal_list;

 private:
  bool try_load_gltf_into_geometry(Geometry& geometry,
                                   const tinygltf::Model& model,
                                   std::function<void(std::string_view)> print_error);
};
