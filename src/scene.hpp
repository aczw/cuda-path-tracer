#pragma once

#include "camera.hpp"
#include "material.hpp"
#include "utilities.cuh"

#include <glm/glm.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/// A triangle is simply a trio of indices pointing at vertex attributes.
using Triangle = glm::ivec3;

struct Geometry {
  enum class Type { Sphere, Cube, Gltf } type;
  char material_id;

  std::vector<Triangle> triangles;
  std::vector<glm::vec3> positions;

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
  std::vector<std::unique_ptr<Geometry>> geometry_list;
  std::vector<Material> material_list;
};
