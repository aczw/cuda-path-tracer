#pragma once

#include "camera.hpp"
#include "material.hpp"

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

struct Geometry {
  enum class Type { Sphere, Cube } type;
  char material_id;

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
  Settings load_from_json(std::filesystem::path scene_file);

  Camera camera;
  std::vector<Geometry> geometry_list;
  std::vector<Material> material_list;
};
