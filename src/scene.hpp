#pragma once

#include "camera.hpp"

#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <vector>

struct Geometry {
  enum class Type { Sphere, Cube };

  Type type;
  char material_id;

  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inv_transform;
  glm::mat4 inv_transpose;
};

struct Material {
  glm::vec3 color;

  struct {
    float exponent;
    glm::vec3 color;
  } specular;

  float has_reflective;
  float has_refractive;
  float index_of_refraction;
  float emittance;
};

struct Settings {
  int max_iterations;
  int max_depth;
  Camera original_camera;
  std::string output_image_name;
};

class Scene {
 public:
  Settings load_from_json(std::string_view scene_file);

  Camera camera;
  std::vector<Geometry> geometry_list;
  std::vector<Material> material_list;
};
