#include "scene.h"

#include "json.hpp"
#include "utilities.cuh"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

Scene::Scene(std::string file_name) {
  std::cout << "Reading scene \"" << file_name << "\"" << std::endl;

  std::string extension = file_name.substr(file_name.find_last_of('.'));

  if (extension == ".json") {
    load_from_json(file_name);
    return;
  } else {
    std::cout << "\"" << file_name << "\" is an invalid scene, not a JSON file" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void Scene::load_from_json(const std::string& json_name) {
  std::ifstream f(json_name);
  json data = json::parse(f);

  const auto& materials_data = data["Materials"];
  std::unordered_map<std::string, char> material_name_to_id;

  for (const auto& item : materials_data.items()) {
    const auto& name = item.key();
    const auto& object = item.value();
    Material new_material{};

    // TODO: handle materials loading differently
    if (object["TYPE"] == "Diffuse") {
      const auto& col = object["RGB"];
      new_material.color = glm::vec3(col[0], col[1], col[2]);
    } else if (object["TYPE"] == "Emitting") {
      const auto& col = object["RGB"];
      new_material.color = glm::vec3(col[0], col[1], col[2]);
      new_material.emittance = object["EMITTANCE"];
    } else if (object["TYPE"] == "Specular") {
      const auto& col = object["RGB"];
      new_material.color = glm::vec3(col[0], col[1], col[2]);
    }

    material_name_to_id[name] = material_list.size();
    material_list.emplace_back(new_material);
  }

  const auto& objects_data = data["Objects"];
  for (const auto& object : objects_data) {
    Geometry new_geometry;
    new_geometry.material_id = material_name_to_id[object["MATERIAL"]];

    const auto& type = object["TYPE"];
    if (type == "cube") {
      new_geometry.type = Geometry::Type::Cube;
    } else {
      new_geometry.type = Geometry::Type::Sphere;
    }

    const auto& trans = object["TRANS"];
    const auto& rotat = object["ROTAT"];
    const auto& scale = object["SCALE"];
    new_geometry.translation = glm::vec3(trans[0], trans[1], trans[2]);
    new_geometry.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
    new_geometry.scale = glm::vec3(scale[0], scale[1], scale[2]);

    glm::vec3 rotation_rad = new_geometry.rotation * (std::numbers::pi_v<float> / 180.f);

    glm::mat4 transform = glm::translate(glm::mat4(), new_geometry.translation);
    transform = glm::rotate(transform, rotation_rad.x, glm::vec3(1.f, 0.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.y, glm::vec3(0.f, 1.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.z, glm::vec3(0.f, 0.f, 1.f));
    transform = glm::scale(transform, new_geometry.scale);

    new_geometry.transform = transform;
    new_geometry.inv_transform = glm::inverse(new_geometry.transform);
    new_geometry.inv_transpose = glm::inverseTranspose(new_geometry.transform);

    geometry_list.push_back(new_geometry);
  }

  const auto& camera_data = data["Camera"];
  Camera& camera = state.camera;
  RenderState& state = this->state;

  camera.resolution.x = camera_data["RES"][0];
  camera.resolution.y = camera_data["RES"][1];
  float fovy = camera_data["FOVY"];
  state.total_iterations = camera_data["ITERATIONS"];
  state.trace_depth = camera_data["DEPTH"];
  state.image_name = camera_data["FILE"];
  const auto& pos = camera_data["EYE"];
  const auto& lookat = camera_data["LOOKAT"];
  const auto& up = camera_data["UP"];
  camera.position = glm::vec3(pos[0], pos[1], pos[2]);
  camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);
  camera.up = glm::vec3(up[0], up[1], up[2]);

  // Calculate FOV based on resolution
  float y_scaled = tan(fovy * (std::numbers::pi / 180));
  float x_scaled = (y_scaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(x_scaled) * 180) / std::numbers::pi;
  camera.fov = glm::vec2(fovx, fovy);
  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixel_length = glm::vec2(2 * x_scaled / static_cast<float>(camera.resolution.x),
                                  2 * y_scaled / static_cast<float>(camera.resolution.y));
  camera.view = glm::normalize(camera.look_at - camera.position);

  // Set up render camera stuff
  state.image.resize(camera.resolution.x * camera.resolution.y);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
