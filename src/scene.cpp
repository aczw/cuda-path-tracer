#include "scene.hpp"

#include "json.hpp"
#include "utilities.cuh"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <unordered_map>

Settings Scene::load_from_json(std::string_view scene_file) {
  std::ifstream stream(scene_file.data());
  nlohmann::json root = nlohmann::json::parse(stream);

  std::unordered_map<std::string, char> material_name_to_id;

  const auto& materials_data = root["Materials"];
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

  const auto& objects_data = root["Objects"];
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

  const auto& camera_data = root["Camera"];
  camera.resolution = glm::ivec2(camera_data["RES"][0], camera_data["RES"][1]);

  const auto& pos = camera_data["EYE"];
  camera.position = glm::vec3(pos[0], pos[1], pos[2]);

  const auto& lookat = camera_data["LOOKAT"];
  camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);

  const auto& up = camera_data["UP"];
  camera.up = glm::vec3(up[0], up[1], up[2]);

  camera.view = glm::normalize(camera.look_at - camera.position);
  camera.right = glm::normalize(glm::cross(camera.view, camera.up));

  // Calculate FOV based on resolution
  float fov_y = camera_data["FOVY"];
  float y_scaled = tan(fov_y * (std::numbers::pi / 180));
  float x_scaled = (y_scaled * camera.resolution.x) / camera.resolution.y;
  float fov_x = (atan(x_scaled) * 180) / std::numbers::pi;
  camera.fov = glm::vec2(fov_x, fov_y);
  camera.pixel_length = glm::vec2(2 * x_scaled / static_cast<float>(camera.resolution.x),
                                  2 * y_scaled / static_cast<float>(camera.resolution.y));

  return {
      .max_iterations = camera_data["ITERATIONS"],
      .max_depth = camera_data["DEPTH"],
      .original_camera = camera,
      .output_image_name = camera_data["FILE"],
  };
};
