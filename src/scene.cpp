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

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename) {
  cout << "Reading scene from " << filename << " ..." << endl;
  cout << " " << endl;
  auto ext = filename.substr(filename.find_last_of('.'));
  if (ext == ".json") {
    loadFromJSON(filename);
    return;
  } else {
    cout << "Couldn't read from " << filename << endl;
    exit(-1);
  }
}

void Scene::loadFromJSON(const std::string& jsonName) {
  std::ifstream f(jsonName);
  json data = json::parse(f);
  const auto& materialsData = data["Materials"];
  std::unordered_map<std::string, uint32_t> MatNameToID;
  for (const auto& item : materialsData.items()) {
    const auto& name = item.key();
    const auto& p = item.value();
    Material newMaterial{};
    // TODO: handle materials loading differently
    if (p["TYPE"] == "Diffuse") {
      const auto& col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
    } else if (p["TYPE"] == "Emitting") {
      const auto& col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
      newMaterial.emittance = p["EMITTANCE"];
    } else if (p["TYPE"] == "Specular") {
      const auto& col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
    }
    MatNameToID[name] = materials.size();
    materials.emplace_back(newMaterial);
  }
  const auto& objectsData = data["Objects"];
  for (const auto& p : objectsData) {
    const auto& type = p["TYPE"];
    Geometry newGeom;
    if (type == "cube") {
      newGeom.type = Geometry::Type::Cube;
    } else {
      newGeom.type = Geometry::Type::Sphere;
    }
    newGeom.material_id = MatNameToID[p["MATERIAL"]];
    const auto& trans = p["TRANS"];
    const auto& rotat = p["ROTAT"];
    const auto& scale = p["SCALE"];
    newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
    newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
    newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

    glm::vec3 rotation_rad = newGeom.rotation * (std::numbers::pi_v<float> / 180.f);

    glm::mat4 transform = glm::translate(glm::mat4(), newGeom.translation);
    transform = glm::rotate(transform, rotation_rad.x, glm::vec3(1.f, 0.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.y, glm::vec3(0.f, 1.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.z, glm::vec3(0.f, 0.f, 1.f));
    transform = glm::scale(transform, newGeom.scale);

    newGeom.transform = transform;
    newGeom.inv_transform = glm::inverse(newGeom.transform);
    newGeom.inv_transpose = glm::inverseTranspose(newGeom.transform);

    geoms.push_back(newGeom);
  }
  const auto& cameraData = data["Camera"];
  Camera& camera = state.camera;
  RenderState& state = this->state;
  camera.resolution.x = cameraData["RES"][0];
  camera.resolution.y = cameraData["RES"][1];
  float fovy = cameraData["FOVY"];
  state.total_iterations = cameraData["ITERATIONS"];
  state.trace_depth = cameraData["DEPTH"];
  state.image_name = cameraData["FILE"];
  const auto& pos = cameraData["EYE"];
  const auto& lookat = cameraData["LOOKAT"];
  const auto& up = cameraData["UP"];
  camera.position = glm::vec3(pos[0], pos[1], pos[2]);
  camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
  camera.up = glm::vec3(up[0], up[1], up[2]);

  // calculate fov based on resolution
  float yscaled = tan(fovy * (std::numbers::pi / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / std::numbers::pi;
  camera.fov = glm::vec2(fovx, fovy);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength =
      glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
