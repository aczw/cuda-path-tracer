#include "scene.hpp"

#include "json.hpp"
#include "utilities.cuh"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <unordered_map>

Opt<Settings> Scene::load_from_json(std::filesystem::path scene_file) {
  std::ifstream stream(scene_file.string().data());
  nlohmann::json root = nlohmann::json::parse(stream);

  std::unordered_map<std::string, char> material_name_to_id;

  // Parse material data
  const auto& materials_data = root["Materials"];
  for (const auto& item : materials_data.items()) {
    const auto& name = item.key();
    const auto& object = item.value();

    Material new_material = {.type = Material::Type::Unknown};

    // Color is common across all materials
    const auto& color_value = object["RGB"];
    glm::vec3 color = glm::vec3(color_value[0], color_value[1], color_value[2]);

    const auto& material_type = object["TYPE"];
    if (material_type == "Diffuse") {
      new_material = {.type = Material::Type::Diffuse, .color = color};
    } else if (material_type == "Emitting") {
      new_material = {
          .type = Material::Type::Light,
          .color = color,
          .emission = object["EMITTANCE"],
      };
    } else if (material_type == "PureReflection") {
      new_material = {.type = Material::Type::PureReflection, .color = color};
    } else if (material_type == "PureTransmission") {
      new_material = {
          .type = Material::Type::PureTransmission,
          .color = color,
          .eta = object["ETA"],
      };
    } else if (material_type == "PerfectSpecular") {
      new_material = {
          .type = Material::Type::PerfectSpecular,
          .color = color,
          .eta = object["ETA"],
      };
    }

    material_name_to_id[name] = static_cast<char>(material_list.size());
    material_list.emplace_back(std::move(new_material));
  }

  // Parse geometry data
  const auto& objects_data = root["Objects"];
  for (const auto& object : objects_data) {
    Geometry new_geometry;

    new_geometry.material_id = material_name_to_id[object["MATERIAL"]];
    new_geometry.tri_begin = -1;
    new_geometry.tri_end = -1;

    const auto& type = object["TYPE"];
    if (type == "cube") {
      new_geometry.type = Geometry::Type::Cube;
    } else if (type == "sphere") {
      new_geometry.type = Geometry::Type::Sphere;
    } else {
      new_geometry.type = Geometry::Type::Gltf;

      std::filesystem::path gltf_path = object["PATH"];
      std::string file_name = gltf_path.filename().string();

      auto print_error = [&](std::string_view message) {
        std::cerr << std::format("[GLTF] Error: {}: {}\n", file_name, message.data());
      };

      if (!std::filesystem::exists(gltf_path)) {
        print_error("file does not exist");
        return {};
      }

      std::filesystem::path extension = gltf_path.extension();
      if (extension != ".gltf" && extension != ".glb") {
        print_error("not a .gltf/.glb file");
        return {};
      }

      tinygltf::Model model;
      std::string error;
      std::string warning;

      bool result = [&]() -> bool {
        tinygltf::TinyGLTF loader;
        std::string canonical = std::filesystem::canonical(gltf_path).string();

        std::cout << std::format("[GTLF] Loading \"{}\"\n", canonical);

        if (extension == ".gltf") {
          return loader.LoadASCIIFromFile(&model, &error, &warning, canonical);
        } else {
          return loader.LoadBinaryFromFile(&model, &error, &warning, canonical);
        }
      }();

      if (!warning.empty()) {
        std::cout << std::format("[GLTF] Warning: {}\n", warning);
      }

      if (!error.empty()) {
        std::cout << std::format("[GLTF] Error: {}\n", error);
      }

      if (!result) {
        return {};
      }

      if (!try_load_gltf_into_geometry(new_geometry, model, print_error)) {
        return {};
      }
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

    geometry_list.push_back(std::move(new_geometry));
  }

  // Parse camera data and other settings
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
  float y_scaled = std::tan(fov_y * (std::numbers::pi / 180));
  float x_scaled = (y_scaled * camera.resolution.x) / camera.resolution.y;
  float fov_x = (std::atan(x_scaled) * 180) / std::numbers::pi;
  camera.fov = glm::vec2(fov_x, fov_y);
  camera.pixel_length = glm::vec2(2 * x_scaled / static_cast<float>(camera.resolution.x),
                                  2 * y_scaled / static_cast<float>(camera.resolution.y));

  return Settings{
      .max_iterations = camera_data["ITERATIONS"],
      .max_depth = camera_data["DEPTH"],
      .original_camera = camera,
      .scene_name = scene_file.stem().string(),
  };
};

bool Scene::try_load_gltf_into_geometry(Geometry& geometry,
                                        const tinygltf::Model& model,
                                        std::function<void(std::string_view)> print_error) {
  using namespace tinygltf;

  const std::vector<Mesh>& meshes = model.meshes;

  if (meshes.empty()) {
    print_error("no meshes to render");
    return false;
  }

  const Mesh& first_mesh = meshes[0];

  if (first_mesh.primitives.empty()) {
    print_error("no primitives in the mesh to render");
    return false;
  }

  // Since there is one global list of triangle data, this geometry "slices" into it
  // with a begin and end index, similar to an iterator
  int tri_begin = triangle_list.size();

  for (const Primitive& primitive : first_mesh.primitives) {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
      print_error("mesh primitive is not a triangle");
      return false;
    }

    if (primitive.indices == -1) {
      print_error("mesh primitive does not specify vertex indices");
      return false;
    }

    const Accessor& idx_accessor = model.accessors[primitive.indices];
    const Accessor& pos_accessor = model.accessors[primitive.attributes.at("POSITION")];

    if (idx_accessor.type != TINYGLTF_TYPE_SCALAR ||
        idx_accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      print_error("vertex indices are not scalars (uint16_t)");
      return false;
    }

    if (pos_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      print_error("position component type is not a float");
      return false;
    }

    const BufferView& pos_bv = model.bufferViews[pos_accessor.bufferView];
    const Buffer& pos_buffer = model.buffers[pos_bv.buffer];
    const float* positions = reinterpret_cast<const float*>(
        &pos_buffer.data[pos_bv.byteOffset + pos_accessor.byteOffset]);

    const BufferView& idx_bv = model.bufferViews[idx_accessor.bufferView];
    const Buffer& idx_buffer = model.buffers[idx_bv.buffer];
    const uint16_t* indices = reinterpret_cast<const uint16_t*>(
        &idx_buffer.data[idx_bv.byteOffset + idx_accessor.byteOffset]);

    // Collect unique positions into global geometry list. We will reference their indices later
    for (int offset = 0; offset < pos_accessor.count; ++offset) {
      glm::vec3 pos(positions[offset * 3], positions[offset * 3 + 1], positions[offset * 3 + 2]);

      if (auto it = std::ranges::find(position_list, pos); it == position_list.end()) {
        position_list.push_back(std::move(pos));
      }
    }

    // Iterate over each triangle
    for (int i = 0; i < idx_accessor.count; i += 3) {
      Triangle triangle;

      // Iterate over each vertex in the triangle, and map to new index
      for (int j = i; j < i + 3; ++j) {
        int old = indices[j];
        glm::vec3 pos(positions[old * 3], positions[old * 3 + 1], positions[old * 3 + 2]);

        // Find the position data (which we added in the previous step) in the list
        // and use its index for this vertex
        if (auto it = std::ranges::find(position_list, pos); it != position_list.end()) {
          auto new_idx = std::ranges::distance(position_list.begin(), it);
          triangle[j - i] = new_idx;
        } else {
          print_error("index referencing previously undiscovered vertex position");
          return false;
        }
      }

      triangle_list.push_back(std::move(triangle));
    }
  }

  // Now that we've finished populating the triangle list, get an index to the ending
  // boundary of this geometry's data
  int tri_end = triangle_list.size();
  geometry.tri_begin = tri_begin;
  geometry.tri_end = tri_end;

  return true;
}
