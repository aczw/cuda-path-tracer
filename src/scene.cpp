#include "scene.hpp"

#include "utilities.cuh"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>

Opt<Settings> Scene::load_from_json(std::filesystem::path scene_file) {
  std::ifstream stream(scene_file.string().data());
  nlohmann::json root = nlohmann::json::parse(stream);

  // First load materials to build a mapping, then use that to load geometry
  if (!parse_geometry(root, parse_materials(root))) {
    return {};
  }

  // Parse camera data and other settings
  const auto& camera_data = root["Camera"];
  const auto& pos = camera_data["EYE"];
  const auto& lookat = camera_data["LOOKAT"];
  const auto& up = camera_data["UP"];

  camera.resolution = glm::ivec2(camera_data["RES"][0], camera_data["RES"][1]);
  camera.position = glm::vec3(pos[0], pos[1], pos[2]);
  camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);
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

Scene::MatNameIdMap Scene::parse_materials(const nlohmann::json& root) {
  const auto& materials_data = root["Materials"];
  MatNameIdMap mat_name_to_id;

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

    mat_name_to_id[name] = static_cast<char>(material_list.size());
    material_list.emplace_back(std::move(new_material));
  }

  return mat_name_to_id;
}

bool Scene::parse_geometry(const nlohmann::json& root, const MatNameIdMap& mat_name_to_id) {
  const auto& objects_data = root["Objects"];

  for (const auto& object : objects_data) {
    Geometry new_geometry;

    new_geometry.material_id = mat_name_to_id.at(object["MATERIAL"]);
    new_geometry.tri_begin = -1;
    new_geometry.tri_end = -1;
    new_geometry.bbox = {
        .min = glm::vec3(cuda::std::numeric_limits<float>::infinity()),
        .max = -glm::vec3(cuda::std::numeric_limits<float>::infinity()),
    };

    const auto& type = object["TYPE"];
    if (type == "cube") {
      new_geometry.type = Geometry::Type::Cube;
    } else if (type == "sphere") {
      new_geometry.type = Geometry::Type::Sphere;
    } else {
      new_geometry.type = Geometry::Type::Gltf;
      if (!parse_gltf(new_geometry, object["PATH"])) return false;
    }

    const auto& trans = object["TRANS"];
    const auto& rotat = object["ROTAT"];
    const auto& scale = object["SCALE"];
    new_geometry.translation = glm::vec3(trans[0], trans[1], trans[2]);
    new_geometry.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
    new_geometry.scale = glm::vec3(scale[0], scale[1], scale[2]);

    // Build transformation matrix
    glm::vec3 rotation_rad = new_geometry.rotation * (std::numbers::pi_v<float> / 180.f);
    glm::mat4 transform = glm::translate(glm::mat4(), new_geometry.translation);
    transform = glm::rotate(transform, rotation_rad.x, glm::vec3(1.f, 0.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.y, glm::vec3(0.f, 1.f, 0.f));
    transform = glm::rotate(transform, rotation_rad.z, glm::vec3(0.f, 0.f, 1.f));
    transform = glm::scale(transform, new_geometry.scale);

    new_geometry.transform = transform;
    new_geometry.inv_transform = glm::inverse(new_geometry.transform);
    new_geometry.inv_transpose = glm::inverseTranspose(new_geometry.transform);

    build_bounding_box(new_geometry);

    geometry_list.push_back(std::move(new_geometry));
  }

  return true;
}

bool Scene::parse_gltf(Geometry& geometry, std::filesystem::path gltf_file) {
  std::string file_name = gltf_file.filename().string();

  auto print_error = [&](std::string_view message) {
    std::cerr << std::format("[GLTF] Error: {}: {}\n", file_name, message.data());
  };

  if (!std::filesystem::exists(gltf_file)) {
    print_error("file does not exist");
    return false;
  }

  std::filesystem::path extension = gltf_file.extension();

  if (extension != ".gltf" && extension != ".glb") {
    print_error("not a .gltf/.glb file");
    return false;
  }

  using namespace tinygltf;

  Model model;

  bool result = [&]() -> bool {
    TinyGLTF loader;
    std::string error;
    std::string warning;
    std::string canonical = std::filesystem::canonical(gltf_file).string();

    std::cout << std::format("[GTLF] Loading \"{}\"\n", canonical);

    bool res = true;
    if (extension == ".gltf") {
      res = loader.LoadASCIIFromFile(&model, &error, &warning, canonical);
    } else {
      res = loader.LoadBinaryFromFile(&model, &error, &warning, canonical);
    }

    if (!warning.empty()) std::cout << std::format("[GLTF] Warning: {}\n", warning);
    if (!error.empty()) std::cout << std::format("[GLTF] Error: {}\n", error);

    return res;
  }();

  if (!result) return false;

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
    std::vector<int> indices;

    switch (idx_accessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        indices = reinterpret_indices_as<uint16_t>(model, idx_accessor);
        break;

      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        indices = reinterpret_indices_as<uint32_t>(model, idx_accessor);
        break;

      default:
        print_error("unknown vertex index component type");
        return false;
    };

    const Accessor& pos_accessor = model.accessors[primitive.attributes.at("POSITION")];
    if (pos_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      print_error("position component type is not a float");
      return false;
    }
    const float* raw_pos = collect_unique_positions(model, pos_accessor);

    const Accessor& nor_accessor = model.accessors[primitive.attributes.at("NORMAL")];
    if (nor_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      print_error("normal component type is not a float");
      return false;
    }
    const float* raw_nor = collect_unique_normals(model, nor_accessor);

    // Iterate over each triangle
    for (int i = 0; i < idx_accessor.count; i += 3) {
      Triangle triangle;

      // Iterate over each vertex in the triangle, and map to new index
      for (int j = i; j < i + 3; ++j) {
        int offset = indices[j] * 3;

        // Find the position data (which we added in the previous step) in the list
        // and use its index for this vertex
        glm::vec3 pos(raw_pos[offset], raw_pos[offset + 1], raw_pos[offset + 2]);
        if (auto it = std::ranges::find(position_list, pos); it != position_list.end()) {
          auto new_idx = std::ranges::distance(position_list.begin(), it);
          triangle[j - i].pos_idx = new_idx;
        } else {
          print_error("index referencing previously undiscovered vertex position");
          return false;
        }

        // Find the normal data (which we added in the previous step) in the list
        // and use its index for this vertex
        glm::vec3 nor(raw_nor[offset], raw_nor[offset + 1], raw_nor[offset + 2]);
        if (auto it = std::ranges::find(normal_list, nor); it != normal_list.end()) {
          auto new_idx = std::ranges::distance(normal_list.begin(), it);
          triangle[j - i].nor_idx = new_idx;
        } else {
          print_error("index referencing previously undiscovered vertex normal");
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

const float* Scene::collect_unique_positions(const tinygltf::Model& model,
                                             const tinygltf::Accessor& pos_accessor) {
  using namespace tinygltf;

  const BufferView& bv = model.bufferViews[pos_accessor.bufferView];
  const Buffer& buf = model.buffers[bv.buffer];
  auto pos = reinterpret_cast<const float*>(&buf.data[bv.byteOffset + pos_accessor.byteOffset]);

  for (int offset = 0; offset < pos_accessor.count; ++offset) {
    glm::vec3 position(pos[offset * 3], pos[offset * 3 + 1], pos[offset * 3 + 2]);

    if (auto it = std::ranges::find(position_list, position); it == position_list.end()) {
      position_list.push_back(std::move(position));
    }
  }

  return pos;
}

const float* Scene::collect_unique_normals(const tinygltf::Model& model,
                                           const tinygltf::Accessor& nor_accessor) {
  using namespace tinygltf;

  const BufferView& bv = model.bufferViews[nor_accessor.bufferView];
  const Buffer& buf = model.buffers[bv.buffer];
  auto nor = reinterpret_cast<const float*>(&buf.data[bv.byteOffset + nor_accessor.byteOffset]);

  for (int offset = 0; offset < nor_accessor.count; ++offset) {
    glm::vec3 normal(nor[offset * 3], nor[offset * 3 + 1], nor[offset * 3 + 2]);

    if (auto it = std::ranges::find(normal_list, normal); it == normal_list.end()) {
      normal_list.push_back(std::move(normal));
    }
  }

  return nor;
}

void build_bounding_box(Geometry& geometry) {
  Aabb& bbox = geometry.bbox;

  switch (geometry.type) {
    case Geometry::Type::Sphere:
    case Geometry::Type::Cube: {
      glm::vec3 world_min = glm::vec3(geometry.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
      glm::vec3 world_max = glm::vec3(geometry.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
      bbox.include(world_min);
      bbox.include(world_max);
      std::cout << std::format("geom min: [{}, {}, {}]\n", world_min.x, world_min.y, world_min.z);
      std::cout << std::format("geom max: [{}, {}, {}]\n", world_max.x, world_max.y, world_max.z);
      break;
    }

    case Geometry::Type::Gltf: {
      break;
    }

    default:
      break;
  }
}
