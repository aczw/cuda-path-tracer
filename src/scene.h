#pragma once

#include "scene_structs.h"

#include <vector>

class Scene {
 private:
  void load_from_json(const std::string& json_name);

 public:
  Scene(std::string file_name);

  std::vector<Geometry> geometry_list;
  std::vector<Material> material_list;
  RenderState state;
};
