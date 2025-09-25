#pragma once

#include "scene_structs.h"

#include <vector>

class Scene {
 private:
  void load_from_json(const std::string& jsonName);

 public:
  Scene(std::string filename);

  std::vector<Geometry> geoms;
  std::vector<Material> materials;
  RenderState state;
};
