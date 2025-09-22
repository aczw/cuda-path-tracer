#pragma once

#include "scene_structs.h"

#include <vector>

class Scene {
 private:
  void loadFromJSON(const std::string& jsonName);

 public:
  Scene(std::string filename);

  std::vector<Geometry> geoms;
  std::vector<Material> materials;
  RenderState state;
};
