#pragma once

#include "scene.hpp"

struct GuiData {
  Settings* settings;

  bool sort_paths_by_material;
  bool discard_oob_paths;
  bool discard_light_isect_paths;
  bool stochastic_sampling;
  bool apply_tone_mapping;
};
