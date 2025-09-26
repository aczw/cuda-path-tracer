#pragma once

struct GuiData {
  int max_depth;
  bool sort_paths_by_material;
  bool discard_oob_paths;
  bool discard_light_isect_paths;
  bool stochastic_sampling;
  bool apply_tone_mapping;
};
