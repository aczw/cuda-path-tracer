#pragma once

#include "gui_data.hpp"
#include "scene.h"
#include "utilities.cuh"

#include <glm/glm.hpp>

class PathTracer {
 public:
  explicit PathTracer(GuiData* gui_data);

  void initialize(Scene* scene);
  void free();

  /**
   * Wrapper for the `__global__` call that sets up the kernel calls
   * and performs path tracing for a singular iteration.
   */
  void run_iteration(uchar4* pbo, int curr_iter);

 private:
  GuiData* gui_data;
};
