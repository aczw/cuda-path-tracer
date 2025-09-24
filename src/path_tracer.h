#pragma once

#include "scene.h"
#include "utilities.cuh"

#include <glm/glm.hpp>

void init_data_container(GuiDataContainer* gui_data);

struct PathTracer {
  void initialize(Scene* scene);
  void free();

  /**
   * Wrapper for the `__global__` call that sets up the kernel calls
   * and performs path tracing for a singular iteration.
   */
  void run_iteration(uchar4* pbo, int curr_iter);
};
