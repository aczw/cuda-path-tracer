#pragma once

#include "kern_exec_config.hpp"
#include "scene.h"
#include "utilities.cuh"

#include <glm/glm.hpp>

void init_data_container(GuiDataContainer* gui_data);

class PathTracer {
 public:
  explicit PathTracer(glm::ivec2 resolution);

  void initialize(Scene* scene);
  void free();

  /**
   * Wrapper for the `__global__` call that sets up the kernel calls
   * and performs the path tracing
   */
  void run(uchar4* pbo, int curr_iter);

 private:
  KernExecConfig config_ray_gen;
  KernExecConfig config_isect;
  KernExecConfig config_shade;
  KernExecConfig config_gather;
  KernExecConfig config_send;
};
