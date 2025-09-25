#pragma once

#include "gui_data.hpp"
#include "render_context.hpp"

#include <glm/glm.hpp>

#include <vector_types.h>

class PathTracer {
 public:
  explicit PathTracer(RenderContext* ctx, GuiData* gui_data);

  void initialize();
  void free();

  /**
   * Wrapper for the `__global__` call that sets up the kernel calls
   * and performs path tracing for a singular iteration.
   */
  void run_iteration(uchar4* pbo, int curr_iter);

 private:
  RenderContext* ctx;
  GuiData* gui_data;
};
