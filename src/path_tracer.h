#pragma once

#include "scene.h"
#include "utilities.h"

void init_data_container(GuiDataContainer* gui_data);

namespace path_tracer {

void initialize(Scene* scene);
void free();

/**
 * Wrapper for the `__global__` call that sets up the kernel calls and performs
 * the path tracing
 */
void run(uchar4* pbo, int curr_iteration);

}  // namespace path_tracer
