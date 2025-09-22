#pragma once

#include "scene.h"
#include "utilities.h"

void init_data_container(GuiDataContainer* gui_data);
void path_trace_init(Scene* scene);
void path_trace_free();
void path_trace(uchar4* pbo, int frame, int curr_iteration);
