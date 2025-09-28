#pragma once

#include "ImGui/imgui.h"
#include "gui_data.hpp"
#include "image.hpp"
#include "scene.hpp"

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/// Stores information about the input at a given frame.
struct InputBundle {
  bool left_mouse_pressed;
  bool right_mouse_pressed;
  bool middle_mouse_pressed;
  bool mouse_over_gui;
};

class RenderContext {
 public:
  RenderContext();

  bool try_open_scene(std::filesystem::path scene_file);
  void save_image(Image::Format format = Image::Format::PNG) const;

  /// Convenience getter for camera width.
  int get_width() const;

  /// Convenience getter for camera height.
  int get_height() const;

  /// Retrieves pointer to GUI data.
  GuiData* get_gui_data() const;

  Scene scene;
  Settings settings;
  std::string start_time;

  std::vector<glm::vec3> image;
  int curr_iteration;

  InputBundle input;
  float zoom, theta, phi;
  double last_cursor_x, last_cursor_y;

  GLuint pbo;
  GLuint display_image;

  std::unique_ptr<GuiData> gui_data;
};
