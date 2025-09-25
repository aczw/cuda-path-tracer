#pragma once

#include "render_context.hpp"

#include <GLFW/glfw3.h>

/// Manages the GLFW context and window.
class Window {
 public:
  explicit Window(RenderContext* ctx);
  ~Window();

  bool try_init();

  /// Retrieves the underlying `GLFWWindow` pointer.
  GLFWwindow* get();

 private:
  GLFWwindow* window;
  RenderContext* ctx;
};
