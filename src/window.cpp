#include "window.hpp"

#include <format>
#include <iostream>
#include <numbers>

namespace callback {

void on_error(int error, const char* description) {
  std::cerr << std::format("[GLFW] Error: {}", description) << std::endl;
}

void on_key(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action != GLFW_PRESS) {
    return;
  }

  RenderContext* ctx = static_cast<RenderContext*>(glfwGetWindowUserPointer(window));

  switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;

    case GLFW_KEY_S:
      ctx->save_image();
      break;

    case GLFW_KEY_SPACE:
      ctx->scene.camera = ctx->settings.original_camera;
      // TODO(aczw): reset image_data as well? reset
      break;
  }
}

void on_cursor_pos(GLFWwindow* window, double xpos, double ypos) {
  RenderContext* ctx = static_cast<RenderContext*>(glfwGetWindowUserPointer(window));

  double last_x = ctx->last_cursor_x;
  double last_y = ctx->last_cursor_y;

  // Otherwise, clicking back into window causes re-start
  if (xpos == last_x || ypos == last_y) {
    return;
  }

  const InputBundle& input = ctx->input;
  Camera& camera = ctx->scene.camera;
  int width = camera.resolution.x;
  int height = camera.resolution.y;

  if (input.left_mouse_pressed) {
    // Compute new camera parameters
    ctx->phi -= (xpos - last_x) / width;
    ctx->theta -= (ypos - last_y) / height;
    ctx->theta = std::fmax(0.001f, std::fmin(ctx->theta, std::numbers::pi));
  } else if (input.right_mouse_pressed) {
    ctx->zoom += (ypos - last_y) / height;
    ctx->zoom = std::fmax(0.1f, ctx->zoom);
  } else if (input.middle_mouse_pressed) {
    glm::vec3 forward(camera.view.x, 0.f, camera.view.z);
    forward = glm::normalize(forward);

    glm::vec3 right(camera.right.x, 0.f, camera.right.z);
    right = glm::normalize(right);

    camera.look_at -= static_cast<float>(xpos - last_x) * right * 0.01f;
    camera.look_at += static_cast<float>(ypos - last_y) * forward * 0.01f;
  }

  ctx->last_cursor_x = xpos;
  ctx->last_cursor_y = ypos;
}

void on_mouse_button(GLFWwindow* window, int button, int action, int mods) {
  RenderContext* ctx = static_cast<RenderContext*>(glfwGetWindowUserPointer(window));
  InputBundle& input = ctx->input;

  if (input.mouse_over_gui) {
    return;
  }

  input.left_mouse_pressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  input.right_mouse_pressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  input.middle_mouse_pressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

}  // namespace callback

Window::Window(RenderContext* ctx) : window(nullptr), ctx(ctx) {}

Window::~Window() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

bool Window::try_init() {
  glfwSetErrorCallback(callback::on_error);

  if (!glfwInit()) {
    return false;
  }

  int width = ctx->get_width();
  int height = ctx->get_height();

  std::cout << std::format("[GLFW] Opening window of size {}x{}...", width, height) << std::endl;

  window = glfwCreateWindow(width, height, "CUDA Path Tracer", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);
  glfwSetWindowUserPointer(window, static_cast<void*>(ctx));

  glfwSetKeyCallback(window, callback::on_key);
  glfwSetCursorPosCallback(window, callback::on_cursor_pos);
  glfwSetMouseButtonCallback(window, callback::on_mouse_button);

  return true;
}

GLFWwindow* Window::get() {
  return window;
}
