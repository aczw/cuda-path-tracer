#include "render_context.hpp"

#include "image.hpp"

#include <array>
#include <ctime>
#include <format>
#include <iostream>

namespace {

std::string get_current_time() {
  std::time_t now = std::time(nullptr);
  std::array<char, sizeof("0000-00-00_00-00-00z")> buffer;
  std::strftime(buffer.data(), sizeof(buffer), "%Y-%m-%d_%H-%M-%Sz", std::gmtime(&now));

  return buffer.data();
}

}  // namespace

RenderContext::RenderContext() : start_time(get_current_time()), curr_iteration(0) {}

bool RenderContext::try_open_scene(std::string_view scene_file) {
  std::cout << std::format("[Scene] Opening \"{}\"...", scene_file) << std::endl;

  std::string_view extension = scene_file.substr(scene_file.find_last_of('.'));

  if (extension != ".json") {
    std::cerr << std::format("[Scene] Error, \"{}\" is not a JSON file", scene_file) << std::endl;
    return false;
  }

  settings = scene.load_from_json(scene_file);
  const Camera& camera = scene.camera;

  // Color initial image black
  image.resize(camera.resolution.x * camera.resolution.y);
  std::fill(image.begin(), image.end(), glm::vec3());

  const glm::vec3& view = camera.view;
  glm::vec3 view_xz = glm::vec3(view.x, 0.f, view.z);
  glm::vec3 view_yz = glm::vec3(0.f, view.y, view.z);

  input = {
      .left_mouse_pressed = false,
      .right_mouse_pressed = false,
      .middle_mouse_pressed = false,
      .mouse_over_gui = false,
  };

  // Compute phi (horizontal) and theta (vertical) relative 3D axis.
  // So [0, 0, 1] is forward, [0, 1, 0] is up
  phi = glm::acos(glm::dot(glm::normalize(view_xz), glm::vec3(0.f, 0.f, -1.f)));
  theta = glm::acos(glm::dot(glm::normalize(view_yz), glm::vec3(0.f, 1.f, 0.f)));
  zoom = glm::length(camera.position - camera.look_at);

  // Will immediately be set by GLFW cursor callback
  last_cursor_x = -1.f;
  last_cursor_y = -1.f;

  pbo = 0;
  display_image = 0;

  return true;
}

void RenderContext::save_image(Image::Format format) const {
  float num_samples = curr_iteration;
  int width = get_width();
  int height = get_height();

  Image output(width, height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int index = x + (y * width);

      glm::vec3 pixel = image[index];
      output.set_pixel(width - 1 - x, y, pixel / num_samples);
    }
  }

  std::string base_name =
      std::format("{}_{}_{}samples", settings.output_image_name, start_time, num_samples);

  switch (format) {
    case Image::Format::PNG:
      output.save_as_png(base_name);
      break;

    case Image::Format::HDR:
      output.save_as_hdr(base_name);
      break;

    default:
      break;
  }
}

int RenderContext::get_width() const {
  return scene.camera.resolution.x;
}

int RenderContext::get_height() const {
  return scene.camera.resolution.y;
}
