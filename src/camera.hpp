#pragma once

#include <glm/glm.hpp>

#include <compare>

class Camera {
 public:
  void update(double zoom, double theta, double phi);

  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 look_at;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixel_length;

  auto operator<=>(const Camera&) const = default;
};
