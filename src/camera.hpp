#pragma once

#include <glm/glm.hpp>

#include <compare>

/// Settings related to configuring how the camera will work and how
/// rays will initially be generated.
struct CameraSettings {
  bool stochastic_sampling;
  bool depth_of_field;
  float lens_radius;
  float focal_distance;

  auto operator<=>(const CameraSettings&) const = default;
};

class Camera {
 public:
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 look_at;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixel_length;

  void update(double zoom, double theta, double phi);

  auto operator<=>(const Camera&) const = default;
};
