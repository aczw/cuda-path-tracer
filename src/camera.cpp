#include "camera.hpp"

void Camera::update(double zoom, double theta, double phi) {
  static const glm::vec3 new_up(0.f, 1.f, 0.f);

  glm::vec3 new_pos;
  new_pos.x = zoom * std::sin(phi) * std::sin(theta);
  new_pos.y = zoom * std::cos(theta);
  new_pos.z = zoom * std::cos(phi) * std::sin(theta);

  view = -glm::normalize(new_pos);
  glm::vec3 new_right = glm::cross(view, new_up);
  up = glm::cross(new_right, view);
  right = new_right;

  new_pos += look_at;
  position = new_pos;
}
