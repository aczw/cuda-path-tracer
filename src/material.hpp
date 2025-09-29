#pragma once

#include <glm/glm.hpp>

struct Material {
  enum class Type : char {
    Unknown,
    Light,
    Diffuse,
    PureReflection,
    PureTransmission,
    PerfectSpecular
  } type;

  glm::vec3 color;
  float emission;

  /// Stores the relative index of refraction (IOR) of this material
  /// over a vacuum, which has a IOR of 1.0.
  float eta;
};
