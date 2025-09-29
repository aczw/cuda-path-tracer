#pragma once

#include <cuda/std/variant>

#include <glm/glm.hpp>

using UnknownMat = cuda::std::monostate;

struct Light {
  glm::vec3 color;
  float emission;
};

struct Diffuse {
  glm::vec3 color;
};

struct PureReflection {};

struct PureTransmission {
  /// Stores the relative index of refraction (IOR) of this material
  /// over a vacuum, which has a IOR of 1.0.
  float eta;
};

struct PerfectSpecular {
  /// Stores the relative index of refraction (IOR) of this material
  /// over a vacuum, which has a IOR of 1.0.
  float eta;
};

using Material = cuda::std::
    variant<UnknownMat, Light, Diffuse, PureReflection, PureTransmission, PerfectSpecular>;
