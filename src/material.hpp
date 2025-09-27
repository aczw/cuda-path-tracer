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

struct Specular {
  glm::vec3 color;
};

using Material = cuda::std::variant<UnknownMat, Light, Diffuse, Specular>;
