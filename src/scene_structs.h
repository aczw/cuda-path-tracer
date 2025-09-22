#pragma once

#include "glm/glm.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Geometry {
  enum class Type { Sphere, Cube };

  Type type;
  int material_id;
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inv_transform;
  glm::mat4 inv_transpose;
};

struct Material {
  glm::vec3 color;

  struct {
    float exponent;
    glm::vec3 color;
  } specular;

  float has_reflective;
  float has_refractive;
  float index_of_refraction;
  float emittance;
};

struct Camera {
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 lookAt;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixelLength;
};

struct RenderState {
  Camera camera;
  unsigned int iterations;
  int traceDepth;
  std::vector<glm::vec3> image;
  std::string imageName;
};

struct PathSegment {
  Ray ray;
  glm::vec3 color;
  int pixelIndex;
  int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
