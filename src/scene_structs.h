#pragma once

#include "camera.hpp"
#include "ray.cuh"

#include <glm/glm.hpp>

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

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

struct RenderState {
  Camera camera;
  unsigned int total_iterations;
  int trace_depth;
  std::vector<glm::vec3> image;
  std::string image_name;
};

struct PathSegment {
  Ray ray;
  glm::vec3 radiance;
  glm::vec3 throughput;
  int pixel_index;
  int remaining_bounces;
};
