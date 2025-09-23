//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include "utilities.cuh"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <iostream>

__host__ __device__ thrust::default_random_engine make_seeded_random_engine(int iteration,
                                                                            int index,
                                                                            int depth) {
  int seed = generate_hash((1 << 31) | (depth << 22) | iteration) ^ generate_hash(index);
  return thrust::default_random_engine(seed);
}

namespace util {

bool replaceString(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos) {
    return false;
  }
  str.replace(start_pos, from.length(), to);
  return true;
}

std::string convertIntToString(int number) {
  std::stringstream ss;
  ss << number;
  return ss.str();
}

glm::vec3 clampRGB(glm::vec3 color) {
  if (color[0] < 0) {
    color[0] = 0;
  } else if (color[0] > 255) {
    color[0] = 255;
  }

  if (color[1] < 0) {
    color[1] = 0;
  } else if (color[1] > 255) {
    color[1] = 255;
  }

  if (color[2] < 0) {
    color[2] = 0;
  } else if (color[2] > 255) {
    color[2] = 255;
  }

  return color;
}

bool epsilonCheck(float a, float b) {
  return fabs(fabs(a) - fabs(b)) < EPSILON;
}

glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
  glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
  glm::mat4 rotationMat =
      glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
  rotationMat =
      rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
  rotationMat =
      rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
  glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
  return translationMat * rotationMat * scaleMat;
}

std::vector<std::string> util::tokenizeString(std::string str) {
  std::stringstream strstr(str);
  std::istream_iterator<std::string> it(strstr);
  std::istream_iterator<std::string> end;
  std::vector<std::string> results(it, end);
  return results;
}

}  // namespace util
