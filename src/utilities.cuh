#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define INV_PI 0.3183098861837906715377675267450287240689f

#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f

#define EPSILON 0.00001f

class GuiDataContainer {
 public:
  GuiDataContainer() : traced_depth(0) {}

  int traced_depth;
};

__host__ __device__ thrust::default_random_engine make_seeded_random_engine(int iteration,
                                                                            int index,
                                                                            int depth);

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int generate_hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);

  return a;
}

namespace util {

bool replaceString(std::string& str, const std::string& from, const std::string& to);
glm::vec3 clampRGB(glm::vec3 color);
bool epsilonCheck(float a, float b);
std::vector<std::string> tokenizeString(std::string str);
glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
std::string convertIntToString(int number);

}  // namespace util
