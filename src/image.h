#pragma once

#include <glm/glm.hpp>

#include <string>

class Image {
 private:
  int xSize;
  int ySize;
  glm::vec3* pixels;

 public:
  Image(int x, int y);
  ~Image();

  void set_pixel(int x, int y, const glm::vec3& pixel);
  void save_as_png(const std::string& file_name);
  void save_as_hdr(const std::string& file_name);
};
