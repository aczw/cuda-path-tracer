#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <string>

class Image {
 public:
  enum class Format { PNG, HDR };

  Image(int x, int y);

  void set_pixel(int x, int y, const glm::vec3& pixel);
  void save_as_png(const std::string& base_name, bool apply_tone_mapping);
  void save_as_hdr(const std::string& base_name);

 private:
  int x_size;
  int y_size;

  std::unique_ptr<glm::vec3[]> pixel_data;
};
