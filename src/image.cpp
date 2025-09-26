#include "image.hpp"

#include "tone_mapping.cuh"

#include <glm/gtc/type_ptr.hpp>
#include <stb_image_write.h>

#include <cmath>
#include <iostream>
#include <string>

Image::Image(int x, int y)
    : x_size(x), y_size(y), pixel_data(std::make_unique<glm::vec3[]>(x * y)) {}

void Image::set_pixel(int x, int y, const glm::vec3& pixel) {
  assert(x >= 0 && y >= 0 && x < x_size && y < y_size);
  pixel_data[(y * x_size) + x] = pixel;
}

void Image::save_as_png(const std::string& base_name, bool apply_tone_mapping) {
  std::unique_ptr bytes = std::make_unique<unsigned char[]>(3 * x_size * y_size);

  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      int index = y * x_size + x;
      glm::vec3 raw_pixel = pixel_data[index];

      glm::vec3 pixel;
      if (apply_tone_mapping) {
        glm::vec3 sdr = glm::clamp(apply_reinhard(raw_pixel), glm::vec3(), glm::vec3(1.f));
        pixel = gamma_correct(sdr) * 255.f;
      } else {
        pixel = glm::clamp(raw_pixel, glm::vec3(), glm::vec3(1.f)) * 255.f;
      }

      bytes[3 * index + 0] = static_cast<unsigned char>(pixel.x);
      bytes[3 * index + 1] = static_cast<unsigned char>(pixel.y);
      bytes[3 * index + 2] = static_cast<unsigned char>(pixel.z);
    }
  }

  std::string file_name = base_name + ".png";
  stbi_write_png(file_name.c_str(), x_size, y_size, 3, bytes.get(), x_size * 3);

  std::cout << "[Info] Saved as " << file_name << "." << std::endl;
}

void Image::save_as_hdr(const std::string& base_name) {
  std::string file_name = base_name + ".hdr";
  stbi_write_hdr(file_name.c_str(), x_size, y_size, 3, glm::value_ptr(pixel_data[0]));

  std::cout << "[Info] Saved as " + file_name + "." << std::endl;
}
