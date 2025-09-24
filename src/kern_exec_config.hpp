#pragma once

#include <glm/glm.hpp>

/// Stores the execution configration (launch parameters) for a kernel.
class KernExecConfig {
 public:
  explicit KernExecConfig(glm::ivec2 resolution, int block_size);

  int get_block_size();
  int get_num_blocks();

 private:
  int block_size;
  int num_blocks;
};
