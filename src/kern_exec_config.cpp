#include "kern_exec_config.hpp"

#include <glm/glm.hpp>

KernExecConfig::KernExecConfig(glm::ivec2 resolution, int block_size)
    : block_size(block_size),
      num_blocks(((resolution.x * resolution.y) + block_size - 1) /
                 block_size) {}

int KernExecConfig::get_block_size() {
  return block_size;
}

int KernExecConfig::get_num_blocks() {
  return num_blocks;
}
