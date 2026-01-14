#pragma once

#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace viz {

// --- Shaders ---
inline const char *vShaderSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
uniform mat4 mvp;
out vec3 vColor;
void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

inline const char *fShaderSrc = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

// --- Math / Hash ---
struct SpatialHash {
  static size_t hash(int x, int y, int z) {
    size_t p1 = 73856093;
    size_t p2 = 19349663;
    size_t p3 = 83492791;
    return (x * p1) ^ (y * p2) ^ (z * p3);
  }
};

// --- Algorithms ---

// 强度转彩虹色
inline void intensity_to_rainbow(float value, float min_v, float max_v,
                                 float &r, float &g, float &b) {
  float h = (1.0f - (value - min_v) / (max_v - min_v)) * 240.0f;
  if (h < 0)
    h = 0;
  if (h > 240)
    h = 240;
  float x = 1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f);
  if (h < 60) {
    r = 1;
    g = x;
    b = 0;
  } else if (h < 120) {
    r = x;
    g = 1;
    b = 0;
  } else if (h < 180) {
    r = 0;
    g = 1;
    b = x;
  } else if (h < 240) {
    r = 0;
    g = x;
    b = 1;
  } else {
    r = x;
    g = 0;
    b = 1;
  }
}

// 体素网格下采样
inline void process_voxel_grid(const std::vector<float> &input,
                               std::vector<float> &output, float leaf_size) {
  if (input.empty())
    return;
  std::unordered_map<size_t, size_t> voxel_map;
  float inv_leaf = 1.0f / leaf_size;
  voxel_map.reserve(input.size() / 6);

  for (size_t i = 0; i < input.size(); i += 6) {
    int ix = static_cast<int>(std::floor(input[i] * inv_leaf));
    int iy = static_cast<int>(std::floor(input[i + 1] * inv_leaf));
    int iz = static_cast<int>(std::floor(input[i + 2] * inv_leaf));
    size_t h = SpatialHash::hash(ix, iy, iz);
    if (voxel_map.find(h) == voxel_map.end()) {
      voxel_map[h] = i;
    }
  }
  output.reserve(voxel_map.size() * 6);
  for (const auto &kv : voxel_map) {
    size_t idx = kv.second;
    for (int k = 0; k < 6; ++k)
      output.push_back(input[idx + k]);
  }
}

} // namespace viz