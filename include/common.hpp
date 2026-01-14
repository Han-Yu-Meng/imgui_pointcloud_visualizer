#pragma once
#include <GL/gl.h>
#include <deque>
#include <unordered_set>
#include <vector>

namespace viz {

struct GPUBatch {
  GLuint vao = 0;
  GLuint vbo = 0;
  size_t count = 0;
  double timestamp = 0.0;

  // [新增] 暂存 CPU 数据，用于跨线程传输
  std::vector<float> cpu_data;

  void release() {
    if (vbo)
      glDeleteBuffers(1, &vbo);
    if (vao)
      glDeleteVertexArrays(1, &vao);
    vbo = 0;
    vao = 0;
    count = 0;
    cpu_data.clear();
  }
};

struct CloudRenderData {
  std::deque<GPUBatch> batches;
  size_t total_points_displayed = 0;

  // Infinite Mode 累积缓存
  std::vector<float> accum_buffer;
  std::unordered_set<size_t> global_voxel_map;

  bool visible = true;
  float point_size = 2.0f;
  float decay_time = 10.0f;
  int max_points_per_frame = 20000;
  float map_voxel_size = 0.1f;
  bool enable_voxel_filter = true;
  float voxel_size = 0.2f;

  ~CloudRenderData() {
    for (auto &b : batches)
      b.release();
  }
};

} // namespace viz