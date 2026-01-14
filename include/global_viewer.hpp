
#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "camera.hpp"
#include "common.hpp"

namespace viz {

class GlobalViewer {
public:
  static GlobalViewer &get();

  void register_active_node();
  void unregister_active_node();

  void update_cloud(const std::string &name,
                    const std::vector<float> &raw_data);

private:
  GlobalViewer() = default;
  ~GlobalViewer();

  void start_thread();
  void stop_thread();
  void render_loop();

  void cleanup_gl_resources();

  void init_shader();
  void init_env_geometry();
  void sync_and_prune();
  void auto_center_camera(const std::vector<float> &data);

  std::map<std::string, CloudRenderData> clouds_;
  std::mutex data_mutex_;
  std::thread render_thread_;

  std::atomic<bool> is_running_{false};
  std::atomic<bool> thread_started_{false};

  std::atomic<int> active_node_count_{0};

  Camera camera_;
  unsigned int shader_program_ = 0;
  unsigned int grid_vao = 0, grid_vbo = 0;
  size_t grid_vertex_count = 0;
  unsigned int axes_vao = 0, axes_vbo = 0;
  size_t axes_vertex_count = 0;
  unsigned int pivot_vao = 0, pivot_vbo = 0;

  bool first_load_ = true;
};

} // namespace viz