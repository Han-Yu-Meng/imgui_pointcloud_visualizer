/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University.
 * Global Viewer - Header-Only Implementation
 ******************************************************************************/

#ifndef GLOBAL_VIEWER_HPP
#define GLOBAL_VIEWER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

namespace viz {

// ============================================================================
// Forward Declarations and Helper Structures
// ============================================================================

struct GPUBatch {
  unsigned int vao = 0;
  unsigned int vbo = 0;
  size_t count = 0;
  double timestamp = 0.0;
  std::vector<float> cpu_data;

  void release() {
    if (vao != 0) {
      glDeleteVertexArrays(1, &vao);
      glDeleteBuffers(1, &vbo);
      vao = 0;
      vbo = 0;
    }
  }
  ~GPUBatch() { release(); }
};

struct CloudState {
  std::deque<GPUBatch> batches;
  std::vector<float> accum_buffer;

  bool visible = true;
  float point_size = 2.0f;
  int max_points_per_frame = 10000;
  float decay_time = -1.0f; // Default: infinite mode
  size_t total_points_displayed = 0;
};

struct TransformState {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  double timestamp = 0.0;
  bool visible = true;
  float axis_length = 1.0f;
  std::string child_frame;
  std::string parent_frame;
};

struct PathState {
  std::vector<Eigen::Vector3f> points;
  bool visible = true;
  float line_width = 2.0f;
  Eigen::Vector3f color = Eigen::Vector3f(1.0f, 0.5f, 0.0f); // Orange
  unsigned int vao = 0;
  unsigned int vbo = 0;
  bool needs_update = true;

  void release() {
    if (vao != 0) {
      glDeleteVertexArrays(1, &vao);
      glDeleteBuffers(1, &vbo);
      vao = 0;
      vbo = 0;
    }
  }
  ~PathState() { release(); }
};

struct Camera {
  Eigen::Vector3f pivot = {0, 0, 0};
  Eigen::Vector3f position = {0, 0, 0};
  float distance = 5.0f;
  float yaw = -90.0f;
  float pitch = 45.0f;

  void update() {
    if (pitch > 89.0f)
      pitch = 89.0f;
    if (pitch < -89.0f)
      pitch = -89.0f;

    float radYaw = yaw * M_PI / 180.0f;
    float radPitch = pitch * M_PI / 180.0f;

    float x = distance * cos(radPitch) * cos(radYaw);
    float y = distance * cos(radPitch) * sin(radYaw);
    float z = distance * sin(radPitch);

    position = pivot + Eigen::Vector3f(x, y, z);
  }

  Eigen::Matrix4f getViewMatrix() {
    update();
    Eigen::Vector3f f = (pivot - position).normalized();
    Eigen::Vector3f worldUp = {0.0f, 0.0f, 1.0f};
    Eigen::Vector3f r = f.cross(worldUp).normalized();
    if (r.norm() < 0.01f)
      r = Eigen::Vector3f(1, 0, 0);
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f view;
    view << r.x(), r.y(), r.z(), -r.dot(position), u.x(), u.y(), u.z(),
        -u.dot(position), -f.x(), -f.y(), -f.z(), f.dot(position), 0, 0, 0, 1;
    return view;
  }

  void pan(float dx, float dy) {
    float radYaw = yaw * M_PI / 180.0f;
    float c = cos(radYaw);
    float s = sin(radYaw);

    // 在 XY 平面上的投影向量
    Eigen::Vector3f forward_g(c, s, 0.0f);
    Eigen::Vector3f right_g(s, -c, 0.0f);

    float speed = distance * 0.002f;
    // 修正后的方向逻辑
    Eigen::Vector3f move = right_g * dx * speed - forward_g * dy * speed;

    pivot += move;
    pivot.z() = 0.0f; // 强制锁定 Z 轴
  }

  void rotate(float dx, float dy) {
    yaw -= dx * 0.5f;
    pitch += dy * 0.5f;
  }

  void zoom(float offset) {
    float speed = 0.1f;
    distance -= offset * distance * speed;
    if (distance < 0.1f)
      distance = 0.1f;
    if (distance > 2000.0f)
      distance = 2000.0f;
  }
};

// ============================================================================
// GlobalViewer Class Declaration
// ============================================================================

class GlobalViewer {
public:
  static GlobalViewer &get();
  ~GlobalViewer();

  void register_active_node();
  void unregister_active_node();

  void update_cloud(const std::string &name, const std::vector<float> &data);
  void update_transform(const std::string &name,
                        const Eigen::Matrix4f &transform,
                        const std::string &child_frame,
                        const std::string &parent_frame);
  void update_path(const std::string &name,
                   const std::vector<Eigen::Vector3f> &points);

private:
  GlobalViewer() = default;
  GlobalViewer(const GlobalViewer &) = delete;
  GlobalViewer &operator=(const GlobalViewer &) = delete;

  void start_thread();
  void stop_thread();
  void render_loop();
  void sync_and_prune();
  void auto_center_camera(const std::vector<float> &data);
  void init_shader();
  void init_env_geometry();
  void init_transform_geometry();
  void cleanup_gl_resources();
  void draw_transform(const TransformState &tf, const Eigen::Matrix4f &proj,
                      const Eigen::Matrix4f &view);
  void draw_path(PathState &path, const Eigen::Matrix4f &proj,
                 const Eigen::Matrix4f &view);

  std::atomic<bool> is_running_{false};
  std::atomic<bool> thread_started_{false};
  std::atomic<int> active_node_count_{0};
  std::thread render_thread_;

  std::mutex data_mutex_;
  std::unordered_map<std::string, CloudState> clouds_;
  std::unordered_map<std::string, TransformState> transforms_;
  std::unordered_map<std::string, PathState> paths_;

  Camera camera_;
  bool first_load_ = true;

  unsigned int shader_program_ = 0;
  unsigned int grid_vao = 0, grid_vbo = 0;
  unsigned int axes_vao = 0, axes_vbo = 0;
  unsigned int pivot_vao = 0, pivot_vbo = 0;
  unsigned int tf_axes_vao = 0, tf_axes_vbo = 0;
  size_t grid_vertex_count = 0;
  size_t axes_vertex_count = 0;
  size_t tf_axes_vertex_count = 0;

  const char *vShaderSrc = R"(
    #version 330 core
    layout(location=0) in vec3 pos;
    layout(location=1) in vec3 col;
    uniform mat4 mvp;
    out vec3 fragColor;
    void main() {
      gl_Position = mvp * vec4(pos,1.0);
      gl_PointSize = 2.0;
      fragColor = col;
    }
  )";

  const char *fShaderSrc = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 color;
    void main() {
      color = vec4(fragColor, 1.0);
    }
  )";
};

// ============================================================================
// Helper Functions Implementation
// ============================================================================

static void random_downsample(const std::vector<float> &input,
                              std::vector<float> &output, int target_count) {
  size_t input_points = input.size() / 6;
  if (input_points <= static_cast<size_t>(target_count)) {
    output = input;
    return;
  }
  output.reserve(target_count * 6);
  float step = static_cast<float>(input_points) / target_count;
  for (float i = 0; i < input_points; i += step) {
    size_t idx = static_cast<size_t>(i) * 6;
    if (idx + 5 < input.size()) {
      for (int k = 0; k < 6; ++k)
        output.push_back(input[idx + k]);
    }
    if (output.size() >= static_cast<size_t>(target_count) * 6)
      break;
  }
}

static GPUBatch create_cpu_batch(const std::vector<float> &data) {
  GPUBatch batch;
  batch.count = data.size() / 6;
  batch.timestamp = glfwGetTime();
  batch.cpu_data = data;
  batch.vao = 0;
  batch.vbo = 0;
  return batch;
}

static void upload_batch_to_gpu(GPUBatch &batch) {
  if (batch.cpu_data.empty() || batch.vao != 0)
    return;

  glGenVertexArrays(1, &batch.vao);
  glGenBuffers(1, &batch.vbo);

  glBindVertexArray(batch.vao);
  glBindBuffer(GL_ARRAY_BUFFER, batch.vbo);
  glBufferData(GL_ARRAY_BUFFER, batch.cpu_data.size() * sizeof(float),
               batch.cpu_data.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  std::vector<float>().swap(batch.cpu_data);
}

// ============================================================================
// GlobalViewer Implementation
// ============================================================================

inline GlobalViewer &GlobalViewer::get() {
  static GlobalViewer instance;
  return instance;
}

inline GlobalViewer::~GlobalViewer() { stop_thread(); }

inline void GlobalViewer::register_active_node() {
  int prev = active_node_count_.fetch_add(1);
  if (prev == 0) {
    start_thread();
  }
}

inline void GlobalViewer::unregister_active_node() {
  active_node_count_.fetch_sub(1);
}

inline void GlobalViewer::start_thread() {
  if (!thread_started_.exchange(true)) {
    is_running_ = true;
    render_thread_ = std::thread(&GlobalViewer::render_loop, this);
  }
}

inline void GlobalViewer::stop_thread() {
  is_running_ = false;
  if (thread_started_.exchange(false)) {
    if (render_thread_.joinable()) {
      render_thread_.join();
    }
  }
}

inline void GlobalViewer::cleanup_gl_resources() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  clouds_.clear();
  paths_.clear();

  if (grid_vao) {
    glDeleteVertexArrays(1, &grid_vao);
    glDeleteBuffers(1, &grid_vbo);
    grid_vao = 0;
  }
  if (axes_vao) {
    glDeleteVertexArrays(1, &axes_vao);
    glDeleteBuffers(1, &axes_vbo);
    axes_vao = 0;
  }
  if (pivot_vao) {
    glDeleteVertexArrays(1, &pivot_vao);
    glDeleteBuffers(1, &pivot_vbo);
    pivot_vao = 0;
  }
  if (tf_axes_vao) {
    glDeleteVertexArrays(1, &tf_axes_vao);
    glDeleteBuffers(1, &tf_axes_vbo);
    tf_axes_vao = 0;
  }
  if (shader_program_) {
    glDeleteProgram(shader_program_);
    shader_program_ = 0;
  }
}

inline void GlobalViewer::update_cloud(const std::string &name,
                                       const std::vector<float> &raw_data) {
  if (!is_running_)
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);

  if (first_load_ && !raw_data.empty()) {
    auto_center_camera(raw_data);
    first_load_ = false;
  }

  auto &cloud = clouds_[name];
  bool infinite_mode = (cloud.decay_time < 0.0f);

  // Downsample input to avoid overcrowding (even in infinite mode)
  std::vector<float> frame_data;
  random_downsample(raw_data, frame_data, cloud.max_points_per_frame);

  if (frame_data.empty())
    return;

  if (infinite_mode) {
    // --- Infinite Mode: Accumulate points in buffer ---
    // Append frame data to accumulation buffer
    for (size_t i = 0; i < frame_data.size(); i += 6) {
      for (int k = 0; k < 6; ++k)
        cloud.accum_buffer.push_back(frame_data[i + k]);
    }

    // Create batch when buffer is large enough (reduces draw calls)
    if (cloud.accum_buffer.size() > 5000 * 6) {
      cloud.batches.push_back(create_cpu_batch(cloud.accum_buffer));
      cloud.accum_buffer.clear();
    }
  } else {
    // --- Decay Mode ---
    // Clear infinite mode leftovers
    if (!cloud.accum_buffer.empty()) {
      cloud.accum_buffer.clear();
      cloud.batches.clear();
    }

    // Push new frame batch
    cloud.batches.push_back(create_cpu_batch(frame_data));
  }
}

inline void GlobalViewer::update_transform(const std::string &name,
                                           const Eigen::Matrix4f &transform,
                                           const std::string &child_frame,
                                           const std::string &parent_frame) {
  if (!is_running_)
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  auto &tf = transforms_[name];
  tf.transform = transform;
  tf.timestamp = glfwGetTime();
  tf.child_frame = child_frame;
  tf.parent_frame = parent_frame;
}

inline void
GlobalViewer::update_path(const std::string &name,
                          const std::vector<Eigen::Vector3f> &points) {
  if (!is_running_)
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  auto &path = paths_[name];
  path.points = points;
  path.needs_update = true;
}

inline void GlobalViewer::sync_and_prune() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  double now = glfwGetTime();

  for (auto &[name, cloud] : clouds_) {
    // 1. Upload Pending Batches
    for (auto &batch : cloud.batches) {
      if (batch.vao == 0 && !batch.cpu_data.empty()) {
        upload_batch_to_gpu(batch);
      }
    }

    bool infinite_mode = (cloud.decay_time < 0.0f);

    // 2. Force upload leftovers in Infinite Mode
    if (infinite_mode && cloud.accum_buffer.size() > 1000 * 6) {
      cloud.batches.push_back(create_cpu_batch(cloud.accum_buffer));
      cloud.accum_buffer.clear();
      upload_batch_to_gpu(cloud.batches.back());
    }

    // 3. Prune old batches in Decay Mode
    if (!infinite_mode && cloud.decay_time > 0.05f) {
      while (!cloud.batches.empty()) {
        if (now - cloud.batches.front().timestamp > cloud.decay_time) {
          cloud.batches.front().release();
          cloud.batches.pop_front();
        } else {
          break;
        }
      }
    }

    // 4. Batch consolidation: merge small batches to reduce draw calls
    // Only do this periodically in infinite mode when we have many small
    // batches
    if (infinite_mode && cloud.batches.size() > 20) {
      // Check if we should consolidate (every ~100 frames)
      static int consolidate_counter = 0;
      if (++consolidate_counter > 100) {
        consolidate_counter = 0;

        // Merge all batches into one large batch
        std::vector<float> merged_data;
        size_t total_size = 0;
        for (const auto &b : cloud.batches) {
          total_size += b.count * 6;
        }
        merged_data.reserve(total_size);

        // Download data from GPU batches
        for (auto &b : cloud.batches) {
          if (b.vao != 0 && b.count > 0) {
            std::vector<float> batch_data(b.count * 6);
            glBindBuffer(GL_ARRAY_BUFFER, b.vbo);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0,
                               batch_data.size() * sizeof(float),
                               batch_data.data());
            merged_data.insert(merged_data.end(), batch_data.begin(),
                               batch_data.end());
            b.release();
          }
        }

        // Create single consolidated batch
        cloud.batches.clear();
        if (!merged_data.empty()) {
          cloud.batches.push_back(create_cpu_batch(merged_data));
          upload_batch_to_gpu(cloud.batches.back());
        }
      }
    }

    // 5. Update stats
    size_t count = 0;
    for (const auto &b : cloud.batches)
      count += b.count;
    cloud.total_points_displayed = count;
  }

  // Same for paths
  for (auto &[name, path] : paths_) {
    if (path.needs_update && !path.points.empty()) {
      if (path.vao == 0) {
        glGenVertexArrays(1, &path.vao);
        glGenBuffers(1, &path.vbo);
      }

      std::vector<float> line_data;
      line_data.reserve(path.points.size() * 6);
      for (const auto &p : path.points) {
        line_data.push_back(p.x());
        line_data.push_back(p.y());
        line_data.push_back(p.z());
        line_data.push_back(path.color.x());
        line_data.push_back(path.color.y());
        line_data.push_back(path.color.z());
      }

      glBindVertexArray(path.vao);
      glBindBuffer(GL_ARRAY_BUFFER, path.vbo);
      glBufferData(GL_ARRAY_BUFFER, line_data.size() * sizeof(float),
                   line_data.data(), GL_STATIC_DRAW);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                            (void *)(3 * sizeof(float)));
      glEnableVertexAttribArray(1);
      glBindVertexArray(0);

      path.needs_update = false;
    }
  }
}

inline void GlobalViewer::auto_center_camera(const std::vector<float> &data) {
  if (data.empty())
    return;
  float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9, min_z = 1e9,
        max_z = -1e9;
  for (size_t i = 0; i < data.size(); i += 6) {
    if (data[i] < min_x)
      min_x = data[i];
    if (data[i] > max_x)
      max_x = data[i];
    if (data[i + 1] < min_y)
      min_y = data[i + 1];
    if (data[i + 1] > max_y)
      max_y = data[i + 1];
    if (data[i + 2] < min_z)
      min_z = data[i + 2];
    if (data[i + 2] > max_z)
      max_z = data[i + 2];
  }
  camera_.pivot = {(min_x + max_x) * 0.5f, (min_y + max_y) * 0.5f, 0.0f};
  float max_dim = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
  if (max_dim > 0.1f)
    camera_.distance = max_dim * 1.5f;
}

inline void GlobalViewer::init_shader() {
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vShaderSrc, NULL);
  glCompileShader(vs);
  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fShaderSrc, NULL);
  glCompileShader(fs);
  shader_program_ = glCreateProgram();
  glAttachShader(shader_program_, vs);
  glAttachShader(shader_program_, fs);
  glLinkProgram(shader_program_);
  glDeleteShader(vs);
  glDeleteShader(fs);
}

inline void GlobalViewer::init_env_geometry() {
  std::vector<float> grid_data;
  float size = 10.0f;
  float step = 1.0f;
  float r = 0.85f, g = 0.85f, b = 0.85f;
  for (float x = -size; x <= size; x += step)
    grid_data.insert(grid_data.end(),
                     {x, -size, 0, r, g, b, x, size, 0, r, g, b});
  for (float y = -size; y <= size; y += step)
    grid_data.insert(grid_data.end(),
                     {-size, y, 0, r, g, b, size, y, 0, r, g, b});

  glGenVertexArrays(1, &grid_vao);
  glGenBuffers(1, &grid_vbo);
  glBindVertexArray(grid_vao);
  glBindBuffer(GL_ARRAY_BUFFER, grid_vbo);
  glBufferData(GL_ARRAY_BUFFER, grid_data.size() * sizeof(float),
               grid_data.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  grid_vertex_count = grid_data.size() / 6;

  std::vector<float> axes = {0, 0, 0, 0.9, 0.2, 0.2, 1, 0, 0, 0.9, 0.2, 0.2,
                             0, 0, 0, 0.2, 0.8, 0.2, 0, 1, 0, 0.2, 0.8, 0.2,
                             0, 0, 0, 0.2, 0.2, 0.9, 0, 0, 1, 0.2, 0.2, 0.9};
  glGenVertexArrays(1, &axes_vao);
  glGenBuffers(1, &axes_vbo);
  glBindVertexArray(axes_vao);
  glBindBuffer(GL_ARRAY_BUFFER, axes_vbo);
  glBufferData(GL_ARRAY_BUFFER, axes.size() * sizeof(float), axes.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  axes_vertex_count = axes.size() / 6;

  float sz = 0.2f;
  float pr = 1.0f, pg = 0.6f, pb = 0.0f;
  std::vector<float> pivot = {-sz, 0,   0, pr, pg, pb, sz, 0,  0, pr, pg, pb,
                              0,   -sz, 0, pr, pg, pb, 0,  sz, 0, pr, pg, pb};
  glGenVertexArrays(1, &pivot_vao);
  glGenBuffers(1, &pivot_vbo);
  glBindVertexArray(pivot_vao);
  glBindBuffer(GL_ARRAY_BUFFER, pivot_vbo);
  glBufferData(GL_ARRAY_BUFFER, pivot.size() * sizeof(float), pivot.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
}

inline void GlobalViewer::init_transform_geometry() {
  float len = 1.0f;
  std::vector<float> tf_axes = {0,   0,   0,   0.9, 0.2, 0.2, len, 0,   0,
                                0.9, 0.2, 0.2, 0,   0,   0,   0.2, 0.8, 0.2,
                                0,   len, 0,   0.2, 0.8, 0.2, 0,   0,   0,
                                0.2, 0.2, 0.9, 0,   0,   len, 0.2, 0.2, 0.9};

  glGenVertexArrays(1, &tf_axes_vao);
  glGenBuffers(1, &tf_axes_vbo);
  glBindVertexArray(tf_axes_vao);
  glBindBuffer(GL_ARRAY_BUFFER, tf_axes_vbo);
  glBufferData(GL_ARRAY_BUFFER, tf_axes.size() * sizeof(float), tf_axes.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  tf_axes_vertex_count = tf_axes.size() / 6;
}

inline void GlobalViewer::draw_transform(const TransformState &tf,
                                         const Eigen::Matrix4f &proj,
                                         const Eigen::Matrix4f &view) {
  if (!tf.visible)
    return;

  Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
  scale(0, 0) = tf.axis_length;
  scale(1, 1) = tf.axis_length;
  scale(2, 2) = tf.axis_length;

  Eigen::Matrix4f mvp = proj * view * tf.transform * scale;
  glUniformMatrix4fv(glGetUniformLocation(shader_program_, "mvp"), 1, GL_FALSE,
                     mvp.data());

  glLineWidth(3.0f);
  glBindVertexArray(tf_axes_vao);
  glDrawArrays(GL_LINES, 0, (GLsizei)tf_axes_vertex_count);
}

inline void GlobalViewer::draw_path(PathState &path,
                                    const Eigen::Matrix4f &proj,
                                    const Eigen::Matrix4f &view) {
  if (!path.visible || path.vao == 0 || path.points.size() < 2)
    return;

  Eigen::Matrix4f mvp = proj * view;
  glUniformMatrix4fv(glGetUniformLocation(shader_program_, "mvp"), 1, GL_FALSE,
                     mvp.data());

  glLineWidth(path.line_width);
  glBindVertexArray(path.vao);
  glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)path.points.size());
}

inline void GlobalViewer::render_loop() {
  if (!glfwInit()) {
    thread_started_ = false;
    return;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  GLFWwindow *window =
      glfwCreateWindow(1600, 900, "FINS PointCloud Viewer", NULL, NULL);
  if (!window) {
    glfwTerminate();
    thread_started_ = false;
    return;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_MULTISAMPLE);

  init_shader();
  init_env_geometry();
  init_transform_geometry();

  ImVec4 clear_color = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);

  while (is_running_ && !glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
        camera_.pan(io.MouseDelta.x, io.MouseDelta.y);
      else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
        camera_.rotate(io.MouseDelta.x, io.MouseDelta.y);
      if (io.MouseWheel != 0.0f)
        camera_.zoom(io.MouseWheel);
    }

    sync_and_prune();

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program_);

    float aspect = (float)w / (float)h;
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    float fov = 45.0f * M_PI / 180.0f;
    float f = 1.0f / tan(fov / 2.0f);
    float zNear = 0.1f, zFar = 5000.0f;
    proj(0, 0) = f / aspect;
    proj(1, 1) = f;
    proj(2, 2) = (zFar + zNear) / (zNear - zFar);
    proj(2, 3) = (2 * zFar * zNear) / (zNear - zFar);
    proj(3, 2) = -1.0f;
    proj(3, 3) = 0.0f;

    Eigen::Matrix4f view = camera_.getViewMatrix();
    Eigen::Matrix4f mvp = proj * view;
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "mvp"), 1,
                       GL_FALSE, mvp.data());

    // Draw Environment
    glLineWidth(1.0f);
    glBindVertexArray(grid_vao);
    glDrawArrays(GL_LINES, 0, (GLsizei)grid_vertex_count);
    glLineWidth(3.0f);
    glBindVertexArray(axes_vao);
    glDrawArrays(GL_LINES, 0, (GLsizei)axes_vertex_count);

    // Draw Scene Objects
    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      // Point Clouds
      for (auto &[name, cloud] : clouds_) {
        if (cloud.visible) {
          glPointSize(cloud.point_size);
          for (const auto &batch : cloud.batches) {
            if (batch.vao != 0) {
              glBindVertexArray(batch.vao);
              glDrawArrays(GL_POINTS, 0, (GLsizei)batch.count);
            }
          }
        }
      }

      // Transforms
      for (const auto &[name, tf] : transforms_) {
        draw_transform(tf, proj, view);
      }

      // Paths
      for (auto &[name, path] : paths_) {
        draw_path(path, proj, view);
      }
    }

    // Draw Pivot
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    model(0, 3) = camera_.pivot.x();
    model(1, 3) = camera_.pivot.y();
    model(2, 3) = camera_.pivot.z();
    Eigen::Matrix4f mvp_pivot = proj * view * model;
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "mvp"), 1,
                       GL_FALSE, mvp_pivot.data());
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glBindVertexArray(pivot_vao);
    glDrawArrays(GL_LINES, 0, 4);
    glEnable(GL_DEPTH_TEST);

    // ImGui UI
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene Manager");

    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextColored(ImVec4(0, 0, 0, 0.5), "Pivot: %.1f %.1f",
                         camera_.pivot.x(), camera_.pivot.y());
      if (ImGui::Button("Reset View")) {
        camera_.pivot = {0, 0, 0};
        camera_.distance = 5.0f;
        camera_.yaw = -90;
        camera_.pitch = 45;
      }
      ImGui::SameLine();
      if (ImGui::Button("Auto Center")) {
        first_load_ = true;
      }
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Point Clouds",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (auto &[name, cloud] : clouds_) {
        ImGui::PushID(name.c_str());
        ImGui::Separator();

        bool open = ImGui::TreeNode(name.c_str());
        ImGui::SameLine();
        ImGui::Checkbox("##vis", &cloud.visible);
        if (open) {
          ImGui::Text("Points: %zu", cloud.total_points_displayed);
          ImGui::Text("Batches: %zu", cloud.batches.size());
          if (!cloud.accum_buffer.empty()) {
            ImGui::Text("Buffer: %zu pts", cloud.accum_buffer.size() / 6);
          }

          ImGui::Separator();
          ImGui::SliderFloat("Point Size", &cloud.point_size, 1.0f, 10.0f);
          ImGui::SliderInt("Max Pts/Frame", &cloud.max_points_per_frame, 1000,
                           50000);

          ImGui::Separator();
          bool inf = (cloud.decay_time < 0.0f);
          if (ImGui::Checkbox("Infinite Map Mode", &inf)) {
            cloud.decay_time = inf ? -1.0f : 10.0f;
            for (auto &b : cloud.batches)
              b.release();
            cloud.batches.clear();
            cloud.accum_buffer.clear();
          }

          if (!inf) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                               "Decay Mode Settings:");
            ImGui::SliderFloat("Decay Time (s)", &cloud.decay_time, 0.1f,
                               120.0f);
          } else {
            ImGui::TextColored(ImVec4(0.0f, 0.5f, 0.0f, 1.0f),
                               "Infinite Mode: Accumulating");
          }

          ImGui::Separator();
          if (ImGui::Button("Clear All Data")) {
            for (auto &b : cloud.batches)
              b.release();
            cloud.batches.clear();
            cloud.accum_buffer.clear();
          }
          ImGui::TreePop();
        }
        ImGui::PopID();
      }
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Transforms")) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (auto &[name, tf] : transforms_) {
        ImGui::PushID(name.c_str());
        ImGui::Separator();

        bool open = ImGui::TreeNode(name.c_str());
        ImGui::SameLine();
        ImGui::Checkbox("##vis", &tf.visible);
        if (open) {
          ImGui::Text("Child: %s", tf.child_frame.c_str());
          ImGui::Text("Parent: %s", tf.parent_frame.c_str());
          ImGui::SliderFloat("Axis Length", &tf.axis_length, 0.1f, 5.0f);

          Eigen::Vector3f pos = tf.transform.block<3, 1>(0, 3);
          ImGui::Text("Position: %.2f, %.2f, %.2f", pos.x(), pos.y(), pos.z());

          ImGui::TreePop();
        }
        ImGui::PopID();
      }
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Paths")) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (auto &[name, path] : paths_) {
        ImGui::PushID(name.c_str());
        ImGui::Separator();

        bool open = ImGui::TreeNode(name.c_str());
        ImGui::SameLine();
        ImGui::Checkbox("##vis", &path.visible);
        if (open) {
          ImGui::Text("Points: %zu", path.points.size());
          ImGui::SliderFloat("Line Width", &path.line_width, 0.5f, 10.0f);

          float color[3] = {path.color.x(), path.color.y(), path.color.z()};
          if (ImGui::ColorEdit3("Color", color)) {
            path.color = Eigen::Vector3f(color[0], color[1], color[2]);
            path.needs_update = true;
          }

          if (ImGui::Button("Clear")) {
            path.points.clear();
            path.release();
          }

          ImGui::TreePop();
        }
        ImGui::PopID();
      }
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  cleanup_gl_resources();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}

} // namespace viz

#endif // GLOBAL_VIEWER_HPP