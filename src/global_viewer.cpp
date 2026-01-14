/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University.
 * Global Viewer Implementation
 ******************************************************************************/

#include "global_viewer.hpp"
#include "camera.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

// OpenGL / ImGui Headers
// Must define this before including gl.h to load modern OpenGL functions
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

namespace viz {

// ==========================================
// 1. Helper Functions
// ==========================================

// Simple random downsampling to reduce render load per frame
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

// Create batch on CPU side (Node Thread)
static GPUBatch create_cpu_batch(const std::vector<float> &data) {
  GPUBatch batch;
  batch.count = data.size() / 6;
  batch.timestamp =
      glfwGetTime(); // Ensure GLFW is initialized before calling this
  batch.cpu_data = data;
  batch.vao = 0;
  batch.vbo = 0;
  return batch;
}

// Upload batch to GPU (Render Thread)
static void upload_batch_to_gpu(GPUBatch &batch) {
  if (batch.cpu_data.empty() || batch.vao != 0)
    return;

  glGenVertexArrays(1, &batch.vao);
  glGenBuffers(1, &batch.vbo);

  glBindVertexArray(batch.vao);
  glBindBuffer(GL_ARRAY_BUFFER, batch.vbo);
  glBufferData(GL_ARRAY_BUFFER, batch.cpu_data.size() * sizeof(float),
               batch.cpu_data.data(), GL_STATIC_DRAW);

  // Pos (x,y,z)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // Color (r,g,b)
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  // Clear CPU memory after upload
  std::vector<float>().swap(batch.cpu_data);
}

// ==========================================
// 3. GlobalViewer Implementation
// ==========================================

GlobalViewer &GlobalViewer::get() {
  static GlobalViewer instance;
  return instance;
}

GlobalViewer::~GlobalViewer() {
  // Force stop thread on destruction
  stop_thread();
}

void GlobalViewer::register_active_node() {
  int prev = active_node_count_.fetch_add(1);
  if (prev == 0) {
    start_thread();
  }
}

void GlobalViewer::unregister_active_node() {
  int prev = active_node_count_.fetch_sub(1);
  // if (prev == 1) { // 1 -> 0
  //   stop_thread();
  // }
}

void GlobalViewer::start_thread() {
  if (!thread_started_.exchange(true)) {
    is_running_ = true;
    render_thread_ = std::thread(&GlobalViewer::render_loop, this);
  }
}

void GlobalViewer::stop_thread() {
  is_running_ = false;
  if (thread_started_.exchange(false)) {
    if (render_thread_.joinable()) {
      render_thread_.join();
    }
  }
}

void GlobalViewer::cleanup_gl_resources() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Clear map triggers GPUBatch destructors -> glDeleteBuffers
  // This MUST happen while GL context is still valid
  clouds_.clear();

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
  if (shader_program_) {
    glDeleteProgram(shader_program_);
    shader_program_ = 0;
  }
}

// [Thread: FINS Node Thread]
void GlobalViewer::update_cloud(const std::string &name,
                                const std::vector<float> &raw_data) {
  if (!is_running_)
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);

  if (first_load_ && !raw_data.empty()) {
    auto_center_camera(raw_data);
    first_load_ = false;
  }

  auto &cloud = clouds_[name];

  // Downsample input to avoid overcrowding
  std::vector<float> frame_data;
  random_downsample(raw_data, frame_data, cloud.max_points_per_frame);
  if (frame_data.empty())
    return;

  bool infinite_mode = (cloud.decay_time < 0.0f);

  if (infinite_mode) {
    // --- Infinite Mode (Spatial Hashing) ---
    float inv_leaf = 1.0f / cloud.map_voxel_size;
    for (size_t i = 0; i < frame_data.size(); i += 6) {
      int ix = static_cast<int>(std::floor(frame_data[i] * inv_leaf));
      int iy = static_cast<int>(std::floor(frame_data[i + 1] * inv_leaf));
      int iz = static_cast<int>(std::floor(frame_data[i + 2] * inv_leaf));
      size_t h = SpatialHash::hash(ix, iy, iz);

      if (cloud.global_voxel_map.find(h) == cloud.global_voxel_map.end()) {
        cloud.global_voxel_map.insert(h);
        for (int k = 0; k < 6; ++k)
          cloud.accum_buffer.push_back(frame_data[i + k]);
      }
    }
    // Accumulate enough points before creating a batch
    if (cloud.accum_buffer.size() > 5000 * 6) {
      cloud.batches.push_back(create_cpu_batch(cloud.accum_buffer));
      cloud.accum_buffer.clear();
    }
  } else {
    // --- Decay Mode ---
    // Cleanup infinite mode leftovers
    if (!cloud.global_voxel_map.empty()) {
      cloud.global_voxel_map.clear();
      cloud.accum_buffer.clear();
      // Just clear vector, GL resource cleanup handles in sync_and_prune or
      // destructor
      cloud.batches.clear();
    }

    // Push new frame batch
    cloud.batches.push_back(create_cpu_batch(frame_data));
  }
}

// [Thread: Render Thread]
void GlobalViewer::sync_and_prune() {
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

    // 4. Update stats
    size_t count = 0;
    for (const auto &b : cloud.batches)
      count += b.count;
    cloud.total_points_displayed = count;
  }
}

void GlobalViewer::auto_center_camera(const std::vector<float> &data) {
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

void GlobalViewer::init_shader() {
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

void GlobalViewer::init_env_geometry() {
  std::vector<float> grid_data;
  float size = 10.0f;
  float step = 1.0f;
  float r = 0.85f, g = 0.85f, b = 0.85f; // Light Grey
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

void GlobalViewer::render_loop() {
  // 1. Init GLFW
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

  // 2. Init ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // 3. Init GL
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_MULTISAMPLE);

  init_shader();
  init_env_geometry();

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

    // Draw Env
    glLineWidth(1.0f);
    glBindVertexArray(grid_vao);
    glDrawArrays(GL_LINES, 0, (GLsizei)grid_vertex_count);
    glLineWidth(3.0f);
    glBindVertexArray(axes_vao);
    glDrawArrays(GL_LINES, 0, (GLsizei)axes_vertex_count);

    // Draw Batches
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
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

    // UI
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene Manager");

    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Separator();
    ImGui::Text("Camera");
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

    ImGui::Separator();
    ImGui::Text("Active Point Clouds");
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (auto &[name, cloud] : clouds_) {
        ImGui::PushID(name.c_str());
        ImGui::Separator();

        bool open = ImGui::TreeNode(name.c_str());
        ImGui::SameLine();
        ImGui::Checkbox("##vis", &cloud.visible);
        if (open) {
          ImGui::Text("Pts: %zu | Batches: %zu", cloud.total_points_displayed,
                      cloud.batches.size());
          ImGui::SliderFloat("Size", &cloud.point_size, 1.0f, 10.0f);
          ImGui::SliderInt("Max Pts/Frame", &cloud.max_points_per_frame, 1000,
                           50000);

          bool inf = (cloud.decay_time < 0.0f);
          if (ImGui::Checkbox("Infinite Map", &inf)) {
            cloud.decay_time = inf ? -1.0f : 10.0f;
            for (auto &b : cloud.batches)
              b.release();
            cloud.batches.clear();
            cloud.global_voxel_map.clear();
            cloud.accum_buffer.clear();
          }
          if (!inf)
            ImGui::SliderFloat("Decay(s)", &cloud.decay_time, 0.0f, 120.0f);
          else
            ImGui::SliderFloat("Map Voxel(m)", &cloud.map_voxel_size, 0.05f,
                               0.5f);

          if (ImGui::Button("Clear")) {
            for (auto &b : cloud.batches)
              b.release();
            cloud.batches.clear();
            cloud.global_voxel_map.clear();
            cloud.accum_buffer.clear();
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

  // Critical: Cleanup GL resources before destroying context
  cleanup_gl_resources();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}

} // namespace viz