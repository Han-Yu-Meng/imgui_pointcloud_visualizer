#pragma once
#include <Eigen/Dense>

namespace viz {

struct Camera {
  Eigen::Vector3f pivot = {0.0f, 0.0f, 0.0f};
  float yaw = -90.0f;
  float pitch = 45.0f;
  float distance = 5.0f;
  Eigen::Vector3f position;

  void update();
  Eigen::Matrix4f getViewMatrix();
  void pan(float dx, float dy);
  void rotate(float dx, float dy);
  void zoom(float offset);
};

} // namespace viz