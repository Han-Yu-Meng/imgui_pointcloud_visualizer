#include "camera.hpp"
#include <cmath>

namespace viz {

// ==========================================
// 2. Camera Implementation
// ==========================================

void Camera::update() {
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

Eigen::Matrix4f Camera::getViewMatrix() {
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

void Camera::pan(float dx, float dy) {
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

void Camera::rotate(float dx, float dy) {
  yaw -= dx * 0.5f;
  pitch += dy * 0.5f;
}

void Camera::zoom(float offset) {
  float speed = 0.1f;
  distance -= offset * distance * speed;
  if (distance < 0.1f)
    distance = 0.1f;
  if (distance > 2000.0f)
    distance = 2000.0f;
}

} // namespace viz