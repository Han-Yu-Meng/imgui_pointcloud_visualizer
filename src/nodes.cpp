#include <fins/node.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include "global_viewer.hpp"

using namespace viz;

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

class BaseImGuiVisualizer : public fins::Node {
protected:
  std::string title_;

public:
  void define() override {
    register_parameter<std::string>("title", &BaseImGuiVisualizer::update_title,
                                    "PCL Cloud");
  }
  void update_title(const std::string &new_title) { title_ = new_title; }
  void initialize() override {}

  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }

  void reset() override {
    std::vector<float> empty;
    GlobalViewer::get().update_cloud(title_, empty);
  }
};

class ImGuiCloudXYZVisualizer : public BaseImGuiVisualizer {
public:
  void define() override {
    BaseImGuiVisualizer::define();
    set_name("ImGuiCloudXYZVisualizer");
    set_description("Visualizes XYZ PointCloud (Black/Grey)");
    set_category("Visualization");
    register_input<pcl::PointCloud<pcl::PointXYZ>::Ptr>(
        "cloud", &ImGuiCloudXYZVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZ>::Ptr> &msg) {
    auto &cloud = *msg;
    if (!cloud || cloud->empty()){
      logger->warn("Empty cloud received in ImGuiCloudXYZVisualizer");
      return;
    }
    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);
    for (const auto &p : *cloud) {
      buffer.push_back(p.x);
      buffer.push_back(p.y);
      buffer.push_back(p.z);
      buffer.push_back(0.15f);
      buffer.push_back(0.15f);
      buffer.push_back(0.15f);
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

class ImGuiCloudXYZIVisualizer : public BaseImGuiVisualizer {
public:
  void define() override {
    BaseImGuiVisualizer::define();
    set_name("ImGuiCloudXYZIVisualizer");
    set_description("Visualizes XYZI PointCloud (Rainbow Intensity)");
    set_category("Visualization");
    register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "cloud", &ImGuiCloudXYZIVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    logger->debug("Received XYZI cloud message");
    auto &cloud = *msg;
    if (!cloud || cloud->empty()) {
      logger->warn("Empty cloud received in ImGuiCloudXYZIVisualizer");
      return ;
    }
    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);

    float min_i = std::numeric_limits<float>::max();
    float max_i = std::numeric_limits<float>::lowest();
    for (const auto &p : *cloud) {
      if (p.intensity < min_i)
        min_i = p.intensity;
      if (p.intensity > max_i)
        max_i = p.intensity;
    }
    if (max_i <= min_i)
      max_i = min_i + 1.0f;

    for (const auto &p : *cloud) {
      buffer.push_back(p.x);
      buffer.push_back(p.y);
      buffer.push_back(p.z);
      float r, g, b;
      intensity_to_rainbow(p.intensity, min_i, max_i, r, g, b);
      buffer.push_back(r);
      buffer.push_back(g);
      buffer.push_back(b);
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

class ImGuiCloudRGBVisualizer : public BaseImGuiVisualizer {
public:
  void define() override {
    BaseImGuiVisualizer::define();
    set_name("ImGuiCloudRGBVisualizer");
    set_description("Visualizes XYZRGB PointCloud");
    set_category("Visualization");
    register_input<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(
        "cloud", &ImGuiCloudRGBVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &msg) {
    logger->debug("Received XYZRGB cloud message");
    auto &cloud = *msg;
    if (!cloud || cloud->empty()){
      logger->warn("Empty cloud received in ImGuiCloudRGBVisualizer");
      return ;
    }
    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);
    for (const auto &p : *cloud) {
      buffer.push_back(p.x);
      buffer.push_back(p.y);
      buffer.push_back(p.z);
      buffer.push_back(p.r / 255.0f);
      buffer.push_back(p.g / 255.0f);
      buffer.push_back(p.b / 255.0f);
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

// Transform Visualizer - Shows coordinate frames like RViz2
class ImGuiTransformVisualizer : public fins::Node {
private:
  std::string title_;
  std::string fixed_frame_;

public:
  void define() override {
    set_name("ImGuiTransformVisualizer");
    set_description("Visualizes TransformStamped as coordinate axes (like RViz2)");
    set_category("Visualization");
    
    register_parameter<std::string>("title", 
                                    &ImGuiTransformVisualizer::update_title,
                                    "Transform");
    register_parameter<std::string>("fixed_frame",
                                    &ImGuiTransformVisualizer::update_fixed_frame,
                                    "camera_init");
    
    register_input<geometry_msgs::msg::TransformStamped>(
        "transform", &ImGuiTransformVisualizer::on_transform);
  }

  void update_title(const std::string &new_title) { title_ = new_title; }
  void update_fixed_frame(const std::string &frame) { fixed_frame_ = frame; }

  void initialize() override {}

  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }

  void reset() override {
    // Clear transform visualization
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    GlobalViewer::get().update_transform(title_, identity, "", "");
  }

  void on_transform(const fins::Msg<geometry_msgs::msg::TransformStamped> &msg) {
    logger->debug("Received TransformStamped message");
    if (!msg){
      logger->warn("Null transform received in ImGuiTransformVisualizer");
      return;
    }

    auto &tf = *msg;
    
    // Convert ROS transform to Eigen Matrix4f
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // Translation
    transform(0, 3) = tf.transform.translation.x;
    transform(1, 3) = tf.transform.translation.y;
    transform(2, 3) = tf.transform.translation.z;
    
    // Rotation (quaternion to matrix)
    Eigen::Quaternionf q(
      tf.transform.rotation.w,
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z
    );
    Eigen::Matrix3f rot = q.toRotationMatrix();
    transform.block<3, 3>(0, 0) = rot;
    
    GlobalViewer::get().update_transform(
      title_, 
      transform,
      tf.child_frame_id,
      tf.header.frame_id
    );
  }
};

// Path Visualizer - Shows trajectory paths
class ImGuiPathVisualizer : public fins::Node {
private:
  std::string title_;
  size_t max_points_;

public:
  void define() override {
    set_name("ImGuiPathVisualizer");
    set_description("Visualizes nav_msgs::Path as a line trajectory");
    set_category("Visualization");
    
    register_parameter<std::string>("title", 
                                    &ImGuiPathVisualizer::update_title,
                                    "Path");
    register_parameter<size_t>("max_points",
                              &ImGuiPathVisualizer::update_max_points,
                              1000);
    
    register_input<nav_msgs::msg::Path>(
        "path", &ImGuiPathVisualizer::on_path);
  }

  void update_title(const std::string &new_title) { title_ = new_title; }
  void update_max_points(size_t max_pts) { max_points_ = max_pts; }

  void initialize() override {
    max_points_ = 1000;
  }

  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }

  void reset() override {
    std::vector<Eigen::Vector3f> empty;
    GlobalViewer::get().update_path(title_, empty);
  }

  void on_path(const fins::Msg<nav_msgs::msg::Path> &msg) {
    auto &path = *msg;
    if (path.poses.empty()){
      logger->warn("Empty path received in ImGuiPathVisualizer");
      return ;
    }

    std::vector<Eigen::Vector3f> points;
    points.reserve(std::min(path.poses.size(), max_points_));
    
    // Downsample if necessary
    size_t step = 1;
    if (path.poses.size() > max_points_) {
      step = path.poses.size() / max_points_;
    }
    
    for (size_t i = 0; i < path.poses.size(); i += step) {
      const auto &pose = path.poses[i].pose.position;
      points.emplace_back(pose.x, pose.y, pose.z);
    }
    
    GlobalViewer::get().update_path(title_, points);
  }
};

// Export nodes
EXPORT_NODE(ImGuiCloudXYZVisualizer)
EXPORT_NODE(ImGuiCloudXYZIVisualizer)
EXPORT_NODE(ImGuiCloudRGBVisualizer)
EXPORT_NODE(ImGuiTransformVisualizer)
EXPORT_NODE(ImGuiPathVisualizer)
DEFINE_PLUGIN_ENTRY()