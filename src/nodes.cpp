#include <fins/node.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "global_viewer.hpp"
#include "utils.hpp"

using namespace viz;

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
    register_input<0, pcl::PointCloud<pcl::PointXYZ>::Ptr>(
        "cloud", &ImGuiCloudXYZVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZ>::Ptr> &msg) {
    if (!msg.data || !(*msg.data) || (*msg.data)->empty())
      return;
    auto &cloud = *msg.data;
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
    register_input<0, pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "cloud", &ImGuiCloudXYZIVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    if (!msg.data || !(*msg.data) || (*msg.data)->empty())
      return;
    auto &cloud = *msg.data;
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
    register_input<0, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(
        "cloud", &ImGuiCloudRGBVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &msg) {
    if (!msg.data || !(*msg.data) || (*msg.data)->empty())
      return;
    auto &cloud = *msg.data;
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

// 导出节点
EXPORT_NODE(ImGuiCloudXYZVisualizer)
EXPORT_NODE(ImGuiCloudXYZIVisualizer)
EXPORT_NODE(ImGuiCloudRGBVisualizer)
DEFINE_PLUGIN_ENTRY()