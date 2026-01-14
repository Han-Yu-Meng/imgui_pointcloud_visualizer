# ImGui PointCloud Visualizer

A FINS-compatible library for cloud visualization using Dear ImGui and modern OpenGL. This package provides a centralized, singleton-based rendering window capable of handling high-frequency LiDAR data (e.g., FAST-LIO2) with features like infinite map building and signal decay.

## Nodes

### 1. ImGuiCloudXYZVisualizer
Visualizes geometric point clouds without color or intensity information.
- **Functionality**: 
  - Renders points in dark grey/black for high contrast against the white background.
  - Ideal for raw geometric data or map structures.
- **Input**:
  - `cloud` (`pcl::PointCloud<pcl::PointXYZ>::Ptr`): The geometric point cloud.
- **Parameters**:
  - `title` (string): The display name in the UI panel (default: "ImGuiCloudXYZVisualizer").

### 2. ImGuiCloudXYZIVisualizer
Visualizes point clouds containing intensity information.
- **Functionality**:
  - **Intensity Mapping**: Automatically maps the intensity field to a Rainbow color spectrum (Blue = Low Intensity, Red = High Intensity).
  - Dynamically calculates min/max intensity range per frame for optimal contrast.
- **Input**:
  - `cloud` (`pcl::PointCloud<pcl::PointXYZI>::Ptr`): Point cloud with intensity channel.
- **Parameters**:
  - `title` (string): The display name in the UI panel (default: "ImGuiCloudXYZIVisualizer").

### 3. ImGuiCloudRGBVisualizer
Visualizes colored point clouds (e.g., from RGB-D cameras or colored maps).
- **Functionality**:
  - Renders points using their native RGB values.
- **Input**:
  - `cloud` (`pcl::PointCloud<pcl::PointXYZRGB>::Ptr`): Point cloud with RGB color data.
- **Parameters**:
  - `title` (string): The display name in the UI panel (default: "ImGuiCloudRGBVisualizer").

## Viewer Features & Controls

The Global Viewer window offers an interactive "Scene Manager" panel with the following capabilities:

- **Camera Controls**:
  - **Left Drag**: Pan the view (Planar movement).
  - **Right Drag**: Orbit/Rotate the camera.
  - **Scroll**: Zoom in/out.
  - **Auto Center**: Automatically focuses the camera on the incoming point cloud.
- **Visualization Modes**:
  - **Real-time**: Displays the latest frame only.
  - **Decay**: Keeps history for $N$ seconds (Sliding Window), useful for visualizing trajectories.
  - **Infinite Map**: Accumulates points indefinitely using **Spatial Hashing** and **Voxel Filtering** to build a global map without memory explosion.
- **Performance**:
  - **Random Downsampling**: Limits the maximum points per frame to maintain high FPS on high-frequency inputs.
  - **Batched Rendering**: Uses ring-buffer VBOs to minimize CPU-GPU bandwidth usage.

## Dependencies

- **FINS SDK**
- **PCL (Point Cloud Library)**: For point cloud data structures.
- **OpenGL 3.3+**: Core profile required.
- **GLFW3**: For window management and input handling.
- **Dear ImGui**: For the user interface overlay.
- **Eigen3**: For matrix and vector operations.

## License

Copyright (c) 2025. IWIN-FINS Lab, Shanghai Jiao Tong University. All rights reserved.