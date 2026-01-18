cmake_minimum_required(VERSION 3.16)
project(imgui_pointcloud_visualizer)

find_package(PCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(rclcpp REQUIRED)
find_package(nav_msgs QUIET)
find_package(geometry_msgs QUIET)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_library(${PROJECT_NAME} SHARED
    src/camera.cpp
    src/nodes.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    fins_sdk
    ${PCL_LIBRARIES}
    glfw
    OpenGL::GL
    rclcpp::rclcpp
)

if(TARGET nav_msgs::nav_msgs)
    target_link_libraries(${PROJECT_NAME} PRIVATE nav_msgs::nav_msgs)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${nav_msgs_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${nav_msgs_LIBRARIES})
endif()

if(TARGET geometry_msgs::geometry_msgs)
    target_link_libraries(${PROJECT_NAME} PRIVATE geometry_msgs::geometry_msgs)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${geometry_msgs_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${geometry_msgs_LIBRARIES})
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PRIVATE GL_GLEXT_PROTOTYPES)