cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(3d_reconstruction)

find_package(OpenCV REQUIRED)

include_directories (
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(structure_from_motion
    src/structure_from_motion.cpp
    src/featuresmatching.cpp
    src/geometry.cpp
    src/loadimages.cpp
    src/norm.cpp
)
target_link_libraries(structure_from_motion ${OpenCV_LIBS})

add_executable(pose_estimation
    src/pose_estimation.cpp
    src/featuresmatching.cpp
    src/geometry.cpp
    src/loadimages.cpp
    src/norm.cpp
)
target_link_libraries(pose_estimation ${OpenCV_LIBS})

add_executable(pmvs
    src/pmvs.cpp
    src/geometry.cpp
    src/norm.cpp
)
target_link_libraries(pmvs ${OpenCV_LIBS})
