cmake_minimum_required(VERSION 3.1)

project(multiple_view_geometry)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV REQUIRED)

## Specify additional locations of header files
include_directories(${CMAKE_SOURCE_DIR}/include)

## Declare a C++ executable
add_executable(structure_from_motion
               src/featuresmatching.cpp
               src/geometry.cpp
               src/loadimages.cpp
               src/norm.cpp
               src/structure_from_motion.cpp)

add_executable(pose_estimation
               src/featuresmatching.cpp
               src/geometry.cpp
               src/loadimages.cpp
               src/norm.cpp
               src/pose_estimation.cpp)

add_executable(pmvs src/geometry.cpp src/norm.cpp src/pmvs.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(structure_from_motion ${OpenCV_LIBS})

target_link_libraries(pose_estimation ${OpenCV_LIBS})

target_link_libraries(pmvs ${OpenCV_LIBS})
