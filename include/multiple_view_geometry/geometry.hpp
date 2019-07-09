#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

#include "multiple_view_geometry/norm.hpp"

cv::Point3d cvCross(cv::Point3d a, cv::Point3d b);
cv::Mat_<double> cvRotationBetweenVectors(cv::Point3d a, cv::Point3d b);
cv::Mat_<double> cvLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);
cv::Mat_<double> cvIterativeLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);
void cv3DAffineEstimation(std::vector<cv::Point3d> src, std::vector<cv::Point3d> dst, cv::Mat_<double>& R, cv::Mat_<double>& T);
void cvIterative3DAffineEstimation(std::vector<cv::Point3d> src, std::vector<cv::Point3d> dst, cv::Mat_<double>& R, cv::Mat_<double>& T);

#endif  // GEOMETRY_HPP
