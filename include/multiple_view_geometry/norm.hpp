#ifndef NORM_HPP
#define NORM_HPP

#include <opencv2/core/core.hpp>

double get_euclid_distance(cv::Point2d point_1, cv::Point2d point_2);
double get_euclid_distance(cv::Point3d point_1, cv::Point3d point_2);

#endif // NORM_HPP
