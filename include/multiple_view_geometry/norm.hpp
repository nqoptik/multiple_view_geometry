#ifndef NORM_HPP
#define NORM_HPP

#include <opencv2/core/core.hpp>

double get_euclid_distance(cv::Point2d p1, cv::Point2d p2);
double get_euclid_distance(cv::Point3d p1, cv::Point3d p2);

#endif // NORM_HPP
