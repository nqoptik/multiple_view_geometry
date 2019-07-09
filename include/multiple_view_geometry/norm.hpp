#ifndef NORM_HPP
#define NORM_HPP

#include <opencv2/core/core.hpp>

double norm_2d(cv::Point2d p1, cv::Point2d p2);
double norm_2d(cv::Point3d p1, cv::Point3d p2);
double cvEuclidDistd(cv::Point3d p1, cv::Point3d p2);

#endif  // NORM_HPP
