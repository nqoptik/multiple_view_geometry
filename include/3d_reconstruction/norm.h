#pragma once

#ifndef _NORM_H_
#define _NORM_H_

#include <opencv2/core/core.hpp>

double norm_2d(cv::Point2d p1, cv::Point2d p2);
double norm_2d(cv::Point3d p1, cv::Point3d p2);
double cvEuclidDistd(cv::Point3d p1, cv::Point3d p2);

#endif /* _NORM_H_ */
