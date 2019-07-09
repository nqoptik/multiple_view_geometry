#pragma once

#ifndef _LOADIMAGES_H_
#define _LOADIMAGES_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

std::string intToImageName(int number, std::string extenstion);
std::vector<cv::Mat> loadImages(std::string path, int first, int last);

#endif /* _LOADIMAGES_H_ */
