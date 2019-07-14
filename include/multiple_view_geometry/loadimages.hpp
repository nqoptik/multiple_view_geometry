#ifndef LOADIMAGES_HPP
#define LOADIMAGES_HPP

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

std::string int_to_image_name(int number, std::string extenstion);
std::vector<cv::Mat> load_images(std::string path, int first, int last);

#endif // LOADIMAGES_HPP
