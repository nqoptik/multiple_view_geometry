#ifndef FEATURESMATCHING_HPP
#define FEATURESMATCHING_HPP

#include <iostream>

#include <opencv2/xfeatures2d.hpp>

void BFMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void FLANNMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void chooseMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, std::vector<float> sortedRatios, unsigned int noMatches, std::vector<cv::DMatch>& acceptedMatches);

#endif // FEATURESMATCHING_HPP
