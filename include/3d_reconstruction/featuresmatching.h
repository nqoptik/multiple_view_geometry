#pragma once

#ifndef _FEATURESMATCHING_H_
#define _FEATURESMATCHING_H_

#include <iostream>

#include <opencv2/xfeatures2d.hpp>

void BFMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void FLANNMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void chooseMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, std::vector<float> sortedRatios, unsigned int noMatches, std::vector<cv::DMatch>& acceptedMatches);

#endif /* _FEATURESMATCHING_H_ */
