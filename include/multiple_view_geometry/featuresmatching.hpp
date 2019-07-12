#ifndef FEATURESMATCHING_HPP
#define FEATURESMATCHING_HPP

#include <iostream>

#include <opencv2/xfeatures2d.hpp>

void brute_force_match_descriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void flann_match_descriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios);
void choose_matches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, std::vector<float> sortedRatios, unsigned int noMatches, std::vector<cv::DMatch>& acceptedMatches);

#endif // FEATURESMATCHING_HPP
