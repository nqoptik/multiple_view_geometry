#ifndef FEATURESMATCHING_HPP
#define FEATURESMATCHING_HPP

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>

void brute_force_match_descriptors(cv::Mat descriptors_0,
                                   cv::Mat descriptors_1,
                                   std::vector<std::vector<cv::DMatch>>& matches,
                                   std::vector<float>& ratios, std::vector<float>& sorted_ratios);
void flann_match_descriptors(cv::Mat descriptors_0,
                             cv::Mat descriptors_1,
                             std::vector<std::vector<cv::DMatch>>& matches,
                             std::vector<float>& ratios,
                             std::vector<float>& sorted_ratios);
void choose_matches(std::vector<std::vector<cv::DMatch>> matches,
                    std::vector<float> ratios,
                    std::vector<float> sorted_ratios,
                    size_t number_of_matches,
                    std::vector<cv::DMatch>& accepted_matches);

#endif // FEATURESMATCHING_HPP
