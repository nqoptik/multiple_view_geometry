#include "3d_reconstruction/featuresmatching.h"

void BFMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios) {
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(des_0, des_1, matches, 2);

    for (unsigned int i = 0; i < matches.size(); i++) {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }

    sortedRatios = ratios;
    std::sort(sortedRatios.begin(), sortedRatios.end());
}

void FLANNMatchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios, std::vector<float>& sortedRatios) {
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(des_0, des_1, matches, 2);

    for (unsigned int i = 0; i < matches.size(); i++) {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }

    sortedRatios = ratios;
    std::sort(sortedRatios.begin(), sortedRatios.end());
}

void chooseMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, std::vector<float> sortedRatios, unsigned int noMatches, std::vector<cv::DMatch>& acceptedMatches) {
    acceptedMatches.clear();
    float acceptedRatio = sortedRatios[MIN(noMatches - 1, sortedRatios.size() - 1)];
    for (unsigned int i = 0; i < matches.size(); i++) {
        if (ratios[i] <= acceptedRatio) {
            acceptedMatches.push_back(matches[i][0]);
        }
    }
}
