#include "multiple_view_geometry/featuresmatching.hpp"

void brute_force_match_descriptors(cv::Mat descriptors_0,
                                   cv::Mat descriptors_1,
                                   std::vector<std::vector<cv::DMatch>>& matches,
                                   std::vector<float>& ratios,
                                   std::vector<float>& sorted_ratios)
{
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors_0, descriptors_1, matches, 2);

    for (size_t i = 0; i < matches.size(); ++i)
    {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }

    sorted_ratios = ratios;
    std::sort(sorted_ratios.begin(), sorted_ratios.end());
}

void flann_match_descriptors(cv::Mat descriptors_0,
                             cv::Mat descriptors_1,
                             std::vector<std::vector<cv::DMatch>>& matches,
                             std::vector<float>& ratios,
                             std::vector<float>& sorted_ratios)
{
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descriptors_0, descriptors_1, matches, 2);

    for (size_t i = 0; i < matches.size(); ++i)
    {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }

    sorted_ratios = ratios;
    std::sort(sorted_ratios.begin(), sorted_ratios.end());
}

void choose_matches(std::vector<std::vector<cv::DMatch>> matches,
                    std::vector<float> ratios,
                    std::vector<float> sorted_ratios,
                    size_t number_of_matches,
                    std::vector<cv::DMatch>& accepted_matches)
{
    accepted_matches.clear();
    float accepted_ratio = sorted_ratios[MIN(number_of_matches - 1, sorted_ratios.size() - 1)];
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (ratios[i] <= accepted_ratio)
        {
            accepted_matches.push_back(matches[i][0]);
        }
    }
}
