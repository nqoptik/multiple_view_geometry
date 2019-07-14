#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "multiple_view_geometry/geometry.hpp"
#include "multiple_view_geometry/norm.hpp"

const int maximum_image_size = 30;
const cv::Mat_<double> K = (cv::Mat_<double>(3, 3) << 2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1);
const std::string input_path = "input_images/";
const std::string output_path = "pointcloud.ply";
const int grid_size = 13;
const int cell_size = 5;
const float good_ratio = 0.55f;
const float minimum_correct_ratio = 0.25f;
const float step = 0.05f;
const int number_of_steps = 10;
const int minimum_pairs_for_fundamental_matrix = 20;
const int number_of_visible_patches = 3;
const double visible_cosin = 0.85;
const double good_photometric = 39;
const double good_expanding_photometric = 21;
const int number_of_iterations = 100;

struct Plane
{
    ///Ax + By + Cz + D = 0
    double A;
    double B;
    double C;
    double D;
};

struct Grid
{
    cv::Mat colour;
    cv::Point3d position[3];
};

struct ImgInfos
{
    cv::Mat image;
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
    cv::Mat_<double> R;
    cv::Mat_<double> T;
    cv::Mat_<double> P;
    cv::Point3d oc;
    cv::Mat C_0;
    cv::Mat C_1;
};

struct Patch
{
    cv::Point3d c;
    cv::Point3d n;
    std::vector<int> V;
    int R;
    double xInR;
    double yInR;
    Grid grid;
    double g;
    int index[maximum_image_size];
};

double get_vector_length(cv::Point3d x);
void load_images(std::vector<cv::Mat>& images);
void apply_pmvs(std::vector<cv::Mat> images);
void detect_keypoints_and_extract_descriptors(cv::Mat image, ImgInfos& iif);
void get_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches);
void get_first_seed_patch(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches, int& minimum_patch_index);
void match_descriptors(cv::Mat descriptors_0,
                       cv::Mat descriptors_1,
                       std::vector<std::vector<cv::DMatch>>& matches,
                       std::vector<float>& ratios);
void choose_matches(std::vector<std::vector<cv::DMatch>> matches,
                    std::vector<float> ratios,
                    float accepted_ratio,
                    std::vector<cv::DMatch>& accepted_matches);
void get_patch_from_matches(std::vector<ImgInfos> iifs,
                            std::vector<Patch>& patches,
                            int index_0,
                            int index_1,
                            std::vector<cv::DMatch> good_matches,
                            std::vector<cv::DMatch> correct_matches);
void get_fundamental_matrix(std::vector<cv::KeyPoint> kp_0,
                            std::vector<cv::KeyPoint> kp_1,
                            std::vector<cv::DMatch> correct_matches,
                            cv::Mat& F);
void triangulate_patches(std::vector<ImgInfos> iifs,
                         std::vector<Patch>& patches,
                         std::vector<cv::DMatch> good_matches,
                         cv::Matx34d P_0,
                         int index_0,
                         cv::Matx34d P_1,
                         int index_1);
void estimate_error_rate(std::vector<cv::Point3d> points_0,
                         std::vector<cv::Point3d> points_1,
                         cv::Mat_<double> R,
                         cv::Mat_<double> T,
                         double& error_rate);
void warp_patches(std::vector<Patch> patches_1, std::vector<Patch>& patches_0, cv::Mat_<double> R, cv::Mat_<double> T);
void get_more_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches, int minimum_patch_index);
void extract_image_information(std::vector<ImgInfos>& iifs);
void calculate_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches);
void get_visible_and_reference_images(std::vector<ImgInfos> iifs, Patch& patch);
void get_grid_position(std::vector<ImgInfos> iifs, Patch& patch);
void get_grid_colour(std::vector<ImgInfos> iifs, Patch& patch);
void get_photometric(std::vector<ImgInfos> iifs, Patch& patch);
void filter_seed_patches(std::vector<Patch>& patches);
void project_seed_patches_to_tmage_cells(std::vector<ImgInfos>& iifs, std::vector<Patch> seed_patches);
void mark_cells_to_expand(std::vector<ImgInfos>& iifs);
void expand_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches);
Plane get_plane(cv::Point3d point, cv::Point3d normal);
cv::Point3d get_intesection(cv::Point3d A, cv::Point3d B, Plane p);
cv::Point3d get_new_normal(cv::Point3d A, cv::Point3d old_normal_vector, cv::Point3d B);
void project_new_patches_to_image_cells(std::vector<ImgInfos>& iifs, std::vector<Patch> seed_patches, int first_new_patch_index);

void draw_patches(std::vector<ImgInfos> iifs, std::vector<Patch> patches);

int main()
{
    std::vector<cv::Mat> images;

    ///Load images
    load_images(images);
    std::cout << "Number of images: " << images.size() << std::endl;

    if (images.size() < 4)
    {
        std::cout << "Not enough images to run pmvs." << std::endl;
        return 0;
    }

    ///Reconstruct model using apply_pmvs
    apply_pmvs(images);

    return 0;
}

double get_vector_length(cv::Point3d x)
{
    return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
}

void load_images(std::vector<cv::Mat>& images)
{
    DIR* directory_ptr;
    directory_ptr = opendir(input_path.c_str());
    if (directory_ptr == NULL)
    {
        std::cout << "Directory not found." << std::endl;
        return;
    }

    struct dirent* dirent_ptr;
    std::vector<std::string> image_paths;
    while ((dirent_ptr = readdir(directory_ptr)) != NULL)
    {
        if (strcmp(dirent_ptr->d_name, ".") == 0 || strcmp(dirent_ptr->d_name, "..") == 0)
        {
            continue;
        }
        std::string image_path = input_path;
        image_path.append(dirent_ptr->d_name);
        image_paths.push_back(image_path);
    }
    closedir(directory_ptr);

    std::sort(image_paths.begin(), image_paths.end());
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        std::cout << image_paths[i] << std::endl;
        cv::Mat image = cv::imread(image_paths[i]);
        if (image.empty())
        {
            break;
        }
        images.push_back(image);
    }
}

void apply_pmvs(std::vector<cv::Mat> images)
{
    ///Detect key points and compute descriptors
    std::vector<ImgInfos> iifs;

    for (size_t i = 0; i < images.size(); ++i)
    {
        std::cout << "Number of key points in images " << i << ": ";
        ImgInfos iif;
        detect_keypoints_and_extract_descriptors(images[i], iif);
        iifs.push_back(iif);
        std::cout << iif.kp.size() << std::endl;
    }

    ///Get seed patches
    std::vector<Patch> seed_patches;
    get_seed_patches(iifs, seed_patches);
    std::cout << "Calculating images informations..." << std::endl;
    extract_image_information(iifs);
    std::cout << "Calculating seed patches..." << std::endl;
    calculate_seed_patches(iifs, seed_patches);
    std::cout << "Filtering seed patches..." << std::endl;
    filter_seed_patches(seed_patches);
    std::cout << "Number of seed patches: " << seed_patches.size() << std::endl;

    int first_new_patch_index = 0;
    for (int i = 0; i < number_of_iterations; ++i)
    {
        std::cout << "expand " << i << "..." << std::endl;
        project_new_patches_to_image_cells(iifs, seed_patches, first_new_patch_index);
        std::cout << "Marking cell to expand..." << std::endl;
        mark_cells_to_expand(iifs);
        first_new_patch_index = seed_patches.size();
        std::cout << "Expanding patches..." << std::endl;
        expand_patches(iifs, seed_patches);
        draw_patches(iifs, seed_patches);
    }
}

void detect_keypoints_and_extract_descriptors(cv::Mat image, ImgInfos& iif)
{
    iif.image = image;

    ///Detect key points using SIFT
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_0, keypoints_1;
    f2d->detect(image, keypoints_0);
    for (size_t i = 0; i < keypoints_0.size(); ++i)
    {
        float x = keypoints_0[i].pt.x;
        float y = keypoints_0[i].pt.y;
        if ((x > grid_size) && (x < image.cols - grid_size - 1) &&
            (y > grid_size) && (y < image.rows - grid_size - 1))
        {
            keypoints_1.push_back(keypoints_0[i]);
        }
    }
    iif.kp = keypoints_1;

    ///Compute descriptors using SIFT
    cv::Mat descriptors;
    f2d->compute(image, keypoints_1, descriptors);
    iif.des = descriptors;
}

void get_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches)
{
    int minimum_patch_index;
    get_first_seed_patch(iifs, seed_patches, minimum_patch_index);
    get_more_seed_patches(iifs, seed_patches, minimum_patch_index);
}

void get_first_seed_patch(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches, int& minimum_patch_index)
{
    ///Match descriptors using brute force
    std::vector<std::vector<cv::DMatch>> matches_0, matches_1;
    std::vector<float> ratios_0, ratios_1;
    match_descriptors(iifs[0].des, iifs[1].des, matches_0, ratios_0);
    match_descriptors(iifs[1].des, iifs[2].des, matches_1, ratios_1);

    ///Chose good matches
    std::vector<cv::DMatch> good_matches_0, good_matches_1;
    choose_matches(matches_0, ratios_0, good_ratio, good_matches_0);
    choose_matches(matches_1, ratios_1, good_ratio, good_matches_1);

    ///Find parameters for CV_FM_LMEDS method
    double minimum_error_rate = 10;
    float best_correct_ratio_0, best_correct_ratio_1;
    std::vector<Patch> best_patches_0, best_patches_1;
    cv::Mat_<double> bestR, bestT;

    for (int loop_i = 0; loop_i < number_of_steps; ++loop_i)
    {
        std::vector<Patch> patches_0;
        std::vector<cv::DMatch> correct_matches_0;
        float correct_ratio_0 = minimum_correct_ratio + loop_i * step;
        choose_matches(matches_0, ratios_0, correct_ratio_0, correct_matches_0);
        if (correct_matches_0.size() < minimum_pairs_for_fundamental_matrix)
        {
            continue;
        }
        get_patch_from_matches(iifs, patches_0, 0, 1,
                               good_matches_0, correct_matches_0);

        for (int loop_j = 0; loop_j < number_of_steps; ++loop_j)
        {
            std::vector<Patch> patches_1;
            std::vector<cv::DMatch> correct_matches_1;
            float correct_ratio_1 = minimum_correct_ratio + loop_j * step;
            choose_matches(matches_1, ratios_1, correct_ratio_1, correct_matches_1);
            if (correct_matches_1.size() < minimum_pairs_for_fundamental_matrix)
            {
                continue;
            }
            get_patch_from_matches(iifs, patches_1, 1, 2,
                                   good_matches_1, correct_matches_1);

            ///Take corresponding points
            std::vector<cv::Point3d> points_0, points_1;
            for (size_t i = 0; i < patches_0.size(); ++i)
            {
                for (size_t j = 0; j < patches_1.size(); ++j)
                {
                    if (patches_1[j].index[1] == patches_0[i].index[1])
                    {
                        points_0.push_back(patches_0[i].c);
                        points_1.push_back(patches_1[j].c);
                        break;
                    }
                }
            }

            ///Estimation rotation, translation and scaling between two point clouds
            cv::Mat_<double> R, T;
            estimate_iterative_3d_affine(points_1, points_0, R, T);

            ///Estimate error rate of R and T
            double error_rate;
            estimate_error_rate(points_0, points_1, R, T, error_rate);

            ///Find min error rate
            if (error_rate < minimum_error_rate)
            {
                minimum_error_rate = error_rate;
                best_correct_ratio_0 = correct_ratio_0;
                best_correct_ratio_1 = correct_ratio_1;
                best_patches_0 = patches_0;
                best_patches_1 = patches_1;
                bestR = R;
                bestT = T;
            }
        }
    }

    iifs[1].R = bestR;
    iifs[1].T = bestT;

    cv::Mat_<double> oc_ = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat_<double> oc = bestR * oc_ + bestT;
    iifs[1].oc = cv::Point3d(oc.at<double>(0, 0), oc.at<double>(1, 0), oc.at<double>(2, 0));
    std::cout << minimum_error_rate << " : " << best_correct_ratio_0 << " : " << best_correct_ratio_1 << std::endl;
    minimum_patch_index = best_patches_0.size();

    ///Warp patches with the best R and T
    warp_patches(best_patches_1, best_patches_0, bestR, bestT);
    seed_patches = best_patches_0;
}

void match_descriptors(cv::Mat descriptors_0,
                       cv::Mat descriptors_1,
                       std::vector<std::vector<cv::DMatch>>& matches,
                       std::vector<float>& ratios)
{
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(descriptors_0, descriptors_1, matches, 2);

    for (size_t i = 0; i < matches.size(); ++i)
    {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }
}

void choose_matches(std::vector<std::vector<cv::DMatch>> matches,
                    std::vector<float> ratios,
                    float accepted_ratio,
                    std::vector<cv::DMatch>& accepted_matches)
{
    accepted_matches.clear();
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (ratios[i] <= accepted_ratio)
        {
            accepted_matches.push_back(matches[i][0]);
        }
    }
}

void get_patch_from_matches(std::vector<ImgInfos> iifs,
                            std::vector<Patch>& patches,
                            int index_0,
                            int index_1,
                            std::vector<cv::DMatch> good_matches,
                            std::vector<cv::DMatch> correct_matches)
{
    ///Find fundamental matrix and essential matrix
    cv::Mat F;
    get_fundamental_matrix(iifs[index_0].kp, iifs[index_1].kp, correct_matches, F);
    cv::Mat_<double> E = K.t() * F * K;

    ///Get perspective matrices
    cv::Matx34d P_0(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

    cv::SVD svd(E);
    cv::Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
    cv::Mat_<double> R = svd.u * cv::Mat(W) * svd.vt;
    cv::Mat_<double> t = svd.u.col(2);

    cv::Matx34d P_1 = cv::Matx34d(
        R(0, 0), R(0, 1), R(0, 2), t(0),
        R(1, 0), R(1, 1), R(1, 2), t(1),
        R(2, 0), R(2, 1), R(2, 2), t(2));

    ///Triangulate point cloud
    triangulate_patches(iifs, patches, good_matches, P_0, index_0, P_1, index_1);
}

void get_fundamental_matrix(std::vector<cv::KeyPoint> kp_0,
                            std::vector<cv::KeyPoint> kp_1,
                            std::vector<cv::DMatch> correct_matches,
                            cv::Mat& F)
{
    ///Take corresponding points
    std::vector<cv::Point2f> points_0, points_1;
    for (size_t i = 0; i < correct_matches.size(); ++i)
    {
        points_0.push_back(kp_0[correct_matches[i].queryIdx].pt);
        points_1.push_back(kp_1[correct_matches[i].trainIdx].pt);
    }

    ///Find fundamental matrix from corresponding points
    cv::Mat mask;
    F = cv::findFundamentalMat(points_0, points_1, CV_FM_LMEDS, 3., 0.99, mask);
}

void triangulate_patches(std::vector<ImgInfos> iifs,
                         std::vector<Patch>& patches,
                         std::vector<cv::DMatch> good_matches,
                         cv::Matx34d P_0,
                         int index_0,
                         cv::Matx34d P_1,
                         int index_1)
{
    patches.clear();
    for (size_t i = 0; i < good_matches.size(); ++i)
    {
        Patch patch;

        for (int j = 0; j < maximum_image_size; ++j)
        {
            patch.index[j] = -1;
        }

        ///Mark image index for each point in cloud
        patch.index[index_0] = good_matches[i].queryIdx;
        patch.index[index_1] = good_matches[i].trainIdx;

        cv::Point2d point_0(iifs[index_0].kp[good_matches[i].queryIdx].pt.x, iifs[index_0].kp[good_matches[i].queryIdx].pt.y);
        cv::Point2d point_1(iifs[index_1].kp[good_matches[i].trainIdx].pt.x, iifs[index_1].kp[good_matches[i].trainIdx].pt.y);

        ///Estimate 3d position
        cv::Point3d u_0(point_0.x, point_0.y, 1.0);
        cv::Point3d u_1(point_1.x, point_1.y, 1.0);
        cv::Mat_<double> um_0 = K.inv() * cv::Mat_<double>(u_0);
        cv::Mat_<double> um_1 = K.inv() * cv::Mat_<double>(u_1);
        u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
        u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
        cv::Mat_<double> point3d = iterative_linear_ls_triangulation(u_0, P_0, u_1, P_1);

        patch.c = cv::Point3d(point3d(0), point3d(1), point3d(2));
        patch.R = index_0;
        patches.push_back(patch);
    }
}

void estimate_error_rate(std::vector<cv::Point3d> points_0,
                         std::vector<cv::Point3d> points_1,
                         cv::Mat_<double> R,
                         cv::Mat_<double> T, double& error_rate)
{
    std::vector<cv::Point3d> pts_0_;
    for (size_t i = 0; i < points_1.size(); ++i)
    {
        cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << points_1[i].x, points_1[i].y, points_1[i].z);
        cv::Mat_<double> pt_0_(3, 1);
        pt_0_ = R * pt_1 + T;
        pts_0_.push_back(cv::Point3d(pt_0_.at<double>(0, 0),
                                     pt_0_.at<double>(1, 0),
                                     pt_0_.at<double>(2, 0)));
    }

    double averge_error = 0;
    for (size_t i = 0; i < points_0.size(); ++i)
    {
        averge_error += get_euclid_distance(pts_0_[i], points_0[i]);
    }

    cv::Point3d centroid(0, 0, 0);
    for (size_t i = 0; i < points_0.size(); ++i)
    {
        centroid += points_0[i];
    }

    centroid.x /= points_0.size();
    centroid.y /= points_0.size();
    centroid.z /= points_0.size();

    double averge_range = 0;
    for (size_t i = 0; i < points_0.size(); ++i)
    {
        averge_range += get_euclid_distance(points_0[i], centroid);
    }

    error_rate = averge_error / averge_range;
}

void warp_patches(std::vector<Patch> patches_1, std::vector<Patch>& patches_0, cv::Mat_<double> R, cv::Mat_<double> T)
{
    for (size_t i = 0; i < patches_1.size(); ++i)
    {
        cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << patches_1[i].c.x, patches_1[i].c.y, patches_1[i].c.z);
        cv::Mat_<double> pt_0_(3, 1);
        pt_0_ = R * pt_1 + T;
        patches_1[i].c.x = pt_0_.at<double>(0, 0);
        patches_1[i].c.y = pt_0_.at<double>(1, 0);
        patches_1[i].c.z = pt_0_.at<double>(2, 0);
        patches_0.push_back(patches_1[i]);
    }
}

void get_more_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches, int minimum_patch_index)
{
    for (size_t index_1 = 3; index_1 < iifs.size(); index_1++)
    {
        int index_0 = index_1 - 1;
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<float> ratios;
        match_descriptors(iifs[index_0].des, iifs[index_1].des, matches, ratios);
        std::vector<cv::DMatch> good_matches;
        choose_matches(matches, ratios, good_ratio, good_matches);
        std::vector<Patch> bestPatches;
        float bestCorrectRatio = minimum_correct_ratio;
        double minimum_error_rate = 10;
        cv::Mat_<double> bestR, bestT;

        for (int loop = 0; loop < number_of_steps; loop++)
        {
            std::vector<Patch> patches;
            std::vector<cv::DMatch> correct_matches;
            float correctRatio = minimum_correct_ratio + loop * step;
            choose_matches(matches, ratios, correctRatio, correct_matches);
            if (correct_matches.size() < minimum_pairs_for_fundamental_matrix)
            {
                continue;
            }
            get_patch_from_matches(iifs, patches, index_0, index_1,
                                   good_matches, correct_matches);
            std::vector<cv::Point3d> points_0, points_1;

            for (size_t i = minimum_patch_index; i < seed_patches.size(); ++i)
            {
                if (seed_patches[i].index[index_0] != -1)
                {
                    for (size_t j = 0; j < patches.size(); ++j)
                    {
                        if (patches[j].index[index_0] == seed_patches[i].index[index_0])
                        {
                            points_0.push_back(seed_patches[i].c);
                            points_1.push_back(patches[j].c);
                            break;
                        }
                    }
                }
            }

            ///Estimation rotation, translation and scaling between two point clouds
            cv::Mat_<double> R, T;
            estimate_iterative_3d_affine(points_1, points_0, R, T);

            ///Estimate error rate of R and T
            double error_rate;
            estimate_error_rate(points_0, points_1, R, T, error_rate);

            ///Find min error rate
            if (error_rate < minimum_error_rate)
            {
                minimum_error_rate = error_rate;
                bestCorrectRatio = correctRatio;
                bestPatches = patches;
                bestR = R;
                bestT = T;
            }
        }

        iifs[index_0].R = bestR;
        iifs[index_0].T = bestT;

        cv::Mat_<double> oc_ = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::Mat_<double> oc = bestR * oc_ + bestT;
        iifs[index_0].oc = cv::Point3d(oc.at<double>(0, 0), oc.at<double>(1, 0), oc.at<double>(2, 0));
        std::cout << minimum_error_rate << "  " << bestCorrectRatio << std::endl;

        ///Warp patches with the best R and T
        minimum_patch_index = seed_patches.size();
        warp_patches(bestPatches, seed_patches, bestR, bestT);
    }
}

void extract_image_information(std::vector<ImgInfos>& iifs)
{
    iifs[0].R = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                 0, 1, 0,
                 0, 0, 1);
    iifs[0].T = (cv::Mat_<double>(3, 1) << 0,
                 0,
                 0);
    iifs[0].P = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0);
    iifs[0].oc = cv::Point3d(0, 0, 0);

    ///Calculate projection matrix
    for (size_t i = 1; i < iifs.size() - 1; ++i)
    {
        cv::Mat_<double> R, T;
        R = iifs[i].R.inv();
        T = -iifs[i].R.inv() * iifs[i].T;
        iifs[i].R = R;
        iifs[i].T = T;
        iifs[i].P = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), T(0, 0),
                     R(1, 0), R(1, 1), R(1, 2), T(1, 0),
                     R(2, 0), R(2, 1), R(2, 2), T(2, 0));
    }

    ///Cells
    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        iifs[i].C_0 = cv::Mat::zeros(floor(iifs[i].image.rows / cell_size) + 1,
                                     floor(iifs[i].image.cols / cell_size) + 1, CV_32SC1);
    }
}

void calculate_seed_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches)
{
    ///Calculate normal vector
    for (size_t i = 0; i < seed_patches.size(); ++i)
    {
        seed_patches[i].n = iifs[seed_patches[i].R].oc - seed_patches[i].c;
        double d = get_vector_length(seed_patches[i].n);
        seed_patches[i].n.x /= d;
        seed_patches[i].n.y /= d;
        seed_patches[i].n.z /= d;
    }

    ///Calculate visible images
    for (size_t i = 0; i < seed_patches.size(); ++i)
    {
        get_visible_and_reference_images(iifs, seed_patches[i]);
        get_grid_position(iifs, seed_patches[i]);
        get_grid_colour(iifs, seed_patches[i]);
        get_photometric(iifs, seed_patches[i]);
    }
}

void get_visible_and_reference_images(std::vector<ImgInfos> iifs, Patch& patch)
{
    double maximum_cosin = 0;
    int RIdx = 0;
    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        cv::Point3d vt;
        vt = iifs[i].oc - patch.c;
        double d = get_vector_length(vt);
        vt.x /= d;
        vt.y /= d;
        vt.z /= d;
        double cos_ = patch.n.x * vt.x +
                      patch.n.y * vt.y + patch.n.z * vt.z;
        if (cos_ > visible_cosin)
        {
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.c.x,
                                  patch.c.y,
                                  patch.c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[i].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            if ((x.at<double>(0, 0) > grid_size) &&
                (x.at<double>(0, 0) < iifs[i].image.cols - grid_size - 1) &&
                (x.at<double>(1, 0) > grid_size) &&
                (x.at<double>(1, 0) < iifs[i].image.rows - grid_size - 1))
            {
                patch.V.push_back(i);
            }
        }
        if (cos_ > maximum_cosin)
        {
            maximum_cosin = cos_;
            RIdx = i;
        }
    }
    patch.R = RIdx;
}

void get_grid_position(std::vector<ImgInfos> iifs, Patch& patch)
{
    cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.c.x,
                          patch.c.y,
                          patch.c.z,
                          1);
    cv::Mat_<double> P = K * iifs[patch.R].P;
    cv::Mat_<double> KR = (cv::Mat_<double>(3, 3) << P(0, 0), P(0, 1), P(0, 2),
                           P(1, 0), P(1, 1), P(1, 2),
                           P(2, 0), P(2, 1), P(2, 2));
    cv::Mat_<double> KT = (cv::Mat_<double>(3, 1) << P(0, 3),
                           P(1, 3),
                           P(2, 3));
    cv::Mat_<double> x = P * X;

    ///(0, 0)
    cv::Mat_<double> x_ = cv::Mat_<double>(3, 1);
    x_.at<double>(0, 0) = x.at<double>(0, 0) - (grid_size / 2) * x.at<double>(2, 0);
    x_.at<double>(1, 0) = x.at<double>(1, 0) - (grid_size / 2) * x.at<double>(2, 0);
    x_.at<double>(2, 0) = x.at<double>(2, 0);
    cv::Mat_<double> xyz_1 = KR.inv() * (x_ - KT);
    cv::Point3d p_1 = cv::Point3d(xyz_1(0, 0), xyz_1(1, 0), xyz_1(2, 0));

    ///(x, y) = (0, grid_size - 1)
    x_.at<double>(0, 0) = x.at<double>(0, 0) - (grid_size / 2) * x.at<double>(2, 0);
    x_.at<double>(1, 0) = x.at<double>(1, 0) + (grid_size - 1 - grid_size / 2) * x.at<double>(2, 0);
    x_.at<double>(2, 0) = x.at<double>(2, 0);
    cv::Mat_<double> xyz_2 = KR.inv() * (x_ - KT);
    cv::Point3d p_2 = cv::Point3d(xyz_2(0, 0), xyz_2(1, 0), xyz_2(2, 0));

    ///Rotate position with normal vector
    cv::Point3d nk = get_cross_product(p_2 - p_1, patch.c - p_1);
    double d = get_vector_length(nk);
    nk.x /= d;
    nk.y /= d;
    nk.z /= d;
    cv::Mat_<double> R = get_rotation_matrix(nk, patch.n);
    xyz_1 = R * xyz_1;
    xyz_2 = R * xyz_2;
    patch.grid.position[0] = patch.c;
    patch.grid.position[1] = cv::Point3d(xyz_1(0, 0), xyz_1(1, 0), xyz_1(2, 0));
    patch.grid.position[2] = cv::Point3d(xyz_2(0, 0), xyz_2(1, 0), xyz_2(2, 0));
}

void get_grid_colour(std::vector<ImgInfos> iifs, Patch& patch)
{
    int i_index = patch.R;
    cv::Point2f source_points[3];
    cv::Point2f destination_points[3];

    ///center
    cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.grid.position[0].x,
                          patch.grid.position[0].y,
                          patch.grid.position[0].z,
                          1);
    cv::Mat_<double> x = K * iifs[i_index].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);

    ///xInR and yInR
    patch.xInR = x.at<double>(0, 0);
    patch.yInR = x.at<double>(1, 0);

    source_points[0] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    destination_points[0] = cv::Point2f(grid_size / 2, grid_size / 2);

    ///(0, 0)
    X = (cv::Mat_<double>(4, 1) << patch.grid.position[1].x,
         patch.grid.position[1].y,
         patch.grid.position[1].z,
         1);
    x = K * iifs[i_index].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);
    source_points[1] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    destination_points[1] = cv::Point2f(0, 0);

    ///(x, y) = (0, grid_size - 1)
    X = (cv::Mat_<double>(4, 1) << patch.grid.position[2].x,
         patch.grid.position[2].y,
         patch.grid.position[2].z,
         1);
    x = K * iifs[i_index].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);
    source_points[2] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    destination_points[2] = cv::Point2f(0, grid_size - 1);

    ///Find colour using affine transformation
    cv::Mat affine_matrix = getAffineTransform(source_points, destination_points);
    patch.grid.colour = cv::Mat::zeros(grid_size, grid_size, CV_8UC3);
    cv::warpAffine(iifs[i_index].image, patch.grid.colour, affine_matrix, patch.grid.colour.size());
}

void get_photometric(std::vector<ImgInfos> iifs, Patch& patch)
{
    if (patch.V.size() <= number_of_visible_patches)
    {
        patch.g = good_photometric + 1;
        return;
    }

    double g = 0;
    for (size_t i = 0; i < patch.V.size(); ++i)
    {
        int i_index = patch.V[i];
        if (i_index != patch.R)
        {
            cv::Point2f source_points[3];
            cv::Point2f destination_points[3];

            ///center
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.grid.position[0].x,
                                  patch.grid.position[0].y,
                                  patch.grid.position[0].z,
                                  1);
            cv::Mat_<double> x = K * iifs[i_index].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            source_points[0] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            destination_points[0] = cv::Point2f(grid_size / 2, grid_size / 2);

            ///(0, 0)
            X = (cv::Mat_<double>(4, 1) << patch.grid.position[1].x,
                 patch.grid.position[1].y,
                 patch.grid.position[1].z,
                 1);
            x = K * iifs[i_index].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            source_points[1] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            destination_points[1] = cv::Point2f(0, 0);

            ///(grid_size - 1, 0) -> (x, y) = (0, grid_size - 1)
            X = (cv::Mat_<double>(4, 1) << patch.grid.position[2].x,
                 patch.grid.position[2].y,
                 patch.grid.position[2].z,
                 1);
            x = K * iifs[i_index].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            source_points[2] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            destination_points[2] = cv::Point2f(0, grid_size - 1);

            ///Find colour using affine_matrix transformation
            cv::Mat affine_matrix = getAffineTransform(source_points, destination_points);
            cv::Mat reprojectColor = cv::Mat::zeros(grid_size, grid_size, CV_8UC3);
            cv::warpAffine(iifs[i_index].image, reprojectColor, affine_matrix, reprojectColor.size());
            g += cv::norm(reprojectColor - patch.grid.colour, cv::NORM_L1);
        }
    }
    patch.g = g / (grid_size * grid_size * (patch.V.size() - 1));
}

void filter_seed_patches(std::vector<Patch>& patches)
{
    ///Eliminate incorrect patches using visibility rule
    for (size_t i = 0; i < patches.size(); ++i)
    {
        if ((patches[i].V.size() <= number_of_visible_patches) || (patches[i].g > good_photometric))
        {
            patches.erase(patches.begin() + i);
            i--;
        }
    }
}

void project_seed_patches_to_tmage_cells(std::vector<ImgInfos>& iifs, std::vector<Patch> seed_patches)
{
    ///Project seed patches to image cells
    for (size_t i = 0; i < seed_patches.size(); ++i)
    {
        for (unsigned j = 0; j < seed_patches[i].V.size(); ++j)
        {
            int i_index = seed_patches[i].V[j];
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seed_patches[i].c.x,
                                  seed_patches[i].c.y,
                                  seed_patches[i].c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[i_index].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            int row = floor(x.at<double>(1, 0) / cell_size);
            int col = floor(x.at<double>(0, 0) / cell_size);
            if (row > 0 && row < iifs[j].C_0.rows &&
                col > 0 && col < iifs[j].C_0.cols)
            {
                if (i_index == seed_patches[i].R)
                {
                    iifs[i_index].C_0.at<int>(row, col) = i;
                }
                else
                {
                    iifs[i_index].C_0.at<int>(row, col) = -1;
                }
            }
            else
            {
                std::cout << "Patch out of visible images." << std::endl;
            }
        }
    }
}

void mark_cells_to_expand(std::vector<ImgInfos>& iifs)
{
    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        iifs[i].C_1 = cv::Mat::zeros(floor(iifs[i].image.rows / cell_size) + 1,
                                     floor(iifs[i].image.cols / cell_size) + 1, CV_32SC1);
    }

    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        for (int k = 1; k < iifs[i].C_0.rows - 1; ++k)
        {
            for (int l = 1; l < iifs[i].C_0.cols - 1; ++l)
            {
                if (iifs[i].C_0.at<int>(k, l) == 0)
                {
                    if (iifs[i].C_0.at<int>(k, l + 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k, l + 1);
                    }
                    else if (iifs[i].C_0.at<int>(k, l - 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k, l - 1);
                    }
                    else if (iifs[i].C_0.at<int>(k + 1, l) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l);
                    }
                    else if (iifs[i].C_0.at<int>(k - 1, l) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l);
                    }
                    else if (iifs[i].C_0.at<int>(k - 1, l - 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l - 1);
                    }
                    else if (iifs[i].C_0.at<int>(k - 1, l + 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l + 1);
                    }
                    else if (iifs[i].C_0.at<int>(k + 1, l - 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l - 1);
                    }
                    else if (iifs[i].C_0.at<int>(k + 1, l + 1) > 0)
                    {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l + 1);
                    }
                }
            }
        }
    }
}

void expand_patches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seed_patches)
{
    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        for (int k = 1; k < iifs[i].C_1.rows - 1; ++k)
        {
            for (int l = 1; l < iifs[i].C_1.cols - 1; ++l)
            {
                int index = iifs[i].C_1.at<int>(k, l);
                if (index > 0)
                {
                    cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seed_patches[index].c.x,
                                          seed_patches[index].c.y,
                                          seed_patches[index].c.z,
                                          1);
                    cv::Mat_<double> P = K * iifs[seed_patches[index].R].P;
                    cv::Mat_<double> KR = (cv::Mat_<double>(3, 3) << P(0, 0), P(0, 1), P(0, 2),
                                           P(1, 0), P(1, 1), P(1, 2),
                                           P(2, 0), P(2, 1), P(2, 2));
                    cv::Mat_<double> KT = (cv::Mat_<double>(3, 1) << P(0, 3),
                                           P(1, 3),
                                           P(2, 3));
                    cv::Mat_<double> x = P * X;
                    cv::Mat_<double> x_ = cv::Mat_<double>(3, 1);
                    x_.at<double>(0, 0) = (l * cell_size + cell_size / 2) * x.at<double>(2, 0);
                    x_.at<double>(1, 0) = (k * cell_size + cell_size / 2) * x.at<double>(2, 0);
                    x_.at<double>(2, 0) = x.at<double>(2, 0);
                    cv::Mat_<double> XYZ = KR.inv() * (x_ - KT);
                    cv::Point3d nK = cv::Point3d(XYZ(0, 0), XYZ(1, 0), XYZ(2, 0));

                    ///Optimize the best centroid of patch
                    cv::Point3d O, C, N;
                    O = iifs[seed_patches[index].R].oc;
                    C = seed_patches[index].c;
                    Plane plane = get_plane(seed_patches[index].c, seed_patches[index].n);
                    N = get_intesection(O, nK, plane);
                    double dCN = get_euclid_distance(C, N);
                    double dON = get_euclid_distance(O, N);
                    cv::Point3d vON = (0.25 * dCN / dON) * (N - O);

                    double bestg = good_expanding_photometric + 1;

                    Patch best_new_patch;
                    for (int loop_p = -2; loop_p < 3; ++loop_p)
                    {
                        Patch new_patch;
                        cv::Point3d loopN = N + loop_p * vON;
                        new_patch.c = loopN;
                        cv::Point3d normal = get_new_normal(C, seed_patches[i].n, loopN);
                        double d = get_vector_length(normal);
                        normal.x /= d;
                        normal.y /= d;
                        normal.z /= d;
                        new_patch.n = normal;
                        get_visible_and_reference_images(iifs, new_patch);
                        get_grid_position(iifs, new_patch);
                        get_grid_colour(iifs, new_patch);
                        get_photometric(iifs, new_patch);

                        if (new_patch.g < bestg)
                        {
                            bestg = new_patch.g;
                            best_new_patch = new_patch;
                        }
                    }

                    if (bestg <= good_expanding_photometric)
                    {
                        seed_patches.push_back(best_new_patch);
                    }
                }
            }
            std::cout << "Done cell in images " << i << "/" << iifs.size() - 2 << ", row " << k << "/" << iifs[i].C_1.rows - 2 << std::endl;
        }
    }
}

Plane get_plane(cv::Point3d point, cv::Point3d normal)
{
    Plane p;
    p.A = normal.x;
    p.B = normal.y;
    p.C = normal.z;
    p.D = -(normal.x * point.x + normal.y * point.y + normal.z * point.z);
    return p;
}

cv::Point3d get_intesection(cv::Point3d A, cv::Point3d B, Plane p)
{
    cv::Point3d AB = B - A;
    double t = -(p.A * A.x + p.B * A.y + p.C * A.z + p.D) /
               (p.A * AB.x + p.B * AB.y + p.C * AB.z);
    cv::Point3d intersection = A + t * AB;
    return intersection;
}

cv::Point3d get_new_normal(cv::Point3d A, cv::Point3d old_normal_vector, cv::Point3d B)
{
    cv::Point3d AB = B - A;
    return get_cross_product(AB, get_cross_product(old_normal_vector, AB));
}

void project_new_patches_to_image_cells(std::vector<ImgInfos>& iifs, std::vector<Patch> seed_patches, int first_new_patch_index)
{
    ///Project seed patches to image cells
    for (size_t i = first_new_patch_index; i < seed_patches.size(); ++i)
    {
        for (unsigned j = 0; j < seed_patches[i].V.size(); ++j)
        {
            int i_index = seed_patches[i].V[j];
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seed_patches[i].c.x,
                                  seed_patches[i].c.y,
                                  seed_patches[i].c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[i_index].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            int row = floor(x.at<double>(1, 0) / cell_size);
            int col = floor(x.at<double>(0, 0) / cell_size);
            if (row > 0 && row < iifs[j].C_0.rows &&
                col > 0 && col < iifs[j].C_0.cols)
            {
                if (i_index == seed_patches[i].R)
                {
                    iifs[i_index].C_0.at<int>(row, col) = i;
                }
                else
                {
                    iifs[i_index].C_0.at<int>(row, col) = -1;
                }
            }
            else
            {
                std::cout << "Patch out of visible images." << std::endl;
            }
        }
    }
}

void draw_patches(std::vector<ImgInfos> iifs, std::vector<Patch> patches)
{
    std::fstream patches_stream;
    patches_stream.open(output_path, std::ios::out);
    patches_stream << "ply" << std::endl;
    patches_stream << "format ascii 1.0" << std::endl;
    patches_stream << "element vertex " << patches.size() + iifs.size() - 1 << std::endl;
    patches_stream << "property double x" << std::endl;
    patches_stream << "property double y" << std::endl;
    patches_stream << "property double z" << std::endl;
    patches_stream << "property uchar red" << std::endl;
    patches_stream << "property uchar green" << std::endl;
    patches_stream << "property uchar blue" << std::endl;
    patches_stream << "element face 0" << std::endl;
    patches_stream << "property list uint8 int32 vertex_indices" << std::endl;
    patches_stream << "end_header" << std::endl;

    for (size_t i = 0; i < patches.size(); ++i)
    {
        patches_stream << patches[i].c.x << " " << patches[i].c.y << " " << patches[i].c.z
                       << " " << std::to_string(patches[i].grid.colour.at<cv::Vec3b>(grid_size / 2, grid_size / 2)[2]) << " "
                       << std::to_string(patches[i].grid.colour.at<cv::Vec3b>(grid_size / 2, grid_size / 2)[1]) << " "
                       << std::to_string(patches[i].grid.colour.at<cv::Vec3b>(grid_size / 2, grid_size / 2)[0]) << std::endl;
    }

    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        patches_stream << iifs[i].oc.x << " " << iifs[i].oc.y << " " << iifs[i].oc.z << " 255 0 0" << std::endl;
    }

    patches_stream.close();
}
