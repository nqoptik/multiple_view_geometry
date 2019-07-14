#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "multiple_view_geometry/featuresmatching.hpp"
#include "multiple_view_geometry/geometry.hpp"
#include "multiple_view_geometry/loadimages.hpp"
#include "multiple_view_geometry/norm.hpp"

/*Constants*/
const int maximum_number_of_images = 50;
const cv::Mat_<double> camera_matrix = (cv::Mat_<double>(3, 3) << 1.6687475306166477e+003, 0, 1151.5, 0, 1.6687475306166477e+003, 863.5, 0, 0, 1);
const cv::Mat_<double> distortion_coefficients_matrix = (cv::Mat_<double>(5, 1) << 2.8838084797262502e-002, -3.0375194693353030e-001, 0, 0, 6.6942909508394288e-001);
const size_t number_of_sampling_matches = 2000;
const size_t number_of_considering_matches = 1000;
const double reprojecting_threshold = 0.5;

/*Structures*/
struct ImgInfos
{
    cv::Mat image;
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
};

struct PointInCL
{
    cv::Point3d position;
    uchar r, g, b;
    int refer;
    int idx[maximum_number_of_images];
    cv::KeyPoint kp;
    cv::Mat des;
};

/*Function headers*/
void register_model(std::vector<PointInCL>& global_cloud, std::string path, int first, int last);
void get_multiple_clouds(std::vector<ImgInfos>& iifs, int index_0, int index_1, std::vector<PointInCL>& cloud);
void find_rotation_translation_matrices(std::vector<ImgInfos> iifs,
                                        std::vector<std::vector<cv::DMatch>> brute_force_matches,
                                        std::vector<float> brute_force_ratios,
                                        std::vector<float> sorted_brute_force_ratios,
                                        std::vector<cv::DMatch> sampling_matches,
                                        int index_0,
                                        int index_1,
                                        cv::Mat_<double>& bestR,
                                        cv::Mat_<double>& bestT);
void merge_clouds(std::vector<std::vector<PointInCL>> multiple_clouds, std::vector<PointInCL>& global_cloud);
void estimate_error_rate(std::vector<cv::Point3d> points_0,
                         std::vector<cv::Point3d> points_1,
                         cv::Mat_<double> R,
                         cv::Mat_<double> T,
                         double& error_rate,
                         int& worst_index,
                         double& quantity);
void draw_cloud(std::vector<PointInCL> cloud, std::string path);
void export_model(std::vector<PointInCL> cloud, std::string path);

/*Main function*/
int main()
{
    std::vector<PointInCL> global_cloud;

    std::cout << "3D recontruction." << std::endl;
    register_model(global_cloud, "nestcafe_build/", 0, 50);
    draw_cloud(global_cloud, "output/pointCL.ply");
    export_model(global_cloud, "cloud.data");

    return 0;
}

void register_model(std::vector<PointInCL>& global_cloud, std::string path, int first, int last)
{
    std::vector<cv::Mat> images = load_images(path, first, last);
    std::cout << "Number of images: " << images.size() << std::endl;

    /*Detect key points and compute descriptors*/
    std::vector<ImgInfos> iifs;
    for (size_t i = 0; i < images.size(); ++i)
    {
        std::cout << "Number of key points in images " << i << ": ";
        cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
        std::vector<cv::KeyPoint> kp;
        cv::Mat des;
        f2d->detectAndCompute(images[i], cv::Mat(), kp, des);
        ImgInfos iif;
        iif.image = images[i];
        iif.kp = kp;
        iif.des = des;
        iifs.push_back(iif);
        std::cout << kp.size() << std::endl;
    }

    /*Get multiple clouds*/
    std::vector<std::vector<PointInCL>> multiple_clouds;
    for (size_t i = 0; i < iifs.size() - 1; ++i)
    {
        std::vector<PointInCL> cloud;
        get_multiple_clouds(iifs, i, i + 1, cloud);
        if (cloud.size() < 50)
        {
            std::cout << "Can't reconstruct from image " << i << std::endl;
            break;
        }
        std::string outPath = "output/pointCL_";
        outPath.append(std::to_string(i));
        outPath.append(".ply");
        draw_cloud(cloud, outPath);
        multiple_clouds.push_back(cloud);
    }

    /*Joint clouds to global cloud*/
    merge_clouds(multiple_clouds, global_cloud);
}

void get_multiple_clouds(std::vector<ImgInfos>& iifs, int index_0, int index_1, std::vector<PointInCL>& cloud)
{
    std::vector<std::vector<cv::DMatch>> brute_force_matches;
    std::vector<float> brute_force_ratios, sorted_brute_force_ratios;
    brute_force_match_descriptors(iifs[index_0].des, iifs[index_1].des, brute_force_matches, brute_force_ratios,
                                  sorted_brute_force_ratios);
    std::vector<cv::DMatch> sampling_matches;
    choose_matches(brute_force_matches, brute_force_ratios, sorted_brute_force_ratios, number_of_sampling_matches,
                   sampling_matches);

    /*Find the best R, T*/
    cv::Mat_<double> bestR, bestT;
    find_rotation_translation_matrices(iifs, brute_force_matches, brute_force_ratios, sorted_brute_force_ratios, sampling_matches,
                                       index_0, index_1, bestR, bestT);

    cv::Matx34d P_0(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Matx34d P_1 = cv::Matx34d(
        bestR(0, 0), bestR(0, 1), bestR(0, 2), bestT(0),
        bestR(1, 0), bestR(1, 1), bestR(1, 2), bestT(1),
        bestR(2, 0), bestR(2, 1), bestR(2, 2), bestT(2));
    std::vector<PointInCL> temporary_cloud;
    std::vector<cv::Point3d> p3ds;
    std::vector<cv::Point2d> p2ds;
    for (size_t i = 0; i < sampling_matches.size(); ++i)
    {
        PointInCL point_in_cloud;
        for (int j = 0; j < maximum_number_of_images; ++j)
        {
            point_in_cloud.idx[j] = -1;
        }

        /*Mark image index for each point in cloud*/
        point_in_cloud.idx[index_0] = sampling_matches[i].queryIdx;
        point_in_cloud.idx[index_1] = sampling_matches[i].trainIdx;

        /*Estimate 3d position*/
        cv::Point2f point_0(iifs[index_0].kp[sampling_matches[i].queryIdx].pt);
        cv::Point2f point_1(iifs[index_1].kp[sampling_matches[i].trainIdx].pt);
        cv::Point3d u_0(point_0.x, point_0.y, 1.0);
        cv::Point3d u_1(point_1.x, point_1.y, 1.0);
        cv::Mat_<double> um_0 = camera_matrix.inv() * cv::Mat_<double>(u_0);
        cv::Mat_<double> um_1 = camera_matrix.inv() * cv::Mat_<double>(u_1);
        u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
        u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
        cv::Mat_<double> point3d = iterative_linear_ls_triangulation(u_0, P_0, u_1, P_1);
        point_in_cloud.position = cv::Point3d(point3d(0), point3d(1), point3d(2));
        point_in_cloud.r = iifs[index_0].image.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[2];
        point_in_cloud.g = iifs[index_0].image.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[1];
        point_in_cloud.b = iifs[index_0].image.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[0];
        point_in_cloud.des = iifs[index_0].des.row(sampling_matches[i].queryIdx);
        point_in_cloud.kp = iifs[index_0].kp[sampling_matches[i].queryIdx];
        point_in_cloud.refer = index_0;
        p3ds.push_back(cv::Point3d(point3d(0), point3d(1), point3d(2)));
        p2ds.push_back(iifs[index_1].kp[sampling_matches[i].trainIdx].pt);

        temporary_cloud.push_back(point_in_cloud);
    }
    cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << bestR(0, 0), bestR(0, 1), bestR(0, 2), bestT(0),
                            bestR(1, 0), bestR(1, 1), bestR(1, 2), bestT(1),
                            bestR(2, 0), bestR(2, 1), bestR(2, 2), bestT(2));

    std::vector<cv::DMatch> correct_matches;
    for (size_t i = 0; i < sampling_matches.size(); ++i)
    {
        cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                              p3ds[i].y,
                              p3ds[i].z,
                              1);
        cv::Mat_<double> x = camera_matrix * PRT * X;
        cv::Point2d reprojected_point(x.at<double>(0, 0) / x.at<double>(2, 0),
                                      x.at<double>(1, 0) / x.at<double>(2, 0));
        double project_error = get_euclid_distance(reprojected_point, p2ds[i]);

        if ((project_error < reprojecting_threshold) &&
            (temporary_cloud[i].position.z > -10) &&
            (temporary_cloud[i].position.z < 0))
        {
            cloud.push_back(temporary_cloud[i]);
            correct_matches.push_back(sampling_matches[i]);
        }
    }
    cv::Mat sampling_matches_image;
    cv::drawMatches(iifs[index_0].image, iifs[index_0].kp, iifs[index_1].image, iifs[index_1].kp,
                    sampling_matches, sampling_matches_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string sampling_matches_path = "log_img/samplingMatches_";
    sampling_matches_path.append(std::to_string(index_0));
    sampling_matches_path.append(".jpg");
    cv::imwrite(sampling_matches_path, sampling_matches_image);

    cv::Mat correct_matches_image;
    cv::drawMatches(iifs[index_0].image, iifs[index_0].kp, iifs[index_1].image, iifs[index_1].kp,
                    correct_matches, correct_matches_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string correct_mathces_path = "log_img/correctMatches_";
    correct_mathces_path.append(std::to_string(index_0));
    correct_mathces_path.append(".jpg");
    cv::imwrite(correct_mathces_path, correct_matches_image);
}

void find_rotation_translation_matrices(std::vector<ImgInfos> iifs,
                                        std::vector<std::vector<cv::DMatch>> brute_force_matches,
                                        std::vector<float> brute_force_ratios,
                                        std::vector<float> sorted_brute_force_ratios,
                                        std::vector<cv::DMatch> sampling_matches,
                                        int index_0,
                                        int index_1,
                                        cv::Mat_<double>& bestR,
                                        cv::Mat_<double>& bestT)
{
    cv::Matx34d P_0(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    int maximum_number_of_good_points = 0;
    size_t best_loop_index = 8;

#pragma omp parallel for
    for (size_t loop = 8; loop < number_of_considering_matches; ++loop)
    {
        std::vector<cv::DMatch> considering_matches;
        choose_matches(brute_force_matches, brute_force_ratios, sorted_brute_force_ratios, loop, considering_matches);
        if (considering_matches.size() < 20)
        {
            continue;
        }

        /*Take corresponding points*/
        std::vector<cv::Point2f> left_points, right_points;
        for (size_t i = 0; i < considering_matches.size(); ++i)
        {
            left_points.push_back(iifs[index_0].kp[considering_matches[i].queryIdx].pt);
            right_points.push_back(iifs[index_1].kp[considering_matches[i].trainIdx].pt);
        }

        cv::Mat F = cv::findFundamentalMat(left_points, right_points, CV_FM_RANSAC,
                                           reprojecting_threshold, 0.99);
        cv::Mat_<double> E = camera_matrix.t() * F * camera_matrix;

        cv::SVD svd(E);
        cv::Matx33d W(
            0, -1, 0,
            1, 0, 0,
            0, 0, 1);
        cv::Mat_<double> R = svd.u * cv::Mat(W) * svd.vt;
        cv::Mat_<double> t = svd.u.col(2);

        cv::Matx34d P_1 = cv::Matx34d(
            R(0, 0), R(0, 1), R(0, 2), t(0),
            R(1, 0), R(1, 1), R(1, 2), t(1),
            R(2, 0), R(2, 1), R(2, 2), t(2));

        /*Measure error*/
        std::vector<cv::Point3d> p3ds;
        std::vector<cv::Point2d> p2ds;
        for (size_t i = 0; i < sampling_matches.size(); ++i)
        {
            ///Estimate 3d position
            cv::Point2f point_0(iifs[index_0].kp[sampling_matches[i].queryIdx].pt);
            cv::Point2f point_1(iifs[index_1].kp[sampling_matches[i].trainIdx].pt);
            cv::Point3d u_0(point_0.x, point_0.y, 1.0);
            cv::Point3d u_1(point_1.x, point_1.y, 1.0);
            cv::Mat_<double> um_0 = camera_matrix.inv() * cv::Mat_<double>(u_0);
            cv::Mat_<double> um_1 = camera_matrix.inv() * cv::Mat_<double>(u_1);
            u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
            u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
            cv::Mat_<double> point3d = iterative_linear_ls_triangulation(u_0, P_0, u_1, P_1);

            p3ds.push_back(cv::Point3d(point3d(0), point3d(1), point3d(2)));
            p2ds.push_back(iifs[index_1].kp[sampling_matches[i].trainIdx].pt);
        }
        cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), t(0),
                                R(1, 0), R(1, 1), R(1, 2), t(1),
                                R(2, 0), R(2, 1), R(2, 2), t(2));

        int number_of_good_points = 0;
        for (size_t i = 0; i < sampling_matches.size(); ++i)
        {
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                                  p3ds[i].y,
                                  p3ds[i].z,
                                  1);
            cv::Mat_<double> x = camera_matrix * PRT * X;
            cv::Point2d reprojected_point(x.at<double>(0, 0) / x.at<double>(2, 0),
                                          x.at<double>(1, 0) / x.at<double>(2, 0));
            double project_error = get_euclid_distance(reprojected_point, p2ds[i]);
            if ((project_error < reprojecting_threshold) &&
                (p3ds[i].z > -10) &&
                (p3ds[i].z < 0))
            {
                number_of_good_points++;
            }
        }

        if (!((R(0, 0) < 0 && R(1, 1) < 0 && R(2, 2) < 0) ||
              (R(0, 0) > 0 && R(1, 1) > 0 && R(2, 2) > 0)))
        {
            number_of_good_points = 0;
        }

        if (number_of_good_points > maximum_number_of_good_points)
        {
            maximum_number_of_good_points = number_of_good_points;
            bestR = R;
            bestT = t;
            best_loop_index = loop;
        }
    }
    std::cout << "Number of good matches: " << maximum_number_of_good_points << ". Best loop: " << best_loop_index << std::endl;
}

void merge_clouds(std::vector<std::vector<PointInCL>> multiple_clouds, std::vector<PointInCL>& global_cloud)
{
    global_cloud = multiple_clouds[0];
    for (size_t loop = 1; loop < multiple_clouds.size(); ++loop)
    {
        std::vector<cv::Point3d> points_0, points_1;
        for (size_t i = 0; i < global_cloud.size(); ++i)
        {
            if (global_cloud[i].idx[loop] != -1)
            {
                for (size_t j = 0; j < multiple_clouds[loop].size(); ++j)
                {
                    if (multiple_clouds[loop][j].idx[loop] == global_cloud[i].idx[loop])
                    {
                        points_0.push_back(global_cloud[i].position);
                        points_1.push_back(multiple_clouds[loop][j].position);
                        break;
                    }
                }
            }
        }

        std::cout << "Number of common points: " << points_0.size();
        /*Estimation rotation, translation and scaling between two point clouds*/
        cv::Mat_<double> R, T;
        double error_rate;

        for (;;)
        {
            estimate_iterative_3d_affine(points_1, points_0, R, T);

            /*Estimate error rate of R and T*/
            double quantity;
            int worst_index;
            estimate_error_rate(points_0, points_1, R, T, error_rate, worst_index, quantity);
            if (quantity > 2)
            {
                points_1.erase(points_1.begin() + worst_index);
                points_0.erase(points_0.begin() + worst_index);
            }
            else
            {
                break;
            }
        }
        std::cout << ". Remain: " << points_0.size() << ". ErrorRate: " << error_rate << std::endl;

        for (size_t i = 0; i < multiple_clouds[loop].size(); ++i)
        {
            cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << multiple_clouds[loop][i].position.x, multiple_clouds[loop][i].position.y, multiple_clouds[loop][i].position.z);
            cv::Mat_<double> pt_0_(3, 1);
            pt_0_ = R * pt_1 + T;
            multiple_clouds[loop][i].position.x = pt_0_.at<double>(0, 0);
            multiple_clouds[loop][i].position.y = pt_0_.at<double>(1, 0);
            multiple_clouds[loop][i].position.z = pt_0_.at<double>(2, 0);
            global_cloud.push_back(multiple_clouds[loop][i]);
        }
    }
}

void estimate_error_rate(std::vector<cv::Point3d> points_0,
                         std::vector<cv::Point3d> points_1,
                         cv::Mat_<double> R,
                         cv::Mat_<double> T,
                         double& error_rate,
                         int& worst_index,
                         double& quantity)
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
    averge_error /= points_0.size();
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
    averge_range /= points_0.size();

    quantity = 0;
    error_rate = averge_error / averge_range;
    worst_index = 0;
    for (size_t i = 0; i < points_0.size(); ++i)
    {
        if (get_euclid_distance(pts_0_[i], points_0[i]) > quantity)
        {
            quantity = get_euclid_distance(pts_0_[i], points_0[i]);
            worst_index = i;
        }
    }
    quantity = quantity / averge_error;
}

void draw_cloud(std::vector<PointInCL> cloud, std::string path)
{
    std::fstream cloud_stream;
    const char* path_str = path.c_str();
    cloud_stream.open(path_str, std::ios::out);
    cloud_stream << "ply" << std::endl;
    cloud_stream << "format ascii 1.0" << std::endl;
    cloud_stream << "element vertex " << cloud.size() << std::endl;
    cloud_stream << "property double x" << std::endl;
    cloud_stream << "property double y" << std::endl;
    cloud_stream << "property double z" << std::endl;
    cloud_stream << "property uchar red" << std::endl;
    cloud_stream << "property uchar green" << std::endl;
    cloud_stream << "property uchar blue" << std::endl;
    cloud_stream << "element face 0" << std::endl;
    cloud_stream << "property list uint8 int32 vertex_indices" << std::endl;
    cloud_stream << "end_header" << std::endl;

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        cloud_stream << cloud[i].position.x << " " << -cloud[i].position.y << " " << cloud[i].position.z << " "
                     << std::to_string(cloud[i].r) << " " << std::to_string(cloud[i].g) << " " << std::to_string(cloud[i].b) << std::endl;
    }
    cloud_stream.close();
}

void export_model(std::vector<PointInCL> cloud, std::string path)
{
    const char* path_str = path.c_str();
    std::ofstream model_stream(path_str);
    if (model_stream.is_open())
    {
        model_stream << cloud.size() << "\n";
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            model_stream << cloud[i].position.x << " " << cloud[i].position.y << " " << cloud[i].position.z << "\n";
            model_stream << std::to_string(cloud[i].r) << " " << std::to_string(cloud[i].g) << " " << std::to_string(cloud[i].b) << "\n";
            model_stream << cloud[i].kp.pt.x << " " << cloud[i].kp.pt.y << " " << cloud[i].refer << "\n";
            for (int j = 0; j < cloud[i].des.cols; ++j)
            {
                model_stream << cloud[i].des.at<float>(0, j) << " ";
            }
            model_stream << "\n";
        }
        model_stream.close();
    }
}
