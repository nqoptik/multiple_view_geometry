#include <ctime>
#include <fstream>
#include <iostream>
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
const cv::Mat_<double> camera_matrix = (cv::Mat_<double>(3, 3) << 8.3464596451385648e+002, 0, 575.5, 0, 8.3464596451385648e+002, 431.5, 0, 0, 1);
const cv::Mat_<double> distortion_coefficients_matrix = (cv::Mat_<double>(5, 1) << 2.8838084797262502e-002, -3.0375194693353030e-001, 0, 0, 6.6942909508394288e-001);

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
void import_model(std::vector<PointInCL>& cloud, std::string path);
void recognite_object(std::vector<PointInCL> global_cloud, std::string path, int first, int last);

/*Main function*/
int main()
{
    std::vector<PointInCL> global_cloud;

    std::cout << "Pose estimation." << std::endl;
    import_model(global_cloud, "cloud.data");
    recognite_object(global_cloud, "nestcafe_test/", 0, 150);

    return 0;
}

void import_model(std::vector<PointInCL>& cloud, std::string path)
{
    cloud.clear();
    const char* path_str = path.c_str();
    std::ifstream cloud_stream(path_str);
    if (cloud_stream.is_open())
    {
        int noPoints;
        cloud_stream >> noPoints;
        for (int i = 0; i < noPoints; ++i)
        {
            PointInCL point_in_cloud;
            cloud_stream >> point_in_cloud.position.x;
            cloud_stream >> point_in_cloud.position.y;
            cloud_stream >> point_in_cloud.position.z;
            int temp;
            cloud_stream >> temp;
            point_in_cloud.r = (uchar)temp;
            cloud_stream >> temp;
            point_in_cloud.g = (uchar)temp;
            cloud_stream >> temp;
            point_in_cloud.b = (uchar)temp;
            cloud_stream >> point_in_cloud.kp.pt.x;
            cloud_stream >> point_in_cloud.kp.pt.y;
            cloud_stream >> point_in_cloud.refer;
            point_in_cloud.des = cv::Mat::zeros(1, 128, CV_32F);
            for (int j = 0; j < 128; ++j)
            {
                cloud_stream >> point_in_cloud.des.at<float>(0, j);
            }
            cloud.push_back(point_in_cloud);
        }
        cloud_stream.close();
    }
}

void recognite_object(std::vector<PointInCL> global_cloud, std::string path, int first, int last)
{
    cv::Mat cloud_descriptors;
    std::vector<cv::KeyPoint> cloud_keyponts;
    for (size_t i = 0; i < global_cloud.size(); ++i)
    {
        cloud_descriptors.push_back(global_cloud[i].des);
        cloud_keyponts.push_back(global_cloud[i].kp);
    }

    cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0);
    cv::Mat_<double> X0 = (cv::Mat_<double>(4, 1) << -1.77,
                           -2.86,
                           -5.13,
                           1);
    cv::Mat_<double> x = camera_matrix * PRT * X0;
    cv::Point p0 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X6 = (cv::Mat_<double>(4, 1) << 1.57,
                           3.02,
                           -7.46,
                           1);
    x = camera_matrix * PRT * X6;
    cv::Point p6 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X7 = (cv::Mat_<double>(4, 1) << 1.95,
                           2.62,
                           -4.82,
                           1);
    x = camera_matrix * PRT * X7;
    cv::Point p7 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X1 = (cv::Mat_<double>(4, 1) << -2.27,
                           -2.34,
                           -7.77,
                           1);
    x = camera_matrix * PRT * X1;
    cv::Point p1 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X2 = (cv::Mat_<double>(4, 1) << 1.40,
                           -2.35,
                           -8.31,
                           1);
    x = camera_matrix * PRT * X2;
    cv::Point p2 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X8 = (cv::Mat_<double>(4, 1) << -1.20,
                           -2.05,
                           -5.77,
                           1);
    x = camera_matrix * PRT * X8;
    cv::Point p8 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X3 = (cv::Mat_<double>(4, 1) << 1.79,
                           -2.85,
                           -5.71,
                           1);
    x = camera_matrix * PRT * X3;
    cv::Point p3 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X4 = (cv::Mat_<double>(4, 1) << -1.82,
                           2.50,
                           -4.27,
                           1);
    x = camera_matrix * PRT * X4;
    cv::Point p4 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X5 = (cv::Mat_<double>(4, 1) << -2.21,
                           2.82,
                           -6.82,
                           1);
    x = camera_matrix * PRT * X5;
    cv::Point p5 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat image = cv::imread("nestcafe_build/00000.png");
    int scene_count = 0;
    for (int i = first; i < last; ++i)
    {
        std::cout << "Detecting..." << std::endl;
        std::string image_path = path;
        std::string image_name = int_to_image_name(i, ".png");
        image_path.append(image_name);
        cv::Mat new_image = cv::imread(image_path);
        if (new_image.empty())
            break;

        double t1 = clock();
        cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(2500);
        std::vector<cv::KeyPoint> image_keypoints;
        cv::Mat image_descriptors;
        f2d->detectAndCompute(new_image, cv::Mat(), image_keypoints, image_descriptors);
        double t2 = clock();

        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<float> ratios, sorted_ratios;
        flann_match_descriptors(cloud_descriptors, image_descriptors, matches, ratios, sorted_ratios);

        std::vector<cv::DMatch> correct_matches;
        choose_matches(matches, ratios, sorted_ratios, 100, correct_matches);
        double t3 = clock();
        std::vector<cv::Point3d> p3ds;
        std::vector<cv::Point2d> p2ds;
        for (size_t i = 0; i < correct_matches.size(); ++i)
        {
            p3ds.push_back(global_cloud[correct_matches[i].queryIdx].position);
            p2ds.push_back(image_keypoints[correct_matches[i].trainIdx].pt);
        }

        std::cout << "Number of pairs: " << p3ds.size() << std::endl;

        cv::Mat_<double> bestPRT;
        int maximum_inliners = 0;
        for (int loop = 0; loop < 1000; loop++)
        {
            cv::Mat_<double> tvec, rvec, R_;
            std::vector<cv::Point3d> p3ds_loop;
            std::vector<cv::Point2d> p2ds_loop;
            for (int require = 0; require < 5; require++)
            {
                int idx_ = rand() % p3ds.size();
                p3ds_loop.push_back(p3ds[idx_]);
                p2ds_loop.push_back(p2ds[idx_]);
            }
            cv::solvePnP(p3ds_loop, p2ds_loop, camera_matrix, distortion_coefficients_matrix, rvec, tvec, false);

            cv::Rodrigues(rvec, R_);
            PRT = (cv::Mat_<double>(3, 4) << R_(0, 0), R_(0, 1), R_(0, 2), tvec(0, 0),
                   R_(1, 0), R_(1, 1), R_(1, 2), tvec(1, 0),
                   R_(2, 0), R_(2, 1), R_(2, 2), tvec(2, 0));
            int count_true = 0;
            for (size_t i = 0; i < p3ds.size(); ++i)
            {
                cv::Mat_<double> P3D = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                                        p3ds[i].y,
                                        p3ds[i].z,
                                        1);
                cv::Mat_<double> P2D = camera_matrix * PRT * P3D;
                cv::Point2d p2d = cv::Point2d(P2D.at<double>(0, 0) / P2D.at<double>(2, 0),
                                              P2D.at<double>(1, 0) / P2D.at<double>(2, 0));
                double reprojection_error = get_euclid_distance(p2d, p2ds[i]);

                if (reprojection_error < 2.0)
                {
                    count_true++;
                }
            }

            if (count_true > maximum_inliners)
            {
                maximum_inliners = count_true;
                bestPRT = PRT;
            }
        }

        double t4 = clock();
        std::cout << "Number of key points: " << image_keypoints.size() << std::endl;
        std::cout << "Time: " << t2 - t1 << " " << t3 - t2 << " " << t4 - t3 << " " << t4 - t1 << std::endl;
        std::cout << "maximum_inliners: " << maximum_inliners << std::endl;
        x = camera_matrix * bestPRT * X0;
        p0 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X1;
        p1 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X2;
        p2 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X6;
        p6 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X7;
        p7 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X8;
        p8 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X3;
        p3 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X4;
        p4 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = camera_matrix * bestPRT * X5;
        p5 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

        cv::line(new_image, p0, p1, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(new_image, p1, p2, cv::Scalar(0, 255, 0), 3, 8);
        cv::line(new_image, p2, p3, cv::Scalar(0, 0, 255), 3, 8);
        cv::line(new_image, p3, p0, cv::Scalar(255, 255, 0), 3, 8);
        cv::line(new_image, p4, p5, cv::Scalar(0, 255, 255), 3, 8);
        cv::line(new_image, p5, p6, cv::Scalar(255, 0, 255), 3, 8);

        cv::line(new_image, p6, p7, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(new_image, p7, p4, cv::Scalar(0, 255, 0), 3, 8);

        cv::line(new_image, p0, p4, cv::Scalar(150, 0, 0), 3, 8);
        cv::line(new_image, p1, p5, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(new_image, p2, p6, cv::Scalar(0, 150, 0), 3, 8);
        cv::line(new_image, p3, p7, cv::Scalar(0, 150, 0), 3, 8);

        cv::Mat matches_3d_to_2d_image;
        cv::drawMatches(image, cloud_keyponts, new_image, image_keypoints,
                        correct_matches, matches_3d_to_2d_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        std::string new_path = "img_out/";
        new_path.append(std::to_string(scene_count));
        scene_count++;
        new_path.append(".jpg");
        cv::imwrite(new_path, matches_3d_to_2d_image);
    }
}
