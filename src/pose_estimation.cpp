#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "3d_reconstruction/norm.h"
#include "3d_reconstruction/geometry.h"
#include "3d_reconstruction/loadimages.h"
#include "3d_reconstruction/featuresmatching.h"

/*Constants*/
const int maxNumOfImages = 50;
const cv::Mat_<double> detectingCamera = (cv::Mat_<double>(3, 3) << 8.3464596451385648e+002, 0, 575.5, 0, 8.3464596451385648e+002, 431.5, 0, 0, 1);
const cv::Mat_<double> detectingDist_Coef = (cv::Mat_<double>(5, 1) << 2.8838084797262502e-002, -3.0375194693353030e-001, 0, 0, 6.6942909508394288e-001);

struct PointInCL {
    cv::Point3d position;
    uchar r, g, b;
    int refer;
    int idx[maxNumOfImages];
    cv::KeyPoint kp;
    cv::Mat des;
};

/*Function headers*/
void importModel(std::vector<PointInCL>& cloud, std::string path);
void objectRecognition(std::vector<PointInCL> glbCloud, std::string path, int first, int last);

/*Main function*/
int main(int argc, char** argv) {
    std::vector<PointInCL> glbCloud;

    std::cout << "Pose estimation." << std::endl;
    importModel(glbCloud, "cloud.data");
    objectRecognition(glbCloud, "nestcafe_test/", 0, 150);

    return 0;
}

void importModel(std::vector<PointInCL>& cloud, std::string path) {
    cloud.clear();
    const char* path_str = path.c_str();
    std::ifstream cloudFile(path_str);
    if (cloudFile.is_open()) {
        int noPoints;
        cloudFile >> noPoints;
        for (int i = 0; i < noPoints; i++) {
            PointInCL pICL;
            cloudFile >> pICL.position.x;
            cloudFile >> pICL.position.y;
            cloudFile >> pICL.position.z;
            int temp;
            cloudFile >> temp;
            pICL.r = (uchar)temp;
            cloudFile >> temp;
            pICL.g = (uchar)temp;
            cloudFile >> temp;
            pICL.b = (uchar)temp;
            cloudFile >> pICL.kp.pt.x;
            cloudFile >> pICL.kp.pt.y;
            cloudFile >> pICL.refer;
            pICL.des = cv::Mat::zeros(1, 128, CV_32F);
            for (int j = 0; j < 128; j++) {
                cloudFile >> pICL.des.at<float>(0, j);
            }
            cloud.push_back(pICL);
        }
        cloudFile.close();
    }
}

void objectRecognition(std::vector<PointInCL> glbCloud, std::string path, int first, int last) {
    cv::Mat cloudDes;
    std::vector<cv::KeyPoint> cloudKp;
    for (unsigned int i = 0; i < glbCloud.size(); i++) {
        cloudDes.push_back(glbCloud[i].des);
        cloudKp.push_back(glbCloud[i].kp);
    }

    cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0);
    cv::Mat_<double> X0 = (cv::Mat_<double>(4, 1) << -1.77,
                           -2.86,
                           -5.13,
                           1);
    cv::Mat_<double> x = detectingCamera * PRT * X0;
    cv::Point p0 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X6 = (cv::Mat_<double>(4, 1) << 1.57,
                           3.02,
                           -7.46,
                           1);
    x = detectingCamera * PRT * X6;
    cv::Point p6 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X7 = (cv::Mat_<double>(4, 1) << 1.95,
                           2.62,
                           -4.82,
                           1);
    x = detectingCamera * PRT * X7;
    cv::Point p7 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X1 = (cv::Mat_<double>(4, 1) << -2.27,
                           -2.34,
                           -7.77,
                           1);
    x = detectingCamera * PRT * X1;
    cv::Point p1 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X2 = (cv::Mat_<double>(4, 1) << 1.40,
                           -2.35,
                           -8.31,
                           1);
    x = detectingCamera * PRT * X2;
    cv::Point p2 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X8 = (cv::Mat_<double>(4, 1) << -1.20,
                           -2.05,
                           -5.77,
                           1);
    x = detectingCamera * PRT * X8;
    cv::Point p8 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X3 = (cv::Mat_<double>(4, 1) << 1.79,
                           -2.85,
                           -5.71,
                           1);
    x = detectingCamera * PRT * X3;
    cv::Point p3 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat_<double> X4 = (cv::Mat_<double>(4, 1) << -1.82,
                           2.50,
                           -4.27,
                           1);
    x = detectingCamera * PRT * X4;
    cv::Point p4 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
    cv::Mat_<double> X5 = (cv::Mat_<double>(4, 1) << -2.21,
                           2.82,
                           -6.82,
                           1);
    x = detectingCamera * PRT * X5;
    cv::Point p5 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                             (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

    cv::Mat image = cv::imread("nestcafe_build/00000.png");
    int scene_count = 0;
    for (int i = first; i < last; i++) {
        std::cout << "Detecting..." << std::endl;
        std::string imgPath = path;
        std::string imgName = intToImageName(i, ".png");
        imgPath.append(imgName);
        cv::Mat newFrame = cv::imread(imgPath);
        if (newFrame.empty())
            break;

        double t1 = clock();
        cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(2500);
        std::vector<cv::KeyPoint> kpFrame;
        cv::Mat desFrame;
        f2d->detectAndCompute(newFrame, cv::Mat(), kpFrame, desFrame);
        double t2 = clock();

        std::vector<std::vector<cv::DMatch> > matches;
        std::vector<float> ratios, sortedRatios;
        FLANNMatchDescriptors(cloudDes, desFrame, matches, ratios, sortedRatios);

        std::vector<cv::DMatch> corMatches;
        chooseMatches(matches, ratios, sortedRatios, 100, corMatches);
        double t3 = clock();
        std::vector<cv::Point3d> p3ds;
        std::vector<cv::Point2d> p2ds;
        for (unsigned int i = 0; i < corMatches.size(); i++) {
            p3ds.push_back(glbCloud[corMatches[i].queryIdx].position);
            p2ds.push_back(kpFrame[corMatches[i].trainIdx].pt);
        }

        std::cout << "Number of pairs: " << p3ds.size() << std::endl;

        cv::Mat_<double> bestPRT;
        int max_inliners = 0;
        for (int loop = 0; loop < 1000; loop++) {
            cv::Mat_<double> tvec, rvec, R_;
            std::vector<cv::Point3d> p3ds_loop;
            std::vector<cv::Point2d> p2ds_loop;
            for (int require = 0; require < 5; require++) {
                int idx_ = rand() % p3ds.size();
                p3ds_loop.push_back(p3ds[idx_]);
                p2ds_loop.push_back(p2ds[idx_]);
            }
            cv::solvePnP(p3ds_loop, p2ds_loop, detectingCamera, detectingDist_Coef, rvec, tvec, false);

            cv::Rodrigues(rvec, R_);
            PRT = (cv::Mat_<double>(3, 4) << R_(0, 0), R_(0, 1), R_(0, 2), tvec(0, 0),
                   R_(1, 0), R_(1, 1), R_(1, 2), tvec(1, 0),
                   R_(2, 0), R_(2, 1), R_(2, 2), tvec(2, 0));
            int count_true = 0;
            for (unsigned int i = 0; i < p3ds.size(); i++) {
                cv::Mat_<double> P3D = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                                        p3ds[i].y,
                                        p3ds[i].z,
                                        1);
                cv::Mat_<double> P2D = detectingCamera * PRT * P3D;
                cv::Point2d p2d = cv::Point2d(P2D.at<double>(0, 0) / P2D.at<double>(2, 0),
                                              P2D.at<double>(1, 0) / P2D.at<double>(2, 0));
                double reproError = norm_2d(p2d, p2ds[i]);

                if (reproError < 2.0) {
                    count_true++;
                }
            }

            if (count_true > max_inliners) {
                max_inliners = count_true;
                bestPRT = PRT;
            }
        }

        double t4 = clock();
        std::cout << "Number of key points: " << kpFrame.size() << std::endl;
        std::cout << "Time: " << t2 - t1 << " " << t3 - t2 << " " << t4 - t3 << " " << t4 - t1 << std::endl;
        std::cout << "max_inliners: " << max_inliners << std::endl;
        x = detectingCamera * bestPRT * X0;
        p0 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X1;
        p1 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X2;
        p2 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X6;
        p6 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X7;
        p7 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X8;
        p8 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X3;
        p3 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X4;
        p4 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));
        x = detectingCamera * bestPRT * X5;
        p5 = cv::Point((int)(x.at<double>(0, 0) / x.at<double>(2, 0)),
                       (int)(x.at<double>(1, 0) / x.at<double>(2, 0)));

        cv::line(newFrame, p0, p1, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(newFrame, p1, p2, cv::Scalar(0, 255, 0), 3, 8);
        cv::line(newFrame, p2, p3, cv::Scalar(0, 0, 255), 3, 8);
        cv::line(newFrame, p3, p0, cv::Scalar(255, 255, 0), 3, 8);
        cv::line(newFrame, p4, p5, cv::Scalar(0, 255, 255), 3, 8);
        cv::line(newFrame, p5, p6, cv::Scalar(255, 0, 255), 3, 8);

        cv::line(newFrame, p6, p7, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(newFrame, p7, p4, cv::Scalar(0, 255, 0), 3, 8);

        cv::line(newFrame, p0, p4, cv::Scalar(150, 0, 0), 3, 8);
        cv::line(newFrame, p1, p5, cv::Scalar(255, 0, 0), 3, 8);
        cv::line(newFrame, p2, p6, cv::Scalar(0, 150, 0), 3, 8);
        cv::line(newFrame, p3, p7, cv::Scalar(0, 150, 0), 3, 8);

        cv::Mat img_3d2dMatches;
        cv::drawMatches(image, cloudKp, newFrame, kpFrame,
                        corMatches, img_3d2dMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        std::string newPath = "img_out/";
        newPath.append(std::to_string(scene_count));
        scene_count++;
        newPath.append(".jpg");
        cv::imwrite(newPath, img_3d2dMatches);
    }
}
