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
const cv::Mat_<double> buidingCamera = (cv::Mat_<double>(3, 3) << 1.6687475306166477e+003, 0, 1151.5, 0, 1.6687475306166477e+003, 863.5, 0, 0, 1);
const cv::Mat_<double> detectingDist_Coef = (cv::Mat_<double>(5, 1) << 2.8838084797262502e-002, -3.0375194693353030e-001, 0, 0, 6.6942909508394288e-001);
const unsigned int noSamplingMatches = 2000;
const unsigned int noConsideringMatches = 1000;
const double reprojectingThreshold = 0.5;

/*Structures*/
struct ImgInfos {
    cv::Mat img;
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
};

struct PointInCL {
    cv::Point3d position;
    uchar r, g, b;
    int refer;
    int idx[maxNumOfImages];
    cv::KeyPoint kp;
    cv::Mat des;
};

/*Function headers*/
void modelRegistration(std::vector<PointInCL>& glbCloud, std::string path, int first, int last);
void getMultipleClouds(std::vector<ImgInfos>& iifs, int idx_0, int idx_1, std::vector<PointInCL>& cloud);
void findTheBestRT(std::vector<ImgInfos> iifs, std::vector<std::vector<cv::DMatch>> BFMatches, std::vector<float> BFRatios, std::vector<float> BFSortedRatios, std::vector<cv::DMatch> samplingMatches, int idx_0, int idx_1, cv::Mat_<double>& bestR, cv::Mat_<double>& bestT);
void jointClouds(std::vector<std::vector<PointInCL>> mtpclouds, std::vector<PointInCL>& glbCloud);
void estimateErrorRate(std::vector<cv::Point3d> pts_0, std::vector<cv::Point3d> pts_1, cv::Mat_<double> R, cv::Mat_<double> T, double& errorRate, int& worstIdx, double& quantity);
void drawCloud(std::vector<PointInCL> cloud, std::string path);
void exportModel(std::vector<PointInCL> cloud, std::string path);

/*Main function*/
int main(int argc, char** argv) {
    std::vector<PointInCL> glbCloud;

    std::cout << "3D recontruction." << std::endl;
    modelRegistration(glbCloud, "nestcafe_build/", 0, 50);
    drawCloud(glbCloud, "output/pointCL.ply");
    exportModel(glbCloud, "cloud.data");

    return 0;
}

void modelRegistration(std::vector<PointInCL>& glbCloud, std::string path, int first, int last) {
    std::vector<cv::Mat> images = loadImages(path, first, last);
    std::cout << "Number of images: " << images.size() << std::endl;

    /*Detect key points and compute descriptors*/
    std::vector<ImgInfos> iifs;
    for (unsigned int i = 0; i < images.size(); i++) {
        std::cout << "Number of key points in images " << i << ": ";
        cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
        std::vector<cv::KeyPoint> kp;
        cv::Mat des;
        f2d->detectAndCompute(images[i], cv::Mat(), kp, des);
        ImgInfos iif;
        iif.img = images[i];
        iif.kp = kp;
        iif.des = des;
        iifs.push_back(iif);
        std::cout << kp.size() << std::endl;
    }

    /*Get multiple clouds*/
    std::vector<std::vector<PointInCL>> mtpClouds;
    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        std::vector<PointInCL> cloud;
        getMultipleClouds(iifs, i, i + 1, cloud);
        if (cloud.size() < 50) {
            std::cout << "Can't reconstruct from image " << i << std::endl;
            break;
        }
        std::string outPath = "output/pointCL_";
        outPath.append(std::to_string(i));
        outPath.append(".ply");
        drawCloud(cloud, outPath);
        mtpClouds.push_back(cloud);
    }

    /*Joint clouds to global cloud*/
    jointClouds(mtpClouds, glbCloud);
}

void getMultipleClouds(std::vector<ImgInfos>& iifs, int idx_0, int idx_1, std::vector<PointInCL>& cloud) {
    std::vector<std::vector<cv::DMatch>> BFMatches;
    std::vector<float> BFRatios, BFSortedRatios;
    BFMatchDescriptors(iifs[idx_0].des, iifs[idx_1].des, BFMatches, BFRatios,
                       BFSortedRatios);
    std::vector<cv::DMatch> samplingMatches;
    chooseMatches(BFMatches, BFRatios, BFSortedRatios, noSamplingMatches,
                  samplingMatches);

    /*Find the best R, T*/
    cv::Mat_<double> bestR, bestT;
    findTheBestRT(iifs, BFMatches, BFRatios, BFSortedRatios, samplingMatches,
                  idx_0, idx_1, bestR, bestT);

    cv::Matx34d P_0(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Matx34d P_1 = cv::Matx34d(
        bestR(0, 0), bestR(0, 1), bestR(0, 2), bestT(0),
        bestR(1, 0), bestR(1, 1), bestR(1, 2), bestT(1),
        bestR(2, 0), bestR(2, 1), bestR(2, 2), bestT(2));
    std::vector<PointInCL> tempPCL;
    std::vector<cv::Point3d> p3ds;
    std::vector<cv::Point2d> p2ds;
    for (unsigned int i = 0; i < samplingMatches.size(); i++) {
        PointInCL pICL;
        for (int j = 0; j < maxNumOfImages; j++) {
            pICL.idx[j] = -1;
        }

        /*Mark image index for each point in cloud*/
        pICL.idx[idx_0] = samplingMatches[i].queryIdx;
        pICL.idx[idx_1] = samplingMatches[i].trainIdx;

        /*Estimate 3d position*/
        cv::Point2f point_0(iifs[idx_0].kp[samplingMatches[i].queryIdx].pt);
        cv::Point2f point_1(iifs[idx_1].kp[samplingMatches[i].trainIdx].pt);
        cv::Point3d u_0(point_0.x, point_0.y, 1.0);
        cv::Point3d u_1(point_1.x, point_1.y, 1.0);
        cv::Mat_<double> um_0 = buidingCamera.inv() * cv::Mat_<double>(u_0);
        cv::Mat_<double> um_1 = buidingCamera.inv() * cv::Mat_<double>(u_1);
        u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
        u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
        cv::Mat_<double> point3d = cvIterativeLinearLSTriangulation(u_0, P_0, u_1, P_1);
        pICL.position = cv::Point3d(point3d(0), point3d(1), point3d(2));
        pICL.r = iifs[idx_0].img.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[2];
        pICL.g = iifs[idx_0].img.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[1];
        pICL.b = iifs[idx_0].img.at<cv::Vec3b>((int)point_0.y, (int)point_0.x)[0];
        pICL.des = iifs[idx_0].des.row(samplingMatches[i].queryIdx);
        pICL.kp = iifs[idx_0].kp[samplingMatches[i].queryIdx];
        pICL.refer = idx_0;
        p3ds.push_back(cv::Point3d(point3d(0), point3d(1), point3d(2)));
        p2ds.push_back(iifs[idx_1].kp[samplingMatches[i].trainIdx].pt);

        tempPCL.push_back(pICL);
    }
    cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << bestR(0, 0), bestR(0, 1), bestR(0, 2), bestT(0),
                            bestR(1, 0), bestR(1, 1), bestR(1, 2), bestT(1),
                            bestR(2, 0), bestR(2, 1), bestR(2, 2), bestT(2));

    std::vector<cv::DMatch> correctMatches;
    for (unsigned int i = 0; i < samplingMatches.size(); i++) {
        cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                              p3ds[i].y,
                              p3ds[i].z,
                              1);
        cv::Mat_<double> x = buidingCamera * PRT * X;
        cv::Point2d reprojectedPoint(x.at<double>(0, 0) / x.at<double>(2, 0),
                                     x.at<double>(1, 0) / x.at<double>(2, 0));
        double projectError = norm_2d(reprojectedPoint, p2ds[i]);

        if ((projectError < reprojectingThreshold) &&
            (tempPCL[i].position.z > -10) &&
            (tempPCL[i].position.z < 0)) {
            cloud.push_back(tempPCL[i]);
            correctMatches.push_back(samplingMatches[i]);
        }
    }
    cv::Mat img_samplingMatches;
    cv::drawMatches(iifs[idx_0].img, iifs[idx_0].kp, iifs[idx_1].img, iifs[idx_1].kp,
                    samplingMatches, img_samplingMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string samplingMathcesPath = "log_img/samplingMatches_";
    samplingMathcesPath.append(std::to_string(idx_0));
    samplingMathcesPath.append(".jpg");
    cv::imwrite(samplingMathcesPath, img_samplingMatches);

    cv::Mat img_correctMatches;
    cv::drawMatches(iifs[idx_0].img, iifs[idx_0].kp, iifs[idx_1].img, iifs[idx_1].kp,
                    correctMatches, img_correctMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string correctMathcesPath = "log_img/correctMatches_";
    correctMathcesPath.append(std::to_string(idx_0));
    correctMathcesPath.append(".jpg");
    cv::imwrite(correctMathcesPath, img_correctMatches);
}

void findTheBestRT(std::vector<ImgInfos> iifs, std::vector<std::vector<cv::DMatch>> BFMatches, std::vector<float> BFRatios, std::vector<float> BFSortedRatios, std::vector<cv::DMatch> samplingMatches, int idx_0, int idx_1, cv::Mat_<double>& bestR, cv::Mat_<double>& bestT) {
    cv::Matx34d P_0(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    int max_NoGoodPoints = 0;
    unsigned int bestLoop = 8;
    for (unsigned int loop = 8; loop < noConsideringMatches; loop++) {
        std::vector<cv::DMatch> consideringMatches;
        chooseMatches(BFMatches, BFRatios, BFSortedRatios, loop, consideringMatches);
        if (consideringMatches.size() < 20) {
            continue;
        }

        /*Take corresponding points*/
        std::vector<cv::Point2f> leftPts, rightPts;
        for (unsigned int i = 0; i < consideringMatches.size(); i++) {
            leftPts.push_back(iifs[idx_0].kp[consideringMatches[i].queryIdx].pt);
            rightPts.push_back(iifs[idx_1].kp[consideringMatches[i].trainIdx].pt);
        }

        cv::Mat F = cv::findFundamentalMat(leftPts, rightPts, CV_FM_RANSAC,
                                       reprojectingThreshold, 0.99);
        cv::Mat_<double> E = buidingCamera.t() * F * buidingCamera;

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
        for (unsigned int i = 0; i < samplingMatches.size(); i++) {
            ///Estimate 3d position
            cv::Point2f point_0(iifs[idx_0].kp[samplingMatches[i].queryIdx].pt);
            cv::Point2f point_1(iifs[idx_1].kp[samplingMatches[i].trainIdx].pt);
            cv::Point3d u_0(point_0.x, point_0.y, 1.0);
            cv::Point3d u_1(point_1.x, point_1.y, 1.0);
            cv::Mat_<double> um_0 = buidingCamera.inv() * cv::Mat_<double>(u_0);
            cv::Mat_<double> um_1 = buidingCamera.inv() * cv::Mat_<double>(u_1);
            u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
            u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
            cv::Mat_<double> point3d = cvIterativeLinearLSTriangulation(u_0, P_0, u_1, P_1);

            p3ds.push_back(cv::Point3d(point3d(0), point3d(1), point3d(2)));
            p2ds.push_back(iifs[idx_1].kp[samplingMatches[i].trainIdx].pt);
        }
        cv::Mat_<double> PRT = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), t(0),
                                R(1, 0), R(1, 1), R(1, 2), t(1),
                                R(2, 0), R(2, 1), R(2, 2), t(2));

        int noGoodPoints = 0;
        for (unsigned int i = 0; i < samplingMatches.size(); i++) {
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << p3ds[i].x,
                                  p3ds[i].y,
                                  p3ds[i].z,
                                  1);
            cv::Mat_<double> x = buidingCamera * PRT * X;
            cv::Point2d reprojectedPoint(x.at<double>(0, 0) / x.at<double>(2, 0),
                                         x.at<double>(1, 0) / x.at<double>(2, 0));
            double projectError = norm_2d(reprojectedPoint, p2ds[i]);
            if ((projectError < reprojectingThreshold) &&
                (p3ds[i].z > -10) &&
                (p3ds[i].z < 0)) {
                noGoodPoints++;
            }
        }

        if (!((R(0, 0) < 0 && R(1, 1) < 0 && R(2, 2) < 0) ||
              (R(0, 0) > 0 && R(1, 1) > 0 && R(2, 2) > 0))) {
            noGoodPoints = 0;
        }

        if (noGoodPoints > max_NoGoodPoints) {
            max_NoGoodPoints = noGoodPoints;
            bestR = R;
            bestT = t;
            bestLoop = loop;
        }
    }
    std::cout << "Number of good matches: " << max_NoGoodPoints << ". Best loop: " << bestLoop << std::endl;
}

void jointClouds(std::vector<std::vector<PointInCL>> mtpclouds, std::vector<PointInCL>& glbCloud) {
    glbCloud = mtpclouds[0];
    for (unsigned int loop = 1; loop < mtpclouds.size(); loop++) {
        std::vector<cv::Point3d> pts_0, pts_1;
        for (unsigned int i = 0; i < glbCloud.size(); i++) {
            if (glbCloud[i].idx[loop] != -1) {
                for (unsigned int j = 0; j < mtpclouds[loop].size(); j++) {
                    if (mtpclouds[loop][j].idx[loop] == glbCloud[i].idx[loop]) {
                        pts_0.push_back(glbCloud[i].position);
                        pts_1.push_back(mtpclouds[loop][j].position);
                        break;
                    }
                }
            }
        }

        std::cout << "Number of common points: " << pts_0.size();
        /*Estimation rotation, translation and scaling between two point clouds*/
        cv::Mat_<double> R, T;
        double errorRate;

        for (;;) {
            cvIterative3DAffineEstimation(pts_1, pts_0, R, T);

            /*Estimate error rate of R and T*/
            double quantity;
            int worstIdx;
            estimateErrorRate(pts_0, pts_1, R, T, errorRate, worstIdx, quantity);
            if (quantity > 2) {
                pts_1.erase(pts_1.begin() + worstIdx);
                pts_0.erase(pts_0.begin() + worstIdx);
            } else {
                break;
            }
        }
        std::cout << ". Remain: " << pts_0.size() << ". ErrorRate: " << errorRate << std::endl;

        for (unsigned int i = 0; i < mtpclouds[loop].size(); i++) {
            cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << mtpclouds[loop][i].position.x, mtpclouds[loop][i].position.y, mtpclouds[loop][i].position.z);
            cv::Mat_<double> pt_0_(3, 1);
            pt_0_ = R * pt_1 + T;
            mtpclouds[loop][i].position.x = pt_0_.at<double>(0, 0);
            mtpclouds[loop][i].position.y = pt_0_.at<double>(1, 0);
            mtpclouds[loop][i].position.z = pt_0_.at<double>(2, 0);
            glbCloud.push_back(mtpclouds[loop][i]);
        }
    }
}

void estimateErrorRate(std::vector<cv::Point3d> pts_0, std::vector<cv::Point3d> pts_1, cv::Mat_<double> R, cv::Mat_<double> T, double& errorRate, int& worstIdx, double& quantity) {
    std::vector<cv::Point3d> pts_0_;
    for (unsigned int i = 0; i < pts_1.size(); i++) {
        cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << pts_1[i].x, pts_1[i].y, pts_1[i].z);
        cv::Mat_<double> pt_0_(3, 1);
        pt_0_ = R * pt_1 + T;
        pts_0_.push_back(cv::Point3d(pt_0_.at<double>(0, 0),
                                     pt_0_.at<double>(1, 0),
                                     pt_0_.at<double>(2, 0)));
    }

    double avgError = 0;
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        avgError += norm_2d(pts_0_[i], pts_0[i]);
    }
    avgError /= pts_0.size();
    cv::Point3d centroid(0, 0, 0);
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        centroid += pts_0[i];
    }

    centroid.x /= pts_0.size();
    centroid.y /= pts_0.size();
    centroid.z /= pts_0.size();

    double avgRange = 0;
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        avgRange += norm_2d(pts_0[i], centroid);
    }
    avgRange /= pts_0.size();

    quantity = 0;
    errorRate = avgError / avgRange;
    worstIdx = 0;
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        if (norm_2d(pts_0_[i], pts_0[i]) > quantity) {
            quantity = norm_2d(pts_0_[i], pts_0[i]);
            worstIdx = i;
        }
    }
    quantity = quantity / avgError;
}

void drawCloud(std::vector<PointInCL> cloud, std::string path) {
    std::fstream f;
    const char* path_str = path.c_str();
    f.open(path_str, std::ios::out);
    f << "ply" << std::endl;
    f << "format ascii 1.0" << std::endl;
    f << "element vertex " << cloud.size() << std::endl;
    f << "property double x" << std::endl;
    f << "property double y" << std::endl;
    f << "property double z" << std::endl;
    f << "property uchar red" << std::endl;
    f << "property uchar green" << std::endl;
    f << "property uchar blue" << std::endl;
    f << "element face 0" << std::endl;
    f << "property list uint8 int32 vertex_indices" << std::endl;
    f << "end_header" << std::endl;

    for (unsigned int i = 0; i < cloud.size(); i++) {
        f << cloud[i].position.x << " " << -cloud[i].position.y << " " << cloud[i].position.z << " " << std::to_string(cloud[i].r) << " " << std::to_string(cloud[i].g) << " " << std::to_string(cloud[i].b) << std::endl;
    }
    f.close();
}

void exportModel(std::vector<PointInCL> cloud, std::string path) {
    const char* path_str = path.c_str();
    std::ofstream cloudFile(path_str);
    if (cloudFile.is_open()) {
        cloudFile << cloud.size() << "\n";
        for (unsigned int i = 0; i < cloud.size(); i++) {
            cloudFile << cloud[i].position.x << " " << cloud[i].position.y << " " << cloud[i].position.z << "\n";
            cloudFile << std::to_string(cloud[i].r) << " " << std::to_string(cloud[i].g) << " " << std::to_string(cloud[i].b) << "\n";
            cloudFile << cloud[i].kp.pt.x << " " << cloud[i].kp.pt.y << " " << cloud[i].refer << "\n";
            for (int j = 0; j < cloud[i].des.cols; j++) {
                cloudFile << cloud[i].des.at<float>(0, j) << " ";
            }
            cloudFile << "\n";
        }
        cloudFile.close();
    }
}
