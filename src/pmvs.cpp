#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include "3d_reconstruction/norm.h"
#include "3d_reconstruction/geometry.h"

const int maxImgSize = 30;
const cv::Mat_<double> K = (cv::Mat_<double>(3, 3) << 2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1);
const std::string inputPath = "input_images/";
const std::string outputPath = "pointcloud.ply";
const int gridSize = 13;
const int cellSize = 5;
const float goodRatio = 0.55f;
const float minCorrectRatio = 0.25f;
const float step = 0.05f;
const int noStep = 10;
const int min_pairs_findFundamentalMat = 20;
const int noVisible = 3;
const double visibleCos = 0.85;
const double goodPhotometric = 39;
const double goodExpandingPhotometric = 21;
const int noIterations = 100;

struct Plane {
    ///Ax + By + Cz + D = 0
    double A;
    double B;
    double C;
    double D;
};

struct Grid {
    cv::Mat color;
    cv::Point3d position[3];
};

struct ImgInfos {
    cv::Mat img;
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
    cv::Mat_<double> R;
    cv::Mat_<double> T;
    cv::Mat_<double> P;
    cv::Point3d oc;
    cv::Mat C_0;
    cv::Mat C_1;
};

struct Patch {
    cv::Point3d c;
    cv::Point3d n;
    std::vector<int> V;
    int R;
    double xInR;
    double yInR;
    Grid grid;
    double g;
    int idx[maxImgSize];
};

double cvNorm2(cv::Point3d x);
void loadImages(std::vector<cv::Mat>& images);
void PMVS(std::vector<cv::Mat> images, std::vector<Patch>& densePatches);
void detectKeyPointsAndComputeDescriptors(cv::Mat image, ImgInfos& iif);
void getSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches);
void getFirstSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches, int& minPatchIdx);
void matchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios);
void chooseMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, float acceptedRatio, std::vector<cv::DMatch>& acceptedMatches);
void getPatchFromMatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches, int idx_0, int idx_1, std::vector<cv::DMatch> goodMatches, std::vector<cv::DMatch> correctMatches);
void finFundamentalMatrix(std::vector<cv::KeyPoint> kp_0, std::vector<cv::KeyPoint> kp_1, std::vector<cv::DMatch> correctMatches, cv::Mat& F);
void triangulatePatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches, std::vector<cv::DMatch> goodMatches, cv::Matx34d P_0, int idx_0, cv::Matx34d P_1, int idx_1);
void estimateErrorRate(std::vector<cv::Point3d> pts_0, std::vector<cv::Point3d> pts_1, cv::Mat_<double> R, cv::Mat_<double> T, double& errorRate);
void warpPatches(std::vector<Patch> patches_1, std::vector<Patch>& patches_0, cv::Mat_<double> R, cv::Mat_<double> T);
void getMoreSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches, int minPatchIdx);
void calculateImgInfos(std::vector<ImgInfos>& iifs);
void calculateSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches);
void calculateVisibleAndReferenceImages(std::vector<ImgInfos> iifs, Patch& patch);
void calculateGridPosition(std::vector<ImgInfos> iifs, Patch& patch);
void calculateGridColorAndxyInR(std::vector<ImgInfos> iifs, Patch& patch);
void calculatePhotometric(std::vector<ImgInfos> iifs, Patch& patch);
void filterSeedPatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches);
void projectSeedPatchesToImageCells(std::vector<ImgInfos>& iifs, std::vector<Patch> seedPatches);
void markCellsToExpand(std::vector<ImgInfos>& iifs);
void expandPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches);
Plane takePlane(cv::Point3d point, cv::Point3d normal);
cv::Point3d takeIntesection(cv::Point3d A, cv::Point3d B, Plane p);
cv::Point3d newNormal(cv::Point3d A, cv::Point3d oldNormal, cv::Point3d B);
void projectNewPatchesToImageCells(std::vector<ImgInfos>& iifs, std::vector<Patch> seedPatches, int firstNewPatchIdx);

void drawPatches(std::vector<ImgInfos> iifs, std::vector<Patch> patches);

int main() {
    std::vector<cv::Mat> images;

    ///Load images
    loadImages(images);
    std::cout << "Number of images: " << images.size() << std::endl;

    if (images.size() < 4) {
        std::cout << "Not enough images to run PMVS." << std::endl;
        return 0;
    }

    ///Reconstruct model using PMVS
    std::vector<Patch> densePatches;
    PMVS(images, densePatches);

    return 0;
}

double cvNorm2(cv::Point3d x) {
    return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
}

void loadImages(std::vector<cv::Mat>& images) {
    DIR* pDir;
    pDir = opendir(inputPath.c_str());
    if (pDir == NULL) {
        std::cout << "Directory not found." << std::endl;
        return;
    }

    struct dirent* pDirent;
    std::vector<std::string> img_paths;
    while ((pDirent = readdir(pDir)) != NULL) {
        if (strcmp(pDirent->d_name, ".") == 0 || strcmp(pDirent->d_name, "..") == 0) {
            continue;
        }
        std::string imgPath = inputPath;
        imgPath.append(pDirent->d_name);
        img_paths.push_back(imgPath);
    }
    closedir(pDir);

    std::sort(img_paths.begin(), img_paths.end());
    for (size_t i = 0; i < img_paths.size(); ++i) {
        std::cout << img_paths[i] << std::endl;
        cv::Mat image = cv::imread(img_paths[i]);
        if (image.empty()) {
            break;
        }
        images.push_back(image);
    }
}

void PMVS(std::vector<cv::Mat> images, std::vector<Patch>& densePatches) {
    ///Detect key points and compute descriptors
    std::vector<ImgInfos> iifs;

    for (unsigned int i = 0; i < images.size(); i++) {
        std::cout << "Number of key points in images " << i << ": ";
        ImgInfos iif;
        detectKeyPointsAndComputeDescriptors(images[i], iif);
        iifs.push_back(iif);
        std::cout << iif.kp.size() << std::endl;
    }

    ///Get seed patches
    std::vector<Patch> seedPatches;
    getSeedPatches(iifs, seedPatches);
    std::cout << "Calculating images informations..." << std::endl;
    calculateImgInfos(iifs);
    std::cout << "Calculating seed patches..." << std::endl;
    calculateSeedPatches(iifs, seedPatches);
    std::cout << "Filtering seed patches..." << std::endl;
    filterSeedPatches(iifs, seedPatches);
    std::cout << "Number of seed patches: " << seedPatches.size() << std::endl;

    int firstNewPatchIdx = 0;
    for (int i = 0; i < noIterations; i++) {
        std::cout << "expand " << i << "..." << std::endl;
        projectNewPatchesToImageCells(iifs, seedPatches, firstNewPatchIdx);
        std::cout << "Marking cell to expand..." << std::endl;
        markCellsToExpand(iifs);
        firstNewPatchIdx = seedPatches.size();
        std::cout << "Expanding patches..." << std::endl;
        expandPatches(iifs, seedPatches);
        drawPatches(iifs, seedPatches);
    }
}

void detectKeyPointsAndComputeDescriptors(cv::Mat image, ImgInfos& iif) {
    iif.img = image;

    ///Detect key points using SIFT
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_0, keypoints_1;
    f2d->detect(image, keypoints_0);
    for (unsigned int i = 0; i < keypoints_0.size(); i++) {
        float x = keypoints_0[i].pt.x;
        float y = keypoints_0[i].pt.y;
        if ((x > gridSize) && (x < image.cols - gridSize - 1) &&
            (y > gridSize) && (y < image.rows - gridSize - 1)) {
            keypoints_1.push_back(keypoints_0[i]);
        }
    }
    iif.kp = keypoints_1;

    ///Compute descriptors using SIFT
    cv::Mat descriptors;
    f2d->compute(image, keypoints_1, descriptors);
    iif.des = descriptors;
}

void getSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches) {
    int minPatchIdx;
    getFirstSeedPatches(iifs, seedPatches, minPatchIdx);
    getMoreSeedPatches(iifs, seedPatches, minPatchIdx);
}

void getFirstSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches, int& minPatchIdx) {
    ///Match descriptors using brute force
    std::vector<std::vector<cv::DMatch>> matches_0, matches_1;
    std::vector<float> ratios_0, ratios_1;
    matchDescriptors(iifs[0].des, iifs[1].des, matches_0, ratios_0);
    matchDescriptors(iifs[1].des, iifs[2].des, matches_1, ratios_1);

    ///Chose good matches
    std::vector<cv::DMatch> goodMatches_0, goodMatches_1;
    chooseMatches(matches_0, ratios_0, goodRatio, goodMatches_0);
    chooseMatches(matches_1, ratios_1, goodRatio, goodMatches_1);

    ///Find parameters for CV_FM_LMEDS method
    double minErrorRate = 10;
    float bestCorrectRatio_0, bestCorrectRatio_1;
    std::vector<Patch> bestPatches_0, bestPatches_1;
    cv::Mat_<double> bestR, bestT;

    for (int loop_i = 0; loop_i < noStep; loop_i++) {
        std::vector<Patch> patches_0;
        std::vector<cv::DMatch> correctMatches_0;
        float correctRatio_0 = minCorrectRatio + loop_i * step;
        chooseMatches(matches_0, ratios_0, correctRatio_0, correctMatches_0);
        if (correctMatches_0.size() < min_pairs_findFundamentalMat) {
            continue;
        }
        getPatchFromMatches(iifs, patches_0, 0, 1,
                            goodMatches_0, correctMatches_0);

        for (int loop_j = 0; loop_j < noStep; loop_j++) {
            std::vector<Patch> patches_1;
            std::vector<cv::DMatch> correctMatches_1;
            float correctRatio_1 = minCorrectRatio + loop_j * step;
            chooseMatches(matches_1, ratios_1, correctRatio_1, correctMatches_1);
            if (correctMatches_1.size() < min_pairs_findFundamentalMat) {
                continue;
            }
            getPatchFromMatches(iifs, patches_1, 1, 2,
                                goodMatches_1, correctMatches_1);

            ///Take corresponding points
            std::vector<cv::Point3d> pts_0, pts_1;
            for (unsigned int i = 0; i < patches_0.size(); i++) {
                for (unsigned int j = 0; j < patches_1.size(); j++) {
                    if (patches_1[j].idx[1] == patches_0[i].idx[1]) {
                        pts_0.push_back(patches_0[i].c);
                        pts_1.push_back(patches_1[j].c);
                        break;
                    }
                }
            }

            ///Estimation rotation, translation and scaling between two point clouds
            cv::Mat_<double> R, T;
            cvIterative3DAffineEstimation(pts_1, pts_0, R, T);

            ///Estimate error rate of R and T
            double errorRate;
            estimateErrorRate(pts_0, pts_1, R, T, errorRate);

            ///Find min error rate
            if (errorRate < minErrorRate) {
                minErrorRate = errorRate;
                bestCorrectRatio_0 = correctRatio_0;
                bestCorrectRatio_1 = correctRatio_1;
                bestPatches_0 = patches_0;
                bestPatches_1 = patches_1;
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
    std::cout << minErrorRate << " : " << bestCorrectRatio_0 << " : " << bestCorrectRatio_1 << std::endl;
    minPatchIdx = bestPatches_0.size();

    ///Warp patches with the best R and T
    warpPatches(bestPatches_1, bestPatches_0, bestR, bestT);
    seedPatches = bestPatches_0;
}

void matchDescriptors(cv::Mat des_0, cv::Mat des_1, std::vector<std::vector<cv::DMatch>>& matches, std::vector<float>& ratios) {
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(des_0, des_1, matches, 2);

    for (unsigned int i = 0; i < matches.size(); i++) {
        ratios.push_back(matches[i][0].distance / matches[i][1].distance);
    }
}

void chooseMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<float> ratios, float acceptedRatio, std::vector<cv::DMatch>& acceptedMatches) {
    acceptedMatches.clear();
    for (unsigned int i = 0; i < matches.size(); i++) {
        if (ratios[i] <= acceptedRatio) {
            acceptedMatches.push_back(matches[i][0]);
        }
    }
}

void getPatchFromMatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches, int idx_0, int idx_1, std::vector<cv::DMatch> goodMatches, std::vector<cv::DMatch> correctMatches) {
    ///Find fundamental matrix and essential matrix
    cv::Mat F;
    finFundamentalMatrix(iifs[idx_0].kp, iifs[idx_1].kp, correctMatches, F);
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
    triangulatePatches(iifs, patches, goodMatches, P_0, idx_0, P_1, idx_1);
}

void finFundamentalMatrix(std::vector<cv::KeyPoint> kp_0, std::vector<cv::KeyPoint> kp_1, std::vector<cv::DMatch> correctMatches, cv::Mat& F) {
    ///Take corresponding points
    std::vector<cv::Point2f> points_0, points_1;
    for (unsigned int i = 0; i < correctMatches.size(); i++) {
        points_0.push_back(kp_0[correctMatches[i].queryIdx].pt);
        points_1.push_back(kp_1[correctMatches[i].trainIdx].pt);
    }

    ///Find fundamental matrix from corresponding points
    cv::Mat mask;
    F = cv::findFundamentalMat(points_0, points_1, CV_FM_LMEDS, 3., 0.99, mask);
}

void triangulatePatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches, std::vector<cv::DMatch> goodMatches, cv::Matx34d P_0, int idx_0, cv::Matx34d P_1, int idx_1) {
    patches.clear();
    for (unsigned int i = 0; i < goodMatches.size(); i++) {
        Patch patch;

        for (int j = 0; j < maxImgSize; j++) {
            patch.idx[j] = -1;
        }

        ///Mark image index for each point in cloud
        patch.idx[idx_0] = goodMatches[i].queryIdx;
        patch.idx[idx_1] = goodMatches[i].trainIdx;

        cv::Point2d point_0(iifs[idx_0].kp[goodMatches[i].queryIdx].pt.x, iifs[idx_0].kp[goodMatches[i].queryIdx].pt.y);
        cv::Point2d point_1(iifs[idx_1].kp[goodMatches[i].trainIdx].pt.x, iifs[idx_1].kp[goodMatches[i].trainIdx].pt.y);

        ///Estimate 3d position
        cv::Point3d u_0(point_0.x, point_0.y, 1.0);
        cv::Point3d u_1(point_1.x, point_1.y, 1.0);
        cv::Mat_<double> um_0 = K.inv() * cv::Mat_<double>(u_0);
        cv::Mat_<double> um_1 = K.inv() * cv::Mat_<double>(u_1);
        u_0 = cv::Point3d(um_0.at<double>(0, 0), um_0.at<double>(1, 0), um_0.at<double>(2, 0));
        u_1 = cv::Point3d(um_1.at<double>(0, 0), um_1.at<double>(1, 0), um_1.at<double>(2, 0));
        cv::Mat_<double> point3d = cvIterativeLinearLSTriangulation(u_0, P_0, u_1, P_1);

        patch.c = cv::Point3d(point3d(0), point3d(1), point3d(2));
        patch.R = idx_0;
        patches.push_back(patch);
    }
}

void estimateErrorRate(std::vector<cv::Point3d> pts_0, std::vector<cv::Point3d> pts_1, cv::Mat_<double> R, cv::Mat_<double> T, double& errorRate) {
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
        avgError += cvEuclidDistd(pts_0_[i], pts_0[i]);
    }

    cv::Point3d centroid(0, 0, 0);
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        centroid += pts_0[i];
    }

    centroid.x /= pts_0.size();
    centroid.y /= pts_0.size();
    centroid.z /= pts_0.size();

    double avgRange = 0;
    for (unsigned int i = 0; i < pts_0.size(); i++) {
        avgRange += cvEuclidDistd(pts_0[i], centroid);
    }

    errorRate = avgError / avgRange;
}

void warpPatches(std::vector<Patch> patches_1, std::vector<Patch>& patches_0, cv::Mat_<double> R, cv::Mat_<double> T) {
    for (unsigned int i = 0; i < patches_1.size(); i++) {
        cv::Mat_<double> pt_1 = (cv::Mat_<double>(3, 1) << patches_1[i].c.x, patches_1[i].c.y, patches_1[i].c.z);
        cv::Mat_<double> pt_0_(3, 1);
        pt_0_ = R * pt_1 + T;
        patches_1[i].c.x = pt_0_.at<double>(0, 0);
        patches_1[i].c.y = pt_0_.at<double>(1, 0);
        patches_1[i].c.z = pt_0_.at<double>(2, 0);
        patches_0.push_back(patches_1[i]);
    }
}

void getMoreSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches, int minPatchIdx) {
    for (unsigned int idx_1 = 3; idx_1 < iifs.size(); idx_1++) {
        int idx_0 = idx_1 - 1;
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<float> ratios;
        matchDescriptors(iifs[idx_0].des, iifs[idx_1].des, matches, ratios);
        std::vector<cv::DMatch> goodMatches;
        chooseMatches(matches, ratios, goodRatio, goodMatches);
        std::vector<Patch> bestPatches;
        float bestCorrectRatio = minCorrectRatio;
        double minErrorRate = 10;
        cv::Mat_<double> bestR, bestT;

        for (int loop = 0; loop < noStep; loop++) {
            std::vector<Patch> patches;
            std::vector<cv::DMatch> correctMatches;
            float correctRatio = minCorrectRatio + loop * step;
            chooseMatches(matches, ratios, correctRatio, correctMatches);
            if (correctMatches.size() < min_pairs_findFundamentalMat) {
                continue;
            }
            getPatchFromMatches(iifs, patches, idx_0, idx_1,
                                goodMatches, correctMatches);
            std::vector<cv::Point3d> pts_0, pts_1;

            for (unsigned int i = minPatchIdx; i < seedPatches.size(); i++) {
                if (seedPatches[i].idx[idx_0] != -1) {
                    for (unsigned int j = 0; j < patches.size(); j++) {
                        if (patches[j].idx[idx_0] == seedPatches[i].idx[idx_0]) {
                            pts_0.push_back(seedPatches[i].c);
                            pts_1.push_back(patches[j].c);
                            break;
                        }
                    }
                }
            }

            ///Estimation rotation, translation and scaling between two point clouds
            cv::Mat_<double> R, T;
            cvIterative3DAffineEstimation(pts_1, pts_0, R, T);

            ///Estimate error rate of R and T
            double errorRate;
            estimateErrorRate(pts_0, pts_1, R, T, errorRate);

            ///Find min error rate
            if (errorRate < minErrorRate) {
                minErrorRate = errorRate;
                bestCorrectRatio = correctRatio;
                bestPatches = patches;
                bestR = R;
                bestT = T;
            }
        }

        iifs[idx_0].R = bestR;
        iifs[idx_0].T = bestT;

        cv::Mat_<double> oc_ = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::Mat_<double> oc = bestR * oc_ + bestT;
        iifs[idx_0].oc = cv::Point3d(oc.at<double>(0, 0), oc.at<double>(1, 0), oc.at<double>(2, 0));
        std::cout << minErrorRate << "  " << bestCorrectRatio << std::endl;

        ///Warp patches with the best R and T
        minPatchIdx = seedPatches.size();
        warpPatches(bestPatches, seedPatches, bestR, bestT);
    }
}

void calculateImgInfos(std::vector<ImgInfos>& iifs) {
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
    for (unsigned int i = 1; i < iifs.size() - 1; i++) {
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
    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        iifs[i].C_0 = cv::Mat::zeros(floor(iifs[i].img.rows / cellSize) + 1,
                                     floor(iifs[i].img.cols / cellSize) + 1, CV_32SC1);
    }
}

void calculateSeedPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches) {
    ///Calculate normal vector
    for (unsigned int i = 0; i < seedPatches.size(); i++) {
        seedPatches[i].n = iifs[seedPatches[i].R].oc - seedPatches[i].c;
        double d = cvNorm2(seedPatches[i].n);
        seedPatches[i].n.x /= d;
        seedPatches[i].n.y /= d;
        seedPatches[i].n.z /= d;
    }

    ///Calculate visible images
    for (unsigned int i = 0; i < seedPatches.size(); i++) {
        calculateVisibleAndReferenceImages(iifs, seedPatches[i]);
        calculateGridPosition(iifs, seedPatches[i]);
        calculateGridColorAndxyInR(iifs, seedPatches[i]);
        calculatePhotometric(iifs, seedPatches[i]);
    }
}

void calculateVisibleAndReferenceImages(std::vector<ImgInfos> iifs, Patch& patch) {
    double maxCos = 0;
    int RIdx = 0;
    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        cv::Point3d vt;
        vt = iifs[i].oc - patch.c;
        double d = cvNorm2(vt);
        vt.x /= d;
        vt.y /= d;
        vt.z /= d;
        double cos_ = patch.n.x * vt.x +
                      patch.n.y * vt.y + patch.n.z * vt.z;
        if (cos_ > visibleCos) {
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.c.x,
                                  patch.c.y,
                                  patch.c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[i].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            if ((x.at<double>(0, 0) > gridSize) &&
                (x.at<double>(0, 0) < iifs[i].img.cols - gridSize - 1) &&
                (x.at<double>(1, 0) > gridSize) &&
                (x.at<double>(1, 0) < iifs[i].img.rows - gridSize - 1)) {
                patch.V.push_back(i);
            }
        }
        if (cos_ > maxCos) {
            maxCos = cos_;
            RIdx = i;
        }
    }
    patch.R = RIdx;
}

void calculateGridPosition(std::vector<ImgInfos> iifs, Patch& patch) {
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
    x_.at<double>(0, 0) = x.at<double>(0, 0) - (gridSize / 2) * x.at<double>(2, 0);
    x_.at<double>(1, 0) = x.at<double>(1, 0) - (gridSize / 2) * x.at<double>(2, 0);
    x_.at<double>(2, 0) = x.at<double>(2, 0);
    cv::Mat_<double> XYZ1 = KR.inv() * (x_ - KT);
    cv::Point3d p_1 = cv::Point3d(XYZ1(0, 0), XYZ1(1, 0), XYZ1(2, 0));

    ///(x, y) = (0, gridSize - 1)
    x_.at<double>(0, 0) = x.at<double>(0, 0) - (gridSize / 2) * x.at<double>(2, 0);
    x_.at<double>(1, 0) = x.at<double>(1, 0) + (gridSize - 1 - gridSize / 2) * x.at<double>(2, 0);
    x_.at<double>(2, 0) = x.at<double>(2, 0);
    cv::Mat_<double> XYZ2 = KR.inv() * (x_ - KT);
    cv::Point3d p_2 = cv::Point3d(XYZ2(0, 0), XYZ2(1, 0), XYZ2(2, 0));

    ///Rotate position with normal vector
    cv::Point3d nk = cvCross(p_2 - p_1, patch.c - p_1);
    double d = cvNorm2(nk);
    nk.x /= d;
    nk.y /= d;
    nk.z /= d;
    cv::Mat_<double> R = cvRotationBetweenVectors(nk, patch.n);
    XYZ1 = R * XYZ1;
    XYZ2 = R * XYZ2;
    patch.grid.position[0] = patch.c;
    patch.grid.position[1] = cv::Point3d(XYZ1(0, 0), XYZ1(1, 0), XYZ1(2, 0));
    patch.grid.position[2] = cv::Point3d(XYZ2(0, 0), XYZ2(1, 0), XYZ2(2, 0));
}

void calculateGridColorAndxyInR(std::vector<ImgInfos> iifs, Patch& patch) {
    int iIdx = patch.R;
    cv::Point2f srcPoints[3];
    cv::Point2f dstPoints[3];

    ///center
    cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.grid.position[0].x,
                          patch.grid.position[0].y,
                          patch.grid.position[0].z,
                          1);
    cv::Mat_<double> x = K * iifs[iIdx].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);

    ///xInR and yInR
    patch.xInR = x.at<double>(0, 0);
    patch.yInR = x.at<double>(1, 0);

    srcPoints[0] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    dstPoints[0] = cv::Point2f(gridSize / 2, gridSize / 2);

    ///(0, 0)
    X = (cv::Mat_<double>(4, 1) << patch.grid.position[1].x,
         patch.grid.position[1].y,
         patch.grid.position[1].z,
         1);
    x = K * iifs[iIdx].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);
    srcPoints[1] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    dstPoints[1] = cv::Point2f(0, 0);

    ///(x, y) = (0, gridSize - 1)
    X = (cv::Mat_<double>(4, 1) << patch.grid.position[2].x,
         patch.grid.position[2].y,
         patch.grid.position[2].z,
         1);
    x = K * iifs[iIdx].P * X;
    x.at<double>(0, 0) /= x.at<double>(2, 0);
    x.at<double>(1, 0) /= x.at<double>(2, 0);
    srcPoints[2] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
    dstPoints[2] = cv::Point2f(0, gridSize - 1);

    ///Find color using Affine transformation
    cv::Mat Affine = getAffineTransform(srcPoints, dstPoints);
    patch.grid.color = cv::Mat::zeros(gridSize, gridSize, CV_8UC3);
    cv::warpAffine(iifs[iIdx].img, patch.grid.color, Affine, patch.grid.color.size());
}

void calculatePhotometric(std::vector<ImgInfos> iifs, Patch& patch) {
    if (patch.V.size() <= noVisible) {
        patch.g = goodPhotometric + 1;
        return;
    }

    double g = 0;
    for (unsigned int i = 0; i < patch.V.size(); i++) {
        int iIdx = patch.V[i];
        if (iIdx != patch.R) {
            cv::Point2f srcPoints[3];
            cv::Point2f dstPoints[3];

            ///center
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << patch.grid.position[0].x,
                                  patch.grid.position[0].y,
                                  patch.grid.position[0].z,
                                  1);
            cv::Mat_<double> x = K * iifs[iIdx].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            srcPoints[0] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            dstPoints[0] = cv::Point2f(gridSize / 2, gridSize / 2);

            ///(0, 0)
            X = (cv::Mat_<double>(4, 1) << patch.grid.position[1].x,
                 patch.grid.position[1].y,
                 patch.grid.position[1].z,
                 1);
            x = K * iifs[iIdx].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            srcPoints[1] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            dstPoints[1] = cv::Point2f(0, 0);

            ///(gridSize - 1, 0) -> (x, y) = (0, gridSize - 1)
            X = (cv::Mat_<double>(4, 1) << patch.grid.position[2].x,
                 patch.grid.position[2].y,
                 patch.grid.position[2].z,
                 1);
            x = K * iifs[iIdx].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);
            srcPoints[2] = cv::Point2f((float)(x.at<double>(0, 0)), (float)(x.at<double>(1, 0)));
            dstPoints[2] = cv::Point2f(0, gridSize - 1);

            ///Find color using Affine transformation
            cv::Mat Affine = getAffineTransform(srcPoints, dstPoints);
            cv::Mat reprojectColor = cv::Mat::zeros(gridSize, gridSize, CV_8UC3);
            cv::warpAffine(iifs[iIdx].img, reprojectColor, Affine, reprojectColor.size());
            g += cv::norm(reprojectColor - patch.grid.color, cv::NORM_L1);
        }
    }
    patch.g = g / (gridSize * gridSize * (patch.V.size() - 1));
}

void filterSeedPatches(std::vector<ImgInfos> iifs, std::vector<Patch>& patches) {
    ///Eliminate incorrect patches using visibility rule
    for (unsigned int i = 0; i < patches.size(); i++) {
        if ((patches[i].V.size() <= noVisible) || (patches[i].g > goodPhotometric)) {
            patches.erase(patches.begin() + i);
            i--;
        }
    }
}

void projectSeedPatchesToImageCells(std::vector<ImgInfos>& iifs, std::vector<Patch> seedPatches) {
    ///Project seed patches to image cells
    for (unsigned int i = 0; i < seedPatches.size(); i++) {
        for (unsigned j = 0; j < seedPatches[i].V.size(); j++) {
            int iIdx = seedPatches[i].V[j];
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seedPatches[i].c.x,
                                  seedPatches[i].c.y,
                                  seedPatches[i].c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[iIdx].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            int row = floor(x.at<double>(1, 0) / cellSize);
            int col = floor(x.at<double>(0, 0) / cellSize);
            if (row > 0 && row < iifs[j].C_0.rows &&
                col > 0 && col < iifs[j].C_0.cols) {
                if (iIdx == seedPatches[i].R) {
                    iifs[iIdx].C_0.at<int>(row, col) = i;
                } else {
                    iifs[iIdx].C_0.at<int>(row, col) = -1;
                }
            } else {
                std::cout << "Patch out of visible images." << std::endl;
            }
        }
    }
}

void markCellsToExpand(std::vector<ImgInfos>& iifs) {
    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        iifs[i].C_1 = cv::Mat::zeros(floor(iifs[i].img.rows / cellSize) + 1,
                                     floor(iifs[i].img.cols / cellSize) + 1, CV_32SC1);
    }

    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        for (int k = 1; k < iifs[i].C_0.rows - 1; k++) {
            for (int l = 1; l < iifs[i].C_0.cols - 1; l++) {
                if (iifs[i].C_0.at<int>(k, l) == 0) {
                    if (iifs[i].C_0.at<int>(k, l + 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k, l + 1);
                    } else if (iifs[i].C_0.at<int>(k, l - 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k, l - 1);
                    } else if (iifs[i].C_0.at<int>(k + 1, l) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l);
                    } else if (iifs[i].C_0.at<int>(k - 1, l) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l);
                    } else if (iifs[i].C_0.at<int>(k - 1, l - 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l - 1);
                    } else if (iifs[i].C_0.at<int>(k - 1, l + 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k - 1, l + 1);
                    } else if (iifs[i].C_0.at<int>(k + 1, l - 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l - 1);
                    } else if (iifs[i].C_0.at<int>(k + 1, l + 1) > 0) {
                        iifs[i].C_1.at<int>(k, l) = iifs[i].C_0.at<int>(k + 1, l + 1);
                    }
                }
            }
        }
    }
}

void expandPatches(std::vector<ImgInfos>& iifs, std::vector<Patch>& seedPatches) {
    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        for (int k = 1; k < iifs[i].C_1.rows - 1; k++) {
            for (int l = 1; l < iifs[i].C_1.cols - 1; l++) {
                int idx = iifs[i].C_1.at<int>(k, l);
                if (idx > 0) {
                    cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seedPatches[idx].c.x,
                                          seedPatches[idx].c.y,
                                          seedPatches[idx].c.z,
                                          1);
                    cv::Mat_<double> P = K * iifs[seedPatches[idx].R].P;
                    cv::Mat_<double> KR = (cv::Mat_<double>(3, 3) << P(0, 0), P(0, 1), P(0, 2),
                                           P(1, 0), P(1, 1), P(1, 2),
                                           P(2, 0), P(2, 1), P(2, 2));
                    cv::Mat_<double> KT = (cv::Mat_<double>(3, 1) << P(0, 3),
                                           P(1, 3),
                                           P(2, 3));
                    cv::Mat_<double> x = P * X;
                    cv::Mat_<double> x_ = cv::Mat_<double>(3, 1);
                    x_.at<double>(0, 0) = (l * cellSize + cellSize / 2) * x.at<double>(2, 0);
                    x_.at<double>(1, 0) = (k * cellSize + cellSize / 2) * x.at<double>(2, 0);
                    x_.at<double>(2, 0) = x.at<double>(2, 0);
                    cv::Mat_<double> XYZ = KR.inv() * (x_ - KT);
                    cv::Point3d nK = cv::Point3d(XYZ(0, 0), XYZ(1, 0), XYZ(2, 0));

                    ///Optimize the best centroid of patch
                    cv::Point3d O, C, N;
                    O = iifs[seedPatches[idx].R].oc;
                    C = seedPatches[idx].c;
                    Plane plane = takePlane(seedPatches[idx].c, seedPatches[idx].n);
                    N = takeIntesection(O, nK, plane);
                    double dCN = cvEuclidDistd(C, N);
                    double dON = cvEuclidDistd(O, N);
                    cv::Point3d vON = (0.25 * dCN / dON) * (N - O);

                    double bestg = goodExpandingPhotometric + 1;

                    Patch bestNewPatch;
                    for (int loop_p = -2; loop_p < 3; loop_p++) {
                        Patch newPatch;
                        cv::Point3d loopN = N + loop_p * vON;
                        newPatch.c = loopN;
                        cv::Point3d normal = newNormal(C, seedPatches[i].n, loopN);
                        double d = cvNorm2(normal);
                        normal.x /= d;
                        normal.y /= d;
                        normal.z /= d;
                        newPatch.n = normal;
                        calculateVisibleAndReferenceImages(iifs, newPatch);
                        calculateGridPosition(iifs, newPatch);
                        calculateGridColorAndxyInR(iifs, newPatch);
                        calculatePhotometric(iifs, newPatch);

                        if (newPatch.g < bestg) {
                            bestg = newPatch.g;
                            bestNewPatch = newPatch;
                        }
                    }

                    if (bestg <= goodExpandingPhotometric) {
                        seedPatches.push_back(bestNewPatch);
                    }
                }
            }
            std::cout << "Done cell in images " << i << "/" << iifs.size() - 2 << ", row " << k << "/" << iifs[i].C_1.rows - 2 << std::endl;
        }
    }
}

Plane takePlane(cv::Point3d point, cv::Point3d normal) {
    Plane p;
    p.A = normal.x;
    p.B = normal.y;
    p.C = normal.z;
    p.D = -(normal.x * point.x + normal.y * point.y + normal.z * point.z);
    return p;
}

cv::Point3d takeIntesection(cv::Point3d A, cv::Point3d B, Plane p) {
    cv::Point3d AB = B - A;
    double t = -(p.A * A.x + p.B * A.y + p.C * A.z + p.D) /
               (p.A * AB.x + p.B * AB.y + p.C * AB.z);
    cv::Point3d intersection = A + t * AB;
    return intersection;
}

cv::Point3d newNormal(cv::Point3d A, cv::Point3d oldNormal, cv::Point3d B) {
    cv::Point3d AB = B - A;
    return cvCross(AB, cvCross(oldNormal, AB));
}

void projectNewPatchesToImageCells(std::vector<ImgInfos>& iifs, std::vector<Patch> seedPatches, int firstNewPatchIdx) {
    ///Project seed patches to image cells
    for (unsigned int i = firstNewPatchIdx; i < seedPatches.size(); i++) {
        for (unsigned j = 0; j < seedPatches[i].V.size(); j++) {
            int iIdx = seedPatches[i].V[j];
            cv::Mat_<double> X = (cv::Mat_<double>(4, 1) << seedPatches[i].c.x,
                                  seedPatches[i].c.y,
                                  seedPatches[i].c.z,
                                  1);
            cv::Mat_<double> x = K * iifs[iIdx].P * X;
            x.at<double>(0, 0) /= x.at<double>(2, 0);
            x.at<double>(1, 0) /= x.at<double>(2, 0);

            int row = floor(x.at<double>(1, 0) / cellSize);
            int col = floor(x.at<double>(0, 0) / cellSize);
            if (row > 0 && row < iifs[j].C_0.rows &&
                col > 0 && col < iifs[j].C_0.cols) {
                if (iIdx == seedPatches[i].R) {
                    iifs[iIdx].C_0.at<int>(row, col) = i;
                } else {
                    iifs[iIdx].C_0.at<int>(row, col) = -1;
                }
            } else {
                std::cout << "Patch out of visible images." << std::endl;
            }
        }
    }
}

void drawPatches(std::vector<ImgInfos> iifs, std::vector<Patch> patches) {
    std::fstream f;
    f.open(outputPath, std::ios::out);
    f << "ply" << std::endl;
    f << "format ascii 1.0" << std::endl;
    f << "element vertex " << patches.size() + iifs.size() - 1 << std::endl;
    f << "property double x" << std::endl;
    f << "property double y" << std::endl;
    f << "property double z" << std::endl;
    f << "property uchar red" << std::endl;
    f << "property uchar green" << std::endl;
    f << "property uchar blue" << std::endl;
    f << "element face 0" << std::endl;
    f << "property list uint8 int32 vertex_indices" << std::endl;
    f << "end_header" << std::endl;

    for (unsigned int i = 0; i < patches.size(); i++) {
        f << patches[i].c.x << " " << patches[i].c.y << " " << patches[i].c.z << " " << std::to_string(patches[i].grid.color.at<cv::Vec3b>(gridSize / 2, gridSize / 2)[2]) << " " << std::to_string(patches[i].grid.color.at<cv::Vec3b>(gridSize / 2, gridSize / 2)[1]) << " " << std::to_string(patches[i].grid.color.at<cv::Vec3b>(gridSize / 2, gridSize / 2)[0]) << std::endl;
    }

    for (unsigned int i = 0; i < iifs.size() - 1; i++) {
        f << iifs[i].oc.x << " " << iifs[i].oc.y << " " << iifs[i].oc.z << " 255 0 0" << std::endl;
    }

    f.close();
}
