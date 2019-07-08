#include "3d_reconstruction/geometry.h"

cv::Point3d cvCross(cv::Point3d a, cv::Point3d b) {
    return cv::Point3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

cv::Mat_<double> cvRotationBetweenVectors(cv::Point3d a, cv::Point3d b) {
    cv::Point3d v = cvCross(a, b);
    double s = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    cv::Mat_<double> I = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                          0, 1, 0,
                          0, 0, 1);
    if (s == 0)
        return I;
    double c = a.x * b.x + a.y * b.y + a.z * b.z;
    cv::Mat_<double> vec = (cv::Mat_<double>(3, 3) << 0, -v.z, v.y,
                            v.z, 0, -v.x,
                            -v.y, v.x, 0);

    return I + vec + vec * vec * (1 - c) / s / s;
}

cv::Mat_<double> cvLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1) {
    /*Build A matrix*/
    cv::Matx43d A(
        u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
        u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
        u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
        u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2));
    /*Build B vector*/
    cv::Matx41d B(
        -(u.x * P(2, 3) - P(0, 3)),
        -(u.y * P(2, 3) - P(1, 3)),
        -(u1.x * P1(2, 3) - P1(0, 3)),
        -(u1.y * P1(2, 3) - P1(1, 3)));
    /*Solve for X*/
    cv::Mat_<double> X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

cv::Mat_<double> cvIterativeLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1) {
    double wi = 1, wi1 = 1;
    cv::Mat_<double> X(4, 1);

    cv::Mat_<double> X_ = cvLinearLSTriangulation(u, P, u1, P1);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;

    for (int i = 0; i < 10; i++) {
        /*Recalculate weights*/
        double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2) * X)(0);
        double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2) * X)(0);

        /*Breaking point*/
        double EPSILON = 1e-4;
        if (fabs(wi - p2x) <= EPSILON && fabs(wi1 - p2x1) <= EPSILON)
            break;

        wi = p2x;
        wi1 = p2x1;

        /*Reweight equations and solve*/
        cv::Matx43d A(
            (u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
            (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
            (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1, (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
            (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1, (u1.y * P1(2, 2) - P1(1, 2)) / wi1);
        cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
                              -(u.y * P(2, 3) - P(1, 3)) / wi,
                              -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
                              -(u1.y * P1(2, 3) - P1(1, 3)) / wi1);

        cv::solve(A, B, X_, cv::DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
    }
    return X;
}

void cv3DAffineEstimation(std::vector<cv::Point3d> src, std::vector<cv::Point3d> dst, cv::Mat_<double>& R, cv::Mat_<double>& T) {
    /*Find the scale by finding the ratio of some distances*/
    int commonSize = src.size();
    double dist_src = 0, dist_dst = 0, scale;
    for (int i = 0; i < commonSize - 1; i++) {
        dist_src += norm_2d(src[i], src[i + 1]);
        dist_dst += norm_2d(dst[i], dst[i + 1]);
    }
    scale = dist_dst / dist_src;

    /*Bring point sets to the same scale*/
    for (int i = 0; i < commonSize; i++) {
        src[i] *= scale;
    }

    /*Find the centroids*/
    cv::Point3d centroid_src(0, 0, 0), centroid_dst(0, 0, 0);
    for (int i = 0; i < commonSize; i++) {
        centroid_src += src[i];
        centroid_dst += dst[i];
    }

    centroid_src.x /= commonSize;
    centroid_src.y /= commonSize;
    centroid_src.z /= commonSize;

    centroid_dst.x /= commonSize;
    centroid_dst.y /= commonSize;
    centroid_dst.z /= commonSize;

    /*Shift to origin*/
    for (int i = 0; i < commonSize; i++) {
        src[i] -= centroid_src;
        dst[i] -= centroid_dst;
    }

    /*Find covariance matrix*/
    cv::Mat_<double> cor(3, 3);
    cor.setTo(cv::Scalar::all(0));
    for (int i = 0; i < commonSize; i++) {
        cv::Mat_<double> cor_i = (cv::Mat_<double>(3, 3) << src[i].x * dst[i].x, src[i].x * dst[i].y, src[i].x * dst[i].z,
                                  src[i].y * dst[i].x, src[i].y * dst[i].y, src[i].y * dst[i].z,
                                  src[i].z * dst[i].x, src[i].z * dst[i].y, src[i].z * dst[i].z);
        cor += cor_i;
    }

    /*Find rotation using SVD*/
    cv::SVD svd(cor);
    R = svd.vt.t() * svd.u.t();
    R *= scale;

    /*Find translation*/
    T = (cv::Mat_<double>(3, 1) << scale * (centroid_dst.x - centroid_src.x),
         scale * (centroid_dst.y - centroid_src.y),
         scale * (centroid_dst.z - centroid_src.z));
}

void cvIterative3DAffineEstimation(std::vector<cv::Point3d> src, std::vector<cv::Point3d> dst, cv::Mat_<double>& R, cv::Mat_<double>& T) {
    cv3DAffineEstimation(src, dst, R, T);

    for (int i = 0; i < 10; i++) {
        std::vector<cv::Point3d> src_;
        for (unsigned int j = 0; j < src.size(); j++) {
            cv::Mat_<double> src_i = (cv::Mat_<double>(3, 1) << src[j].x, src[j].y, src[j].z);
            cv::Mat_<double> dst_i = R * src_i + T;
            src_.push_back(cv::Point3d(dst_i.at<double>(0, 0),
                                       dst_i.at<double>(1, 0),
                                       dst_i.at<double>(2, 0)));
        }

        cv::Mat_<double> R_, T_;
        cv3DAffineEstimation(src_, dst, R_, T_);
        R = R_ * R;
        T = R_ * T + T_;

        /*Breaking point*/
        double EPSILON = 1e-4;
        if (fabs(R_.at<double>(0, 0) - 1) <= EPSILON &&
            fabs(R_.at<double>(1, 1) - 1) <= EPSILON &&
            fabs(R_.at<double>(2, 2) - 1) <= EPSILON &&
            fabs(R_.at<double>(0, 1)) <= EPSILON &&
            fabs(R_.at<double>(0, 2)) <= EPSILON &&
            fabs(R_.at<double>(1, 0)) <= EPSILON &&
            fabs(R_.at<double>(1, 2)) <= EPSILON &&
            fabs(R_.at<double>(2, 0)) <= EPSILON &&
            fabs(R_.at<double>(2, 1)) <= EPSILON &&
            fabs(T_.at<double>(0, 0)) <= EPSILON &&
            fabs(T_.at<double>(1, 0)) <= EPSILON &&
            fabs(T_.at<double>(2, 0)) <= EPSILON)
            break;
    }
}
