#include "multiple_view_geometry/geometry.hpp"

cv::Point3d get_cross_product(cv::Point3d a, cv::Point3d b)
{
    return cv::Point3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

cv::Mat_<double> get_rotation_matrix(cv::Point3d a, cv::Point3d b)
{
    cv::Point3d v = get_cross_product(a, b);
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

cv::Mat_<double> linear_ls_triangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1)
{
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

cv::Mat_<double> iterative_linear_ls_triangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1)
{
    double wi = 1, wi1 = 1;
    cv::Mat_<double> X(4, 1);

    cv::Mat_<double> X_ = linear_ls_triangulation(u, P, u1, P1);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;

    for (int i = 0; i < 10; ++i)
    {
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

void estimate_3d_affine(std::vector<cv::Point3d> source, std::vector<cv::Point3d> destination, cv::Mat_<double>& R, cv::Mat_<double>& T)
{
    /*Find the scale by finding the ratio of some distances*/
    int common_size = source.size();
    double source_distance = 0, destination_distance = 0, scale;
    for (int i = 0; i < common_size - 1; ++i)
    {
        source_distance += get_euclid_distance(source[i], source[i + 1]);
        destination_distance += get_euclid_distance(destination[i], destination[i + 1]);
    }
    scale = destination_distance / source_distance;

    /*Bring point sets to the same scale*/
    for (int i = 0; i < common_size; ++i)
    {
        source[i] *= scale;
    }

    /*Find the centroids*/
    cv::Point3d centroid_source(0, 0, 0), centroid_destination(0, 0, 0);
    for (int i = 0; i < common_size; ++i)
    {
        centroid_source += source[i];
        centroid_destination += destination[i];
    }

    centroid_source.x /= common_size;
    centroid_source.y /= common_size;
    centroid_source.z /= common_size;

    centroid_destination.x /= common_size;
    centroid_destination.y /= common_size;
    centroid_destination.z /= common_size;

    /*Shift to origin*/
    for (int i = 0; i < common_size; ++i)
    {
        source[i] -= centroid_source;
        destination[i] -= centroid_destination;
    }

    /*Find covariance matrix*/
    cv::Mat_<double> covariance_matrix(3, 3);
    covariance_matrix.setTo(cv::Scalar::all(0));
    for (int i = 0; i < common_size; ++i)
    {
        cv::Mat_<double> covariance_matrix_i = (cv::Mat_<double>(3, 3) << source[i].x * destination[i].x, source[i].x * destination[i].y, source[i].x * destination[i].z,
                                                source[i].y * destination[i].x, source[i].y * destination[i].y, source[i].y * destination[i].z,
                                                source[i].z * destination[i].x, source[i].z * destination[i].y, source[i].z * destination[i].z);
        covariance_matrix += covariance_matrix_i;
    }

    /*Find rotation using SVD*/
    cv::SVD svd(covariance_matrix);
    R = svd.vt.t() * svd.u.t();
    R *= scale;

    /*Find translation*/
    T = (cv::Mat_<double>(3, 1) << scale * (centroid_destination.x - centroid_source.x),
         scale * (centroid_destination.y - centroid_source.y),
         scale * (centroid_destination.z - centroid_source.z));
}

void estimate_iterative_3d_affine(std::vector<cv::Point3d> source, std::vector<cv::Point3d> destination, cv::Mat_<double>& R, cv::Mat_<double>& T)
{
    estimate_3d_affine(source, destination, R, T);

    for (int i = 0; i < 10; ++i)
    {
        std::vector<cv::Point3d> src_;
        for (size_t j = 0; j < source.size(); ++j)
        {
            cv::Mat_<double> src_i = (cv::Mat_<double>(3, 1) << source[j].x, source[j].y, source[j].z);
            cv::Mat_<double> dst_i = R * src_i + T;
            src_.push_back(cv::Point3d(dst_i.at<double>(0, 0),
                                       dst_i.at<double>(1, 0),
                                       dst_i.at<double>(2, 0)));
        }

        cv::Mat_<double> R_, T_;
        estimate_3d_affine(src_, destination, R_, T_);
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
