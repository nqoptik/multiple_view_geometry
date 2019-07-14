#include "multiple_view_geometry/norm.hpp"

double get_euclid_distance(cv::Point2d point_1, cv::Point2d point_2)
{
    double d_x = point_2.x - point_1.x;
    double d_y = point_2.y - point_1.y;
    return sqrt(d_x * d_x + d_y * d_y);
}

double get_euclid_distance(cv::Point3d point_1, cv::Point3d point_2)
{
    double d_x = point_2.x - point_1.x;
    double d_y = point_2.y - point_1.y;
    double d_z = point_2.z - point_1.z;
    return sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
}
