#include "multiple_view_geometry/norm.hpp"

double get_euclid_distance(cv::Point2d p1, cv::Point2d p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}

double get_euclid_distance(cv::Point3d p1, cv::Point3d p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dz = p2.z - p1.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}
