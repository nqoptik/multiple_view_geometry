#include "multiple_view_geometry/loadimages.hpp"

std::string int_to_image_name(int number, std::string extenstion)
{
    std::string image_name;
    if (number < 10)
    {
        image_name.append("0000");
    }
    else if (number < 100)
    {
        image_name.append("000");
    }
    else if (number < 1000)
    {
        image_name.append("00");
    }
    else if (number < 10000)
    {
        image_name.append("0");
    }
    image_name.append(std::to_string(number));
    image_name.append(extenstion);
    return image_name;
}

std::vector<cv::Mat> load_images(std::string path, int first, int last)
{
    std::vector<cv::Mat> images;
    for (int i = first; i <= last; ++i)
    {
        std::string image_path = path;
        std::string image_name = int_to_image_name(i, ".png");
        image_path.append(image_name);
        cv::Mat image = cv::imread(image_path);
        if (image.empty())
            break;
        images.push_back(image);
    }
    return images;
}
