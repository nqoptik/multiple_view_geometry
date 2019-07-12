#include "multiple_view_geometry/loadimages.hpp"

std::string int_to_image_name(int number, std::string extenstion)
{
    std::string imgName;
    if (number < 10)
    {
        imgName.append("0000");
    }
    else if (number < 100)
    {
        imgName.append("000");
    }
    else if (number < 1000)
    {
        imgName.append("00");
    }
    else if (number < 10000)
    {
        imgName.append("0");
    }
    imgName.append(std::to_string(number));
    imgName.append(extenstion);
    return imgName;
}

std::vector<cv::Mat> load_images(std::string path, int first, int last)
{
    std::vector<cv::Mat> images;
    for (int i = first; i <= last; i++)
    {
        std::string imgPath = path;
        std::string imgName = int_to_image_name(i, ".png");
        imgPath.append(imgName);
        cv::Mat image = cv::imread(imgPath);
        if (image.empty())
            break;
        images.push_back(image);
    }
    return images;
}
