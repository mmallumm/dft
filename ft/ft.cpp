#include <cmath>
#include "ft.hpp"

cv::Mat ft::simpleDFT(const cv::Mat img) { 
    cv::Mat temp, res;
    img.convertTo(temp, CV_64FC1);
    res.create(temp.size(), CV_64FC2);

    res.forEach<cv::Vec2d>([&](cv::Vec2d &pixel, const int position[]) { 
        std::complex<double> tempPixel = {0.0, 0.0};
        for(uint i = 0; i < temp.cols; ++i) {
            for(uint j = 0; j < temp.rows; ++j) {
                std::complex<double> firstExp = exp(-ft::j * (2 * M_PI / temp.cols) * static_cast<double>(i * position[0]));
                std::complex<double> secondExp = exp(-ft::j * (2 * M_PI / temp.rows) * static_cast<double>(j * position[1]));
                tempPixel += std::complex<double>(static_cast<double>(temp.at<double>(i, j)), 0.0) * firstExp * secondExp;
            }
        }
        pixel[0] = tempPixel.real();
        pixel[1] = tempPixel.imag();
    });

    return res;
}

cv::Mat ft::postprocessDFT(const cv::Mat ftImg) {
    cv::Mat magnitude;
    cv::Mat ftSplittedImg[2];
    cv::split(ftImg, ftSplittedImg);
    cv::magnitude(ftSplittedImg[0], ftSplittedImg[1], magnitude);
    magnitude += cv::Scalar::all(1);
    cv::log(magnitude, magnitude);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NormTypes::NORM_MINMAX);
    return magnitude;
}