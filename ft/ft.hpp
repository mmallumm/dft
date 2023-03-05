#ifndef FT_H
#define FT_H

#include <opencv2/core.hpp>
#include <complex>

namespace ft {
    constexpr std::complex<double> j(0.0, 1.0);

    cv::Mat simpleDFT(const cv::Mat);

    cv::Mat postprocessDFT(const cv::Mat);
}
#endif