#include "ft/ft.hpp"
#include <opencv2/opencv.hpp>

int main(void) {
    cv::Mat lenaPng = cv::imread("/home/maxi/projects/cv_education/lab_4/source/Lenna.png");
    cv::resize(lenaPng, lenaPng, cv::Size(150, 150));
    cv::cvtColor(lenaPng, lenaPng, cv::COLOR_BGR2GRAY);
    lenaPng.convertTo(lenaPng, CV_64F);

    cv::Mat lenFTCV;
    cv::dft(lenaPng, lenFTCV, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat lenFT = ft::simpleDFT(lenaPng);

    lenFTCV = ft::postprocessDFT(lenFTCV);
    lenFT = ft::postprocessDFT(lenFT);
    //cv::imshow("img", lenaPng);

    cv::imshow("ft", lenFT);
    cv::imshow("ftCV", lenFTCV);
    cv::waitKey(-1);

    cv::normalize(lenFT, lenFT, 0, 255, cv::NormTypes::NORM_MINMAX);
    cv::normalize(lenFTCV, lenFTCV, 0, 255, cv::NormTypes::NORM_MINMAX);
    cv::imwrite("my.png", lenFT);
    cv::imwrite("etalon.png", lenFTCV);
    return 0;
}