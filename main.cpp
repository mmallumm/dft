#include "ft/ft.hpp"
#include <opencv2/opencv.hpp>

#define MODE 3

#if MODE == 0
int main(void)
{
    cv::Mat img = cv::imread("/home/maxi/projects/cv_education/lab_4/source/dance.jpg");
    cv::resize(img, img, cv::Size(300, 400));
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("original", img);
    cv::waitKey(0);

    cv::Mat paddedImg;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, paddedImg, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));
    paddedImg.convertTo(paddedImg, CV_32F);

    // OpenCV dft
    cv::TickMeter timerDftCV;
    cv::Mat complexImg_0;
    ft::prepareForDft(complexImg_0, paddedImg);
    timerDftCV.start();
    cv::dft(complexImg_0, complexImg_0);
    timerDftCV.stop();
    std::cout << "OpenCV dft, total time: " << timerDftCV.getTimeSec() << std::endl;
    ft::drawSpektr(complexImg_0, 0, "dft_opencv");

    // Custom simple dft
    cv::TickMeter timerDftCustom;
    cv::Mat complexImg_1;
    ft::prepareForDft(complexImg_1, paddedImg);
    timerDftCustom.start();
    ft::simpleDft(complexImg_1, 0);
    timerDftCustom.stop();
    std::cout << "Custom simple dft, total time: " << timerDftCustom.getTimeSec() << std::endl;
    ft::drawSpektr(complexImg_1, 0, "dft_custom_simple");

    // Custom simple inverse dft
    ft::simpleDft(complexImg_1, 1);
    ft::drawSpektr(complexImg_1, 1, "dft_custom_simple_inverse");

    // Custom fast dft
    cv::TickMeter timerfdftCustom;
    cv::Mat paddedImage_fft;
    int m_fft = pow(2, ceil(log2(img.rows)));
    int n_fft = pow(2, ceil(log2(img.cols)));
    cv::copyMakeBorder(img, paddedImage_fft, 0, m_fft - img.rows, 0, n_fft - img.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));
    paddedImage_fft.convertTo(paddedImage_fft, CV_32F);
    cv::Mat imag_fft;
    imag_fft.create(paddedImage_fft.rows, paddedImage_fft.cols, CV_32F);
    cv::Mat real_fft;
    real_fft.create(paddedImage_fft.rows, paddedImage_fft.cols, CV_32F);
    timerfdftCustom.start();
    ft::fastDft(paddedImage_fft, real_fft, imag_fft);
    timerfdftCustom.stop();
    std::cout << "Custom fast dft, total time: " << timerfdftCustom.getTimeSec() << std::endl;

    // displaying spectrum of fft
    cv::Mat spectrum_fft;
    cv::magnitude(real_fft, imag_fft, spectrum_fft);
    spectrum_fft += cv::Scalar::all(1);
    cv::log(spectrum_fft, spectrum_fft);
    cv::normalize(spectrum_fft, spectrum_fft, 0, 1, cv::NormTypes::NORM_MINMAX);
    ft::krasivSpektr(spectrum_fft);
    imshow("spectr_fft", spectrum_fft);
    cv::waitKey(0);

    return 0;
}

#elif MODE == 1
int main(void)
{
    // convolution
    // making different kernels
    cv::Mat img = cv::imread("/home/maxi/projects/cv_education/lab_4/source/dance.jpg");
    cv::resize(img, img, cv::Size(300, 400));
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("original", img);
    cv::waitKey(0);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));
    padded.convertTo(padded, CV_32F);

    cv::Mat laplace_kernel = cv::Mat::zeros(3, 3, CV_32F);
    laplace_kernel.at<float>(0, 1) = 1;
    laplace_kernel.at<float>(1, 0) = 1;
    laplace_kernel.at<float>(2, 1) = 1;
    laplace_kernel.at<float>(1, 2) = 1;
    laplace_kernel.at<float>(1, 1) = -4;

    cv::Mat box_kernel = cv::Mat::zeros(3, 3, CV_32F);
    box_kernel.at<float>(0, 0) = 1;
    box_kernel.at<float>(0, 1) = 1;
    box_kernel.at<float>(0, 2) = 1;
    box_kernel.at<float>(1, 0) = 1;
    box_kernel.at<float>(1, 1) = 1;
    box_kernel.at<float>(1, 2) = 1;
    box_kernel.at<float>(2, 0) = 1;
    box_kernel.at<float>(2, 1) = 1;
    box_kernel.at<float>(2, 2) = 1;

    cv::Mat sobel_vert_kernel = cv::Mat::zeros(3, 3, CV_32F);
    sobel_vert_kernel.at<float>(0, 0) = 1;
    sobel_vert_kernel.at<float>(0, 1) = 2;
    sobel_vert_kernel.at<float>(0, 2) = 1;
    sobel_vert_kernel.at<float>(2, 0) = -1;
    sobel_vert_kernel.at<float>(2, 1) = -2;
    sobel_vert_kernel.at<float>(2, 2) = -1;

    cv::Mat sobel_hor_kernel = cv::Mat::zeros(3, 3, CV_32F);
    sobel_hor_kernel.at<float>(0, 0) = -1;
    sobel_hor_kernel.at<float>(0, 2) = +1;
    sobel_hor_kernel.at<float>(1, 0) = -2;
    sobel_hor_kernel.at<float>(1, 2) = +2;
    sobel_hor_kernel.at<float>(2, 0) = -1;
    sobel_hor_kernel.at<float>(2, 2) = +1;

    // making convolution with different kernels
    ft::convolution(padded, laplace_kernel, img.cols, img.rows);
    ft::convolution(padded, box_kernel, img.cols, img.rows);
    ft::convolution(padded, sobel_vert_kernel, img.cols, img.rows);
    ft::convolution(padded, sobel_hor_kernel, img.cols, img.rows);
}

#elif MODE == 2
int main(void)
{
    // convolution
    // making different kernels
    cv::Mat img = cv::imread("/home/maxi/projects/cv_education/lab_4/source/dance.jpg");
    cv::resize(img, img, cv::Size(300, 400));
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("original", img);
    cv::waitKey(0);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));
    padded.convertTo(padded, CV_32F);

    // frequency filtering
    const int highPass = 0;
    const int lowPass = 1;
    // ft::frequencyFilter(padded, highPass, img.cols, img.rows);
    ft::frequencyFilter(padded, lowPass, img.cols, img.rows);
}

#elif MODE == 3
int main(void)
{
    // detecting numbers

    cv::Mat carNumber = cv::imread("/home/maxi/projects/cv_education/lab_4/source/car_number.png");
    cv::cvtColor(carNumber, carNumber, cv::COLOR_BGR2GRAY);
    cv::imshow("car_numbers", carNumber);
    cv::waitKey(0);

    cv::Mat number1 = cv::imread("/home/maxi/projects/cv_education/lab_4/source/9.png");
    cv::cvtColor(number1, number1, cv::COLOR_BGR2GRAY);
    cv::imshow("number", number1);
    cv::waitKey(0);
    ft::detectNumber(carNumber, number1);

    cv::Mat number2 = cv::imread("/home/maxi/projects/cv_education/lab_4/source/B.png");
    cv::cvtColor(number2, number2, cv::COLOR_BGR2GRAY);
    cv::imshow("number", number2);
    cv::waitKey(0);
    ft::detectNumber(carNumber, number2);

    cv::Mat number3 = cv::imread("/home/maxi/projects/cv_education/lab_4/source/K.png");
    cv::cvtColor(number3, number3, cv::COLOR_BGR2GRAY);
    cv::imshow("number", number3);
    cv::waitKey(0);
    ft::detectNumber(carNumber, number3);
}
#endif