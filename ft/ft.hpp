#ifndef FT_H
#define FT_H

#include <complex>
#include <opencv2/core.hpp>

const double PI = 3.14159265358979323846;

namespace ft
{

// public part
void krasivSpektr(cv::Mat &magI);
void fastDft(cv::Mat &padded, cv::Mat &real, cv::Mat &imag);
void simpleDft(cv::Mat &values, int flag);
void drawSpektr(cv::Mat &values, int flag, std::string a);
void prepareForDft(cv::Mat &complexImg, cv::Mat &real);
void convolution(cv::Mat &values, cv::Mat &kernel, int cols, int rows);
void frequencyFilter(cv::Mat &values, int flag, int cols, int rows);
void detectNumber(cv::Mat &carNumber, cv::Mat &number);

// private part
unsigned int inverse(unsigned int x, int length);
static void reposeElemnts(std::vector<int> &number, int size);
void complexInit(std::complex<float> &a, std::complex<float> &b, int size, int n);
void calcFourier(std::vector<std::complex<float>> &values);

} // namespace ft
#endif