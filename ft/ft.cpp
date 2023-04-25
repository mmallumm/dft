#include "ft.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

void ft::fastDft(Mat &padded, Mat &real, Mat &imag)
{
    int cols = padded.cols;
    int rows = padded.rows;

    std::vector<std::vector<std::complex<float>>> values_complex(
        rows, std::vector<std::complex<float>>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            values_complex[i][j] = std::complex<float>(padded.at<float>(i, j), 0);
        }
    }

    for (int i = 0; i < cols; i++)
    {
        std::vector<std::complex<float>> column(rows);
        for (int j = 0; j < rows; j++)
        {
            column[j] = values_complex[j][i];
        }
        calcFourier(column);
        for (int j = 0; j < rows; j++)
        {
            values_complex[j][i] = column[j];
        }
    }

    for (int i = 0; i < rows; i++)
    {
        calcFourier(values_complex[i]);
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            real.at<float>(i, j) = values_complex[i][j].real();
            imag.at<float>(i, j) = values_complex[i][j].imag();
        }
    }
}

void ft::simpleDft(Mat &values, int flag)
{

    double cols = values.cols;
    double rows = values.rows;
    Mat complex_img(values.size(), CV_32FC2);

    for (int u = 0; u < cols; u++)
    {
        for (int v = 0; v < rows; v++)
        {
            std::complex<float> sum(0.0, 0.0);
            for (int x = 0; x < rows; x++)
            {
                float cosinus = (float)cos(2 * (PI / (double)rows) * (double)v * (double)x);
                float sinus = -(float)sin(2 * (PI / (double)rows) * (double)v * (double)x);
                std::complex<float> b((float)cosinus, (float)sinus);

                if (flag == 1)
                {
                    std::complex<float> sopr(values.at<std::complex<float>>(x, u).real(),
                                             -values.at<std::complex<float>>(x, u).imag());
                    sum = sum + sopr * b;
                }
                else
                {

                    sum = sum + values.at<std::complex<float>>(x, u) * b;
                }
            }
            if (flag == 1)
            {
                std::complex<float> sum2(sum.real() / (float)cols, -sum.imag() / (float)cols);
                sum = sum2;
            }
            complex_img.at<std::complex<float>>(v, u) = sum;
        }
    }

    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            std::complex<float> sum(0.0, 0.0);
            for (int x = 0; x < cols; x++)
            {
                float cosinus = (float)cos(2 * (PI / (double)cols) * (double)u * (double)x);
                float sinus = -(float)sin(2 * (PI / (double)cols) * (double)u * (double)x);
                std::complex<float> b((float)cosinus, (float)sinus);

                if (flag == 1)
                {
                    std::complex<float> sopr(complex_img.at<std::complex<float>>(v, x).real(),
                                             -complex_img.at<std::complex<float>>(v, x).imag());
                    sum = sum + sopr * b;
                }
                else
                {

                    sum = sum + complex_img.at<std::complex<float>>(v, x) * b;
                }
            }
            if (flag == 1)
            {
                std::complex<float> sum2(sum.real() / (float)cols, -sum.imag() / (float)cols);
                sum = sum2;
            }
            values.at<std::complex<float>>(v, u) = sum;
        }
    }
}

void ft::convolution(Mat &values, Mat &kernel, int cols, int rows)
{
    Mat imagePadded = Mat::zeros(Size(values.cols + 2, values.rows + 2), CV_32F);
    Mat kernelPadded = imagePadded.clone();
    Mat readyPicture;

    values.copyTo(imagePadded(cv::Rect(0, 0, values.cols, values.rows)));
    kernel.copyTo(kernelPadded(cv::Rect(0, 0, kernel.cols, kernel.rows)));

    Mat imgDft;
    imgDft.create(imagePadded.rows, imagePadded.cols, CV_32FC2);
    prepareForDft(imgDft, imagePadded);
    simpleDft(imgDft, 0);
    drawSpektr(imgDft, 0, "image_magnitude");
    Mat filterDft;
    filterDft.create(kernelPadded.rows, kernelPadded.cols, CV_32FC2);
    prepareForDft(filterDft, kernelPadded);
    simpleDft(filterDft, 0);
    drawSpektr(filterDft, 0, "kernel_magnitude");

    mulSpectrums(imgDft, filterDft, readyPicture, 0, false);
    drawSpektr(readyPicture, 0, "final_magnitude");
    simpleDft(readyPicture, 1);
    Mat cropped = readyPicture(cv::Rect(0, 0, cols, rows)).clone();
    drawSpektr(cropped, 1, "filtered_image");
}

void ft::frequencyFilter(Mat &values, int flag, int cols, int rows)
{
    Mat frequency;

    prepareForDft(frequency, values);
    simpleDft(frequency, 0);
    drawSpektr(frequency, 0, "img_magnitude");

    krasivSpektr(frequency);
    if (flag == 0)
    {
        cv::circle(frequency, cv::Point(frequency.cols / 2, frequency.rows / 2), 20,
                   cv::Scalar::all(0), -1);
    }
    else
    {
        Mat mask = frequency.clone();
        circle(mask, Point(frequency.cols / 2, frequency.rows / 2), 20, Scalar::all(0), -1);
        cv::bitwise_xor(frequency, mask, frequency);
    }
    krasivSpektr(frequency);
    drawSpektr(frequency, 0, "frequency");

    simpleDft(frequency, 1);
    Mat cropped = frequency(cv::Rect(0, 0, cols, rows)).clone();

    Mat splitted[2];
    split(cropped, splitted);
    splitted[0].convertTo(splitted[0], CV_8U);
    imshow("filtered_image", splitted[0]);
    waitKey(0);
}

void ft::detectNumber(Mat &carNumber, Mat &number)
{
    Mat carNumbersPadded = Mat::zeros(
        Size(carNumber.cols + number.cols - 1, carNumber.rows + number.rows - 1), CV_32F);
    Mat numberPadded = carNumbersPadded.clone();
    carNumber.copyTo(carNumbersPadded(cv::Rect(0, 0, carNumber.cols, carNumber.rows)));
    number.copyTo(numberPadded(cv::Rect(0, 0, number.cols, number.rows)));

    Mat complexCarNumbers;
    prepareForDft(complexCarNumbers, carNumbersPadded);
    simpleDft(complexCarNumbers, 0);

    Mat complex_number;
    prepareForDft(complex_number, numberPadded);
    simpleDft(complex_number, 0);

    Mat readyPicture;
    mulSpectrums(complexCarNumbers, complex_number, readyPicture, 0, true);

    simpleDft(readyPicture, 1);
    Mat croppedCarNumber = readyPicture(cv::Rect(0, 0, carNumber.cols, carNumber.rows)).clone();
    drawSpektr(croppedCarNumber, 1, "picture_back");

    Mat splitted[2];
    split(croppedCarNumber, splitted);
    normalize(splitted[0], splitted[0], 0, 1, NormTypes::NORM_MINMAX);

    // treshold
    double maxValue;
    minMaxLoc(splitted[0], nullptr, &maxValue);
    cv::threshold(splitted[0], splitted[0], maxValue - 0.01, 1, 1);
    imshow("result", splitted[0]);
    waitKey(0);
}

void ft::krasivSpektr(Mat &magI)
{
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

unsigned int ft::inverse(unsigned int x, int length)
{
    unsigned int result = 0;
    unsigned int bit = 1u;
    unsigned int reverse = 1u << (length - 1);
    for (int i = 0; i < length && x != 0; i++)
    {
        if (x & bit)
        {
            result |= reverse;
            x &= ~bit;
        }
        bit <<= 1;
        reverse >>= 1;
    }
    return result;
}

static void ft::reposeElemnts(std::vector<int> &number, int size)
{
    int length = 0;
    while (1u << length < (unsigned int)size)
        length++;

    for (int i = 0; i < size; i++)
    {
        unsigned int j = inverse(i, length);
        if (i <= j)
            swap(number[i], number[j]);
    }
}

void ft::complexInit(std::complex<float> &a, std::complex<float> &b, int size, int n)
{
    double real = cos(2 * PI / (double)size * (double)n);
    double imag = -sin(2 * PI / (double)size * (double)n);
    std::complex<float> W((float)real, (float)imag);
    std::complex<float> X = a + W * b;
    std::complex<float> Y = a - W * b;
    a = X;
    b = Y;
}

void ft::calcFourier(std::vector<std::complex<float>> &values)
{
    int size = (int)values.size();
    std::vector<int> number(values.size());
    int a = 0;
    for (int i = 0; i < values.size(); i++)
    {
        number[i] = a;
        a++;
    }
    reposeElemnts(number, (int)values.size());

    std::vector<std::complex<float>> values2(values.size());

    for (int i = 0; i < values.size(); i++)
    {
        values2[i] = values[i];
    }

    int k = 2;
    int counter;
    while ((size / k) >= 1)
    {
        int b = 0;
        for (int i = 0; i < size / k; i++)
        {
            counter = 0;
            for (int j = b; j < (b + k / 2); j++)
            {
                complexInit(values2[number[j]], values2[number[j + k / 2]], k, counter);
                counter++;
            }
            b = b + k;
        }
        k = k * 2;
    }

    for (int i = 0; i < values.size(); i++)
    {
        values[i] = values2[number[i]];
    }
}

void ft::prepareForDft(Mat &complexImg, Mat &real)
{
    Mat planes2[] = {Mat(real), Mat::zeros(real.size(), CV_32F)};
    merge(planes2, 2, complexImg);
}

void ft::drawSpektr(Mat &values, int flag, std::string image_name)
{

    Mat splitted[2];
    split(values, splitted);
    magnitude(splitted[0], splitted[1], splitted[0]);
    splitted[0] += Scalar::all(1);
    if (flag == 0)
    {
        log(splitted[0], splitted[0]);
    }
    normalize(splitted[0], splitted[0], 0, 1, NormTypes::NORM_MINMAX);

    if (flag == 0)
    {
        krasivSpektr(splitted[0]);
    }
    imshow(image_name, splitted[0]);
    waitKey(0);
}