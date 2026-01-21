/*******************************************************
* File: processing.cpp
*
* Description: Functions for processing video
*
* Author: Logan Schmid
*
* Revision history
*
********************************************************/
// #include <cmath>
#include <opencv2/opencv.hpp>
#include "processing.hpp"

using namespace cv;
using namespace std;

/*-----------------------------------------------------
* Function: to442_grayscale
*
* Description: Converts image to grayscale using the BT.709 algorithm
* Gray = 0.0722B + 0.7152G + 0.2126R
*
* param frame: Mat
*
* return: Mat
*--------------------------------------------------------*/ 
 Mat to442_grayscale(Mat frame) {
    int num_rows = frame.rows;
    int num_cols = frame.cols;
    
    Mat frame_gray(num_rows, num_cols, CV_8UC1, Scalar(0));
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            Vec3b &pix = frame.at<Vec3b>(row,col);
            uchar gray = static_cast<uchar>(0.0722*pix[0] + 0.7152*pix[1] + 0.2126*pix[2]);
            frame_gray.at<uchar>(row,col) = gray;
        }
    }
    return frame_gray;
}

/*-----------------------------------------------------
* Function: to442_sobel
*
* Description: Applies a Sobel filter to an image using a manual implementation
*
* param frame: Mat: the input grayscale image
*
* return: Mat
*--------------------------------------------------------*/
Mat to442_sobel(Mat frame) {
    int G_x[3][3] = GX;
    int G_y[3][3] = GY;
    int num_rows = frame.rows;
    int num_cols = frame.cols;
    Mat sobel_frame(num_rows-2, num_cols-2, CV_8UC1, Scalar(0));  // act as Gx at first

    // loop through all pixels except the outermost pixel border
    for (int row = 1; row < num_rows-1; row++) {
        for (int col = 1; col < num_cols-1; col++) {
            // For each pixel, convolute the 3x3 matrices GX and GY
            int16_t G_x_sum = 0;
            int16_t G_y_sum = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j =-1; j <= 1; j++) {
                    G_x_sum += static_cast<int16_t>(G_x[i+1][j+1]*frame.at<uchar>(row+i, col+j));
                    G_y_sum += static_cast<int16_t>(G_y[i+1][j+1]*frame.at<uchar>(row+i, col+j));
                }
            }
            sobel_frame.at<uchar>(row-1, col-1) = static_cast<uint8_t>(abs(G_x_sum) + abs(G_y_sum));
        }
    }
    return sobel_frame;
}

Mat builtin_sobel(Mat frame) {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;

    // Use 16-bit signed to avoid overflow
    Sobel(frame, grad_x, CV_16S, 1, 0, 3);
    Sobel(frame, grad_y, CV_16S, 0, 1, 3);  // [web:2][web:6]

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Combine X and Y gradients
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);  // [web:2]

    cout << "The depth of grad from cv sobel fxn is: " << grad.depth() << endl;
    return grad;
}