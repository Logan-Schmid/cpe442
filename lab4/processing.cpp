/*******************************************************
* File: processing.cpp
*
* Description: Functions for processing video
*
* Author: Logan Schmid, Enrique Murillo
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
* Function: to442_grayscale_worker
*
* Description: Converts image to grayscale using the BT.709 algorithm
* Gray = 0.0722B + 0.7152G + 0.2126R
*
* param threadArgs: *void: RGB frame with border information
*
* return: void *
*--------------------------------------------------------*/ 
 void* to442_grayscale_worker(void *threadArgs) {
    threadArgs_t* args = static_cast<threadArgs_t*>(threadArgs);
    
    const Mat& src = *args->src;
    Mat& dst = *args->dst;
    for (int row = args->row_0; row < args->row_0 + args->h; row++) {
        for (int col = args->col_0; col < args->col_0 + args->w; col++) {
            const Vec3b &pix = src.at<Vec3b>(row,col);
            uchar gray = static_cast<uchar>(0.0722*pix[0] + 0.7152*pix[1] + 0.2126*pix[2]);
            dst.at<uchar>(row,col) = gray;
        }
    }
    return nullptr;
}

/*-----------------------------------------------------
* Function: to442_sobel_worker
*
* Description: Applies a Sobel filter to an image using a manual implementation
*
* param threadArgs: *void: grayscaled frame with border information
*
* return: void *
*--------------------------------------------------------*/
void* to442_sobel_worker(void* threadArgs) {
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
            uint16_t G = abs(G_x_sum) + abs(G_y_sum);
            if (G > 255) {
                G = 255;
            }
            sobel_frame.at<uchar>(row-1, col-1) = static_cast<uint8_t>(G);
        }
    }
    return nullptr;
}