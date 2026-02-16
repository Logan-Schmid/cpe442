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
* param src: Mat*: the input color image
* param dst: Mat*: the output grayscale image
* param r0: int: the starting row index
* param c0: int: the starting column index
* param h: int: the height of the processing region
* param w: int: the width of the processing region
*
* return: void
*--------------------------------------------------------*/ 
 void to442_grayscale(Mat* src, Mat* dst, int r0, int c0, int h, int w) {
    for (int row = r0; row < r0 + h; row++) {
        for (int col = c0; col < c0 + w; col++) {
            Vec3b &pix = (*src).at<Vec3b>(row,col);
            uchar gray = static_cast<uchar>(0.0722*pix[0] + 0.7152*pix[1] + 0.2126*pix[2]);
            (*dst).at<uchar>(row,col) = gray;
        }
    }
}

/*-----------------------------------------------------
* Function: to442_sobel
*
* Description: Applies a Sobel filter to an image using a manual implementation
*
* param src: Mat*: the input grayscale image
* param dst: Mat*: the output edge-detected image
* param r0: int: the starting row index
* param c0: int: the starting column index
* param h: int: the height of the processing region
* param w: int: the width of the processing region
*
* return: void
*--------------------------------------------------------*/
void to442_sobel(Mat* src, Mat* dst, int r0, int c0, int h, int w) {
    int G_x[3][3] = GX;
    int G_y[3][3] = GY;
    // loop through all pixels except the outermost pixel border
    for (int row = r0+1; row < r0+h-1; row++) {
        for (int col = c0+1; col < c0+w-1; col++) {
            // For each pixel, convolute the 3x3 matrices GX and GY
            int16_t G_x_sum = 0;
            int16_t G_y_sum = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j =-1; j <= 1; j++) {
                    G_x_sum += static_cast<int16_t>(G_x[i+1][j+1]*(*src).at<uchar>(row+i, col+j));
                    G_y_sum += static_cast<int16_t>(G_y[i+1][j+1]*(*src).at<uchar>(row+i, col+j));
                }
            }
            uint16_t G = abs(G_x_sum) + abs(G_y_sum);
            if (G > 255) {
                G = 255;
            }
            (*dst).at<uchar>(row-1, col-1) = static_cast<uint8_t>(G);
        }
    }
}