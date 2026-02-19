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
#include <arm_neon.h>
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
    int16_t G_x[3][3] = GX;
    int16_t G_y[3][3] = GY;
    uint8_t* rowPtr;
    // loop through all pixels except the outermost pixel border
    for (int row = r0+1; row < r0+h-1; row++) {
        for (int col = c0+1; col < c0+w-1; col += 8) {
            // For each pixel, convolute the 3x3 matrices GX and GY
            int16x8_t G_x_sum = vdupq_n_s16(0);
            int16x8_t G_y_sum = vdupq_n_s16(0);
            for (int i = -1; i <= 1; i++) {
                rowPtr = (*src).ptr<uint8_t>(row+i);
                for (int j =-1; j <= 1; j++) {
                    uint8x8_t pixs = vld1_u8(rowPtr+col+j);
                    uint16x8_t pixs_intermed_u = vmovl_u8(pixs);
                    int16x8_t pixs_intermed_s = vreinterpretq_s16_u16(pixs_intermed_u);

                    int16x8_t conv_prod_x = vmulq_n_s16(pixs_intermed_s, G_x_sum[i+1][j+1]);
                    int16x8_t conv_prod_y = vmulq_n_s16(pixs_intermed_s, G_y_sum[i+1][j+1]);
                    
                    G_x_sum = vaddq_s16(G_x_sum, conv_prod_x);
                    G_y_sum = vaddq_s16(G_y_sum, conv_prod_y);
                }
            }
            int16x8_t G_x_sum_abs = vabsq_s16(G_x_sum);
            int16x8_t G_y_sum_abs = vabsq_s16(G_y_sum);

            int16x8_t G_16 = vaddq_s16(G_x_sum_abs, G_y_sum_abs);
            
            uint8x8_t G_8 = vqmovun_s16(G_16);

            uint8_t* rowPtr_dst = (*dst).ptr<uint8_t>(row-1);
            vst1_u8(G_8, rowPtr_dst+col-1);
        }
    }
}