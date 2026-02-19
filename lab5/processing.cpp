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
#inlcude <arm_noen.h>
#include <cstdio>
#include <cstdint>

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
//Src = source img color
//dst = output img 
//row = starting row index
//col = starting col index
//h = heigh of the processing region
//w = width of the processing region

//we iterate through the row
//then inner iterate through the col

void to442_grayscale(Mat* source, Mat *output, int row_size, int col_size, int h, int w){
    for(int row = row_size, row < 0 + h; row++){
        uchar* source_pointer = source->ptr<uchar>(row) + 3 * col_size;
        uchar* output_pointer = output->ptr<uchar>(row) + col_size;
        int col = 0;
        for(; col <= w - 16; col += 16){

            
            uint8x16x3_t bgr = vld3q_u8(source_pointer);

            uint16x8_t b_lo = vmovl_u8(veg_low_u8(bgr.val[0]));
            uint16x8_t b_hi = vmovl_u8(veg_high_u8(bgr.val[0]));
            uint16x8_t g_lo = vmovl_u8(veg_low_u8(bgr.val[1]));
            uint16x8_t g_hi = vmovl_u8(veg_high_u8(bgr.val[1]));
            uint16x8_t r_lo = vmovl_u8(veg_low_u8(bgr.val[2]));
            uint16x8_t r_hi = vmovl_u8(veg_high_u8(bgr.val[2]));
            

            // float gray_lo = vmulq_n_u16(b_lo, 19);
            // gray_lo = vmlaq_n_u16(gray_lo, g_lo, 183);
            // gray_lo = vmlaq_n_u16(gray_lo, r_lo, 54);
            // // gray_lo = gray_lo + (g_lo * 183);

            // float gray_hi = vmulq_n_u16(b_hi, 19);
            // gray_hi = vmlaq_n_u16(gray_hi, g_hi, 183);
            // gray_hi = vmlaq_n_u16(gray_hi, r_hi, 54);


            uint16x8_t gray_lo = vmulq_n_u16(b_lo, 19);
            gray_lo = vmlaq_n_u16(gray_lo, g_lo, 183);
            gray_lo = vmlaq_n_u16(gray_lo, r_lo, 54);
            // gray_lo = gray_lo + (g_lo * 183);

            uint16x8_t gray_hi = vmulq_n_u16(b_hi, 19);
            gray_hi = vmlaq_n_u16(gray_hi, g_hi, 183);
            gray_hi = vmlaq_n_u16(gray_hi, r_hi, 54);

            gray_hi = vshrq_n_u16(gray_hi, 8);
            gray_lo = vshrq_n_u16(gray_lo, 8);

            uint8x16_t gray = vcombine_u8(vqmovn_u16(gray_lo), vqmovn_u16(gray_hi));

            source_pointer += 16 *3;
            output_pointer += 16;
            // gray_lo += r_lo 
        }

        for(; col < w; col++){
            Vec3b &pix = source->at<Vec3b>(row, col_size + col);
            uchar gray = (19*pix[0] + 183*pix[1] + 54*pix[2]) >> 8;
            output->at<uchar>(row, col_size + col) = gray;
        }

    }
}
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