/*******************************************************
* File: processing.hpp
*
* Description: Functions for processing video
*
* Author: Logan Schmid, Enrique Murillo
*
* Revision history
*
********************************************************/
#ifndef _PROCESSING_HPP
#define _PROCESSING_HPP

using namespace cv;

# define GX { \
    {-1, 0, 1}, \
    {-2, 0, 2}, \
    {-1, 0, 1} \
}

# define GY { \
    {1, 2, 1}, \
    {0, 0, 0}, \
    {-1, -2, -1} \
}

typedef struct {
    const Mat* src;
    Mat* dst;
    int row_0;
    int col_0;
    int h;
    int w;
} threadArgs_t;

/*-----------------------------------------------------
* Function: to442_grayscale_worker
*
* Description: Converts image to grayscale using the BT.709 algorithm
* Gray = 0.0722B + 0.7152G + 0.2126R
*
* param frame: cv::Mat: the input color image
*
* return: void
*--------------------------------------------------------*/ 
void* to442_grayscale_worker(void* threadArgs);


/*-----------------------------------------------------
* Function: to442_sobel_worker
*
* Description: Applies a Sobel filter to an image using a manual implementation
*
* param frame: cv::Mat: the input grayscale image
*
* return: void
*--------------------------------------------------------*/ 
void* to442_sobel_worker(void* threadArgs);


#endif //- _PROCESSING_HPP