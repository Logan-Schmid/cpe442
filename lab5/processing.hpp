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
    Mat* src;
    Mat* gray;
    Mat* sobel;
    int row_0;
    int col_0;
    int h;
    int w;
    int thread_id;
} threadArgs_t;

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
void to442_grayscale(Mat* src, Mat* dst, int r0, int c0, int h, int w);


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
void to442_sobel(Mat* src, Mat* dst, int r0, int c0, int h, int w);


#endif // _PROCESSING_HPP