/*******************************************************
* File: processing.hpp
*
* Description: Functions for processing video
*
* Author: Logan Schmid
*
* Revision history
*
********************************************************/
#ifndef _PROCESSING_HPP
#define _PROCESSING_HPP

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

/*-----------------------------------------------------
* Function: to442_grayscale
*
* Description: Converts image to grayscale using the BT.709 algorithm
* Gray = 0.0722B + 0.7152G + 0.2126R
*
* param frame: cv::Mat: the input color image
*
* return: cv::Mat
*--------------------------------------------------------*/ 
cv::Mat to442_grayscale(cv::Mat frame);


/*-----------------------------------------------------
* Function: to442_sobel
*
* Description: Applies a Sobel filter to an image using a manual implementation
*
* param frame: cv::Mat: the input grayscale image
*
* return: cv::Mat
*--------------------------------------------------------*/ 
cv::Mat to442_sobel(cv::Mat frame);


/*-----------------------------------------------------
* Function: builtin_sobel
*
* Description: Applies a Sobel filter to an image using openCV's Sobel filter. Used for validation of to442_sobel.
*
* param frame: cv::Mat: the input grayscale image
*
* return: cv::Mat
*--------------------------------------------------------*/ 
cv::Mat builtin_sobel(cv::Mat frame);

#endif //- _PROCESSING_HPP