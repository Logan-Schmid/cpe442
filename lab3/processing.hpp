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

//- #defines go here

/*-----------------------------------------------------
* Function: to442_grayscale
*
* Description: Converts image to grayscale using the BT.709 algorithm
* Gray = 0.0722B + 0.7152G + 0.2126R
*
* param frame: cv::Mat
*
* return: cv::Mat
*--------------------------------------------------------*/ 
cv::Mat to442_grayscale(cv::Mat frame);

#endif //- _PROCESSING_HPP