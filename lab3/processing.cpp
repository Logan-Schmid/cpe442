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

    
    Mat frame_gray(num_rows, num_cols, CV_32FC1, Scalar(0.0f));
    // cout << "The depth of frame_gray is: " << frame_gray.depth() << endl;
    float gray;
    // frame.convertTo(frame_gray, CV_32F); // seems to work, the lowest bits of gray.flags make up 5 which chat says corresponds to CV_32F
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            Vec3b &pix = frame.at<Vec3b>(row,col);
            gray = 0.0722*(float)pix[0] + 0.7152*(float)pix[1] + 0.2126*(float)pix[2];
            pix[0] = gray;
            pix[1] = gray;
            pix[2] = gray;
            frame_gray.at<float>(row,col) = gray;
            // if (row == 0 && col < 10) cout << "B: " << (int)pix[0] << " G: " << (int)pix[1] << " R: " << (int)pix[2] << " --> gray: " << gray << endl;
        }
    }
    Mat frame_gray8u;
    frame_gray.convertTo(frame_gray8u, CV_8UC1);
    return frame_gray8u;
}