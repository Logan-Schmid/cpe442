/*********************************************************
* File: edge_detector.cpp
*
* Description: Grayscale and apply edge detection on video using opencv
*
* Authors: Logan Schmid
*
* Revisions:
*
**********************************************************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include "processing.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Incorrect usage - use via: 'edge_detector [video_path]" << endl;
        return 1;
    }

    string vid_path = argv[1];
    if (vid_path == "0") {
        VideoCapture vid(0); // connect to webcam using int 0
    } else {
        Videocapture vid(vid_path);
    }

    if (!vid.isOpened()) {
        cout << "Error: Could not open video file: " << vid_path << endl;
        return 1;
    } else {
        cout << "Video file opened successfully!" << endl;
    }

    Mat frame;
    bool ret = vid.read(frame);

    if (ret) {
        imshow("Display window", frame);
        waitKey(0); // Wait for a keystroke to close the window
    } else {
        cout << "Error: Could not read the frame." << endl;
    }

    vid.release();
    return 0;
}