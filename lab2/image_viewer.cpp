/*********************************************************
* File: image_viewer.cpp
*
* Description: Simple image viewer using opencv
*
* Author: Logan Schmid
*
* Revisions:
*
**********************************************************/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        perror("Incorrect usage - use via: 'image_viewer [image_path]");
        return 1;
    }

    string image_path = argv[1];
    Mat im = imread(image_path, IMREAD_COLOR);

    if(im.empty()) {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }

    imshow("Display window", im);
    waitKey(0); // Wait for a keystroke to close the window

    return 0;
}