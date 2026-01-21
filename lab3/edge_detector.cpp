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
    if (argc != 3) {
        cerr << "Incorrect usage - use via: 'edge_detector [video_path] [sobel_option = {\"442\"/\"cv\"}]" << endl;
        return -1;
    }

    string cap_path = argv[1];
    string sobel_option = argv[2];

    // Initialize video reader
    VideoCapture cap(cap_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file: " << cap_path << endl;
        return -1;
    } else {
        cout << "\"" << cap_path << "\" opened successfully!" << endl;
    }

    // get and print video attributes
    int frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));    
    cout << "Total frames: " << frame_count << ", FPS: " << fps << endl;
    cout << "Width: " << width << ", Height: " << height << endl;

    // Initialize video writer
    string out_filename;
    Size out_size;
    if (sobel_option == "442") {
        out_filename = "442_sobel_" + cap_path;
        out_size = Size(width-2, height-2);
    } else {
        out_filename = "cv_sobel_" + cap_path;
        out_size = Size(width, height);
    }
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
    bool isColor = false;
    VideoWriter writer(out_filename, codec, fps, out_size, isColor);
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write" << endl;
        return -1;
    }

    // Read, save, and display each frame of the video
    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "End of video or error occurred." << endl;
            break;
        }

        // convert vid to grayscale (uchar dtype)
        Mat frame_gray = to442_grayscale(frame);

        Mat frame_sobel;
        // apply Sobel filter (edge-detection)
        if (sobel_option == "442") {
            frame_sobel = to442_sobel(frame_gray);
        } else if (sobel_option == "cv") {
            frame_sobel = builtin_sobel(frame_gray);
        } else {
            cerr << "Error: sobel_option command-line arg must be either \"442\" or \"cv\"" << endl;
            return -1;
        }

        // write filtered frame
        writer.write(frame_sobel);
        
        // Display the frame
        imshow("Display Window", frame_sobel);

        // Wait for appropriate time between frames and check if 'q' is pressed to exit
        // char key = waitKey(1000/fps);
        char key = waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    // Release the video capture object and close any OpenCV windows
    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;
}