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
        cerr << "Incorrect usage - use via: 'edge_detector [video_path]" << endl;
        return 1;
    }

    string cap_path = argv[1];
    VideoCapture cap;
    cap.open(cap_path);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file: " << cap_path << endl;
        return 1;
    } else {
        cout << "\"" << cap_path << "\" opened successfully!" << endl;
    }

    int frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));    
    cout << "Total frames: " << frame_count << ", FPS: " << fps << endl;
    cout << "Width: " << width << ", Height: " << height << endl;

    // Read and display each frame of the video
    int ctr = 0;
    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "End of video or error occurred." << endl;
            break;
        }

        Mat frame_gray = to442_grayscale(frame);

        // Display the frame
        imshow("Display Window", frame_gray);

        // Wait for appropriate time between frames and check if 'q' is pressed to exit
        // char key = waitKey(1000/fps);
        char key = waitKey(1);
        if (key == 'q') {
            break;
        }
        ctr++;
    }

    // Save processed video

    // Videowriter out("./processed_video.mp4", CV_FOURCC('M', 'P', '4', 'V'), 30.0, Size(width-2, height-2), false);

    // Release the video capture object and close any OpenCV windows
    cap.release();
    destroyAllWindows();
    return 0;
}