/*********************************************************
* File: edge_detector_threads.cpp
*
* Description: CPE 442 Lab 4 - Grayscale and apply edge
* detection on video using threads to speed up processing
*
* Authors: Logan Schmid, Enrique Murillo
*
* Revisions:
*
**********************************************************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h>
#include "processing.hpp"

#define NUM_THREADS 4

using namespace cv;
using namespace std;

// initialize threads for parallelization of image processing
pthread_t threads[NUM_THREADS];
pthread_barrier_t barrier;

void *thread_statuses[NUM_THREADS];
// void setupBarrier(int numThreads);
Mat grayscale_parallelized(Mat frame);
Mat sobel_parallelized(Mat frame);

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Incorrect usage - use via: 'edge_detector [video_path]'" << endl;
        return -1;
    }

    string cap_path = argv[1];
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
    string out_filename ="442_sobel_" + cap_path;
    Size out_size = Size(width-2, height-2);
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
    bool isColor = false;
    VideoWriter writer(out_filename, codec, fps, out_size, isColor);
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write" << endl;
        return -1;
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Read, save, and display each frame of the video
    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "End of video or error occurred." << endl;
            break;
        }

        setupBarrier(NUM_THREADS+1); // main thread + 4 processing threads
        // parallelized grayscale
        Mat frame_gray = grayscale_parallelized(frame);

        // parallelized sobel filter
        // Mat frame_sobel = sobel_parallelized(frame_gray);

        // write filtered frame
        // writer.write(frame_sobel);
        
        // Display the frame
        imshow("Display Window", frame_gray);
        // imshow("Display Window", frame_sobel);

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

void setupBarrier(int numThreads) {
    pthread_barrier_wait(&barrier, NULL, numThreads);
}

Mat grayscale_parallelized(Mat frame) {
    int total_h = frame.rows;
    int total_w = frame.cols;

    // calc quadrant dims
    int top_h = total_h / 2;
    int bot_h = total_h - top_h;
    int left_w = total_w / 2;
    int right_w = total_w - left_w;
    
    Mat frame_gray(total_h, total_w, CV_8UC1);
    Mat sobel_frame(total_h-2, total_w-2, CV_8UC1);  // act as Gx at first
    // set threadArgs for each quadrant
    threadArgs_t top_left_args = {&frame, &frame_gray, 0, 0, top_h, left_w};
    threadArgs_t top_right_args = {&frame, &frame_gray, 0, left_w, top_h, right_w};
    threadArgs_t bot_left_args = {&frame, &frame_gray, top_h, 0, bot_h, left_w};
    threadArgs_t bot_right_args = {&frame, &frame_gray, top_h, left_w, bot_h, right_w};

    threadArgs_t thread_args[] = {top_left_args, top_right_args, bot_left_args, bot_right_args};
    int pthread_create_ret_vals[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create_ret_vals[i] = pthread_create(&threads[i], NULL, to442_grayscale_worker, (void*)&thread_args[i]);
        if (pthread_create_ret_vals[i] != 0) {
            fprintf(stderr, "pthread_create #%d failed: %d\n", i, pthread_create_ret_vals[i]);
            abort();
        }
    }

    int pthread_join_ret_vals[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join_ret_vals[i] = pthread_join(threads[i], &thread_statuses[i]);
        if (pthread_join_ret_vals[i] != 0) {
            fprintf(stderr, "pthread_join #%d failed: %d\n", i, pthread_join_ret_vals[i]);
            abort();
        }
    }

    return frame_gray;
}

Mat sobel_parallelized(Mat frame) {
    Mat frame_sobel(12, 8, CV_8UC1, Scalar(0));  // dummy return to avoid compiler issues for now
    return frame_sobel;
}