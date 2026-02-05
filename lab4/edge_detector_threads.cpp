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
#include <chrono>
#include "processing.hpp"

#define NUM_THREADS 4

using namespace cv;
using namespace std;

// initialize threads for parallelization of image processing
pthread_t threads[NUM_THREADS];
pthread_barrier_t barrier;
void *thread_statuses[NUM_THREADS];
bool keep_working = true;

/*-----------------------------------------------------
* Function: process_quadrant
*
* Description: Used with a child thread to grayscale and then 
* apply a Sobel filter to an image quadrant
*
* param threadArgs: void*: pointer to the struct with args for the worker
*
* return: void*
*--------------------------------------------------------*/ 
void* process_quadrant(void* threadArgs) {
    threadArgs_t* args = static_cast<threadArgs_t*>(threadArgs);

    while (keep_working) {
        pthread_barrier_wait(&barrier); // wait for main thread to load the current frame
        to442_grayscale(args->src, args->gray, args->row_0, args->col_0, args->h, args->w);
        pthread_barrier_wait(&barrier); // wait for all other worker threads to be done grayscaling
        to442_sobel(args->gray, args->sobel, args->row_0, args->col_0, args->h, args->w);
        pthread_barrier_wait(&barrier); // wait for all other work threads to be done applying sobel
    }
    return NULL;
}

int main(int argc, char** argv) {
    auto start = chrono::high_resolution_clock::now(); // start timer for runtime

    if (argc != 2) {
        cerr << "Incorrect usage - use via: 'edge_detector [video_path]'" << endl;
        return -1;
    }

    // Initialize video reader
    string cap_path = argv[1];
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
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));    
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    cout << "Total frames: " << frame_count << ", FPS: " << fps << endl;
    cout << "Width: " << width << ", Height: " << height << endl;

    // define Mats for each processing stage to be accessed by threads
    Mat frame;
    Mat frame_gray(height, width, CV_8UC1);
    Mat frame_sobel(height-2, width-2, CV_8UC1);  // act as Gx at first
    
    // init thread attr and barrier
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_barrier_init(&barrier, NULL, NUM_THREADS+1);

    // calc quadrant dims, assuming even dims
    int sub_h = height / 4;
    
    // set threadArgs for each quadrant
    threadArgs_t row_0_args = {&frame, &frame_gray, &frame_sobel, 0,         0, sub_h+2, width, 0};
    threadArgs_t row_1_args = {&frame, &frame_gray, &frame_sobel, sub_h-1,   0, sub_h+2, width, 1};
    threadArgs_t row_2_args = {&frame, &frame_gray, &frame_sobel, 2*sub_h-1, 0, sub_h+2, width, 2};
    threadArgs_t row_3_args = {&frame, &frame_gray, &frame_sobel, 3*sub_h-1, 0, sub_h+1, width, 3};

    threadArgs_t thread_args[] = {row_0_args, row_1_args, row_2_args, row_3_args};
    int pthread_create_ret_vals[NUM_THREADS];

    // start each child thread and check creation return values
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create_ret_vals[i] = pthread_create(&threads[i], NULL, process_quadrant, (void*)&thread_args[i]);
        if (pthread_create_ret_vals[i] != 0) {
            fprintf(stderr, "pthread_create #%d failed: %d\n", i, pthread_create_ret_vals[i]);
            abort();
        }
    }

    // Read and display each frame of the video
    for (int i = 0; i < frame_count; i++) {
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "Error occurred in reading a frame." << endl;
            return 1;
        }
        // barrier to prevent worker threads from processing until the new frame is loaded
        pthread_barrier_wait(&barrier);

        // wait for worker threads to grayscale current frame
        pthread_barrier_wait(&barrier);

        // wait for worker threads to sobel filter current frame
        pthread_barrier_wait(&barrier);
        
        // Display the frame
        imshow("Display Window", frame_sobel);

        // Wait for appropriate time between frames and check if 'q' is pressed to exit
        char key = waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    // Stop all worker threads once all frames have been processed
    // Need three pthread_barrier_waits due to the same number in process_quadrant
    keep_working = false;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);

    // Join threads (cleanup)
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_barrier_destroy(&barrier);

    // Release the video capture object and close any OpenCV windows
    cap.release();
    destroyAllWindows();

    // calculate and print the runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << "Program Runtime: " << (float)duration.count()/1000 << " seconds\n";  // e.g., 150000 us [web:2]

    return 0;
}