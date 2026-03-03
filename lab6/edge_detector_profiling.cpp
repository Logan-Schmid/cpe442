/*********************************************************
* File: edge_detector_profiling.cpp
*
* Description: CPE 442 Lab 6 - Grayscale and apply edge
* detection on video using threads and vector instructions
* (via C instrinsics) to speed up processing. Measuring
* performance counters to see where to further optimize.
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

extern "C" {
	#include <papi.h>
}

#define NUM_THREADS 4
#define TOT_EVENTS 6

using namespace cv;
using namespace std;

// initialize threads for parallelization of image processing
pthread_t threads[NUM_THREADS];
pthread_barrier_t barrier;
void *thread_statuses[NUM_THREADS];
bool frames_remaining = true;

/*-----------------------------------------------------
* Function: handle_papi_error
*
* Description: Handle papi error by printing error and
* exiting code with error 1
*
* param retval: int: return value of papi command
*--------------------------------------------------------*/ 
void handle_papi_error(int retval) {
    if (retval != PAPI_OK) {
        cerr << "PAPI error: " << PAPI_strerror(retval) << endl;
        exit(1);
    }
}

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

    // Pin thread to core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(args->thread_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

	int EventSet = PAPI_NULL;
	long long values[TOT_EVENTS]; // holds event counter results

	// Create the Event Set
	handle_papi_error(PAPI_create_eventset(&EventSet));

	// Add all events of interest to the Event Set
    handle_papi_error(PAPI_add_event(EventSet, PAPI_L1_DCM));
    handle_papi_error(PAPI_add_event(EventSet, PAPI_L1_ICM));
    handle_papi_error(PAPI_add_event(EventSet, PAPI_L2_DCM));
    handle_papi_error(PAPI_add_event(EventSet, PAPI_TOT_CYC));
    handle_papi_error(PAPI_add_event(EventSet, PAPI_BR_MSP));
    handle_papi_error(PAPI_add_event(EventSet, PAPI_TOT_INS));

	// Start counting events in the Event Set
    handle_papi_error(PAPI_start(EventSet));

    while (frames_remaining) {
        pthread_barrier_wait(&barrier); // wait for main thread to load the current frame
        to442_grayscale(args->src, args->gray, args->row_0, args->col_0, args->h, args->w);
        pthread_barrier_wait(&barrier); // wait for all other worker threads to be done grayscaling
        to442_sobel(args->gray, args->sobel, args->row_0, args->col_0, args->h, args->w);
        pthread_barrier_wait(&barrier); // wait for all other work threads to be done applying sobel
    }

    // Stop the counting of events in the Event Set
    handle_papi_error(PAPI_stop(EventSet, values));

    // store counter values into threadargs to read back in main
    args->l1_data_cache_misses = values[0];
    args->l1_instr_cache_misses = values[1];
    args->l2_data_cache_misses = values[2];
    args->tot_cycles = values[3];
    args->branch_mispredicts = values[4];
    args->tot_intructions = values[5];

    PAPI_cleanup_eventset(EventSet);
    PAPI_destory_eventset(&EventSet);
    return NULL;
}

int main(int argc, char** argv) {
    auto start = chrono::high_resolution_clock::now(); // start timer for runtime
    if (argc != 2) {
        cerr << "Incorrect usage - use via: 'edge_detector [video_path]'" << endl;
        return -1;
    }
    
	// Initialize the PAPI library
	int retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
        handle_papi_error(retval);
	}

    // Initialize thread support
    handle_papi_error(PAPI_thread_init(pthread_self));

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
    threadArgs_t row_0_args = {&frame, &frame_gray, &frame_sobel, 0,         0, sub_h+2, width, 0, 0, 0, 0, 0, 0, 0};
    threadArgs_t row_1_args = {&frame, &frame_gray, &frame_sobel, sub_h-1,   0, sub_h+2, width, 1, 0, 0, 0, 0, 0, 0};
    threadArgs_t row_2_args = {&frame, &frame_gray, &frame_sobel, 2*sub_h-1, 0, sub_h+2, width, 2, 0, 0, 0, 0, 0, 0};
    threadArgs_t row_3_args = {&frame, &frame_gray, &frame_sobel, 3*sub_h-1, 0, sub_h+1, width, 3, 0, 0, 0, 0, 0, 0};

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
    frames_remaining = false;
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
    if (getWindowProperty("Display Window", WND_PROP_VISIBLE) >= 0) {
	destroyWindow("Display Window");
    }
    // destoryAllWindows();

    // calculate and print the runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    float duration_secs = (float)duration.count()/1000;
    cout << "Program Runtime: " << duration_secs << " seconds\n";  // e.g., 150000 us [web:2]
    cout << "Average FPS: " << frame_count / duration_secs << endl;
    
    // Calculate and print the average events counted for each core
    for (int i = 0; i < NUM_THREADS < i++) {
        cout << "Core " << i << " Avg L1 Data Cache Misses Per Frame: " << thread_args[i].l1_data_cache_misses / frame_count << endl;
        cout << "Core " << i << " Avg L1 Instruction Cache Misses Per Frame: " << thread_args[i].l1_instruction_cache_misses / frame_count << endl;
        cout << "Core " << i << " Avg L2 Data Cache Misses Per Frame: " << thread_args[i].l2_data_cache_misses / frame_count << endl;
        cout << "Core " << i << " Avg Cycles Per Frame: " << thread_args[i].cycles / frame_count << endl;
        cout << "Core " << i << " Avg Branch Mispredictions Per Frame: " << thread_args[i].branch_mispredicts / frame_count << endl;
        cout << "Core " << i << " Avg Instructions Per Frame: " << thread_args[i].tot_intructions / frame_count << endl;
        cout << endl;
    }

    return 0;
}
