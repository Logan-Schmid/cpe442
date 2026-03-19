#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace cv;
using namespace std;

// Helper to load SPIR-V files
vector<uint32_t> load_spirv(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) throw runtime_error("Failed to open SPIR-V file: " + filename);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, ios::beg);
    vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

void process_video_vulkan(const string& videoPath) {
    auto start = chrono::high_resolution_clock::now();
    // 1. Open video
    VideoCapture cap;
    cap.open(videoPath);
    if (!cap.isOpened()) {
        throw runtime_error("Error: Could not open video.");
    }

    const uint32_t width = static_cast<uint32_t>(cap.get(CAP_PROP_FRAME_WIDTH));
    const uint32_t height = static_cast<uint32_t>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Video initialized. Input: " << width << "x" << height << " | Total Frames: " << total_frames << endl;
    
    // 2. Initialize the Vulkan Compute Manager
    kp::Manager mgr;

    // Create the Dimensions Tensor (NEW)
    // We sync this to the device immediately since it never changes
    vector<uint32_t> dims = { (uint32_t)width, (uint32_t)height };
    auto tensorDims = mgr.tensorT<uint32_t>(dims);
    // Sync to device using OpSyncDevice
    vector<shared_ptr<kp::Memory>> syncDims = { tensorDims };
    mgr.sequence()->record<kp::OpSyncDevice>(syncDims)->eval();

    // 3. Allocate Host-Visible Tensors (UPDATED FOR FLOATS)
    // tensorIn is still 4 bytes (RGBA bytes packed into uint32_t)
    vector<uint32_t> inData(width * height, 0);
    auto tensorIn = mgr.tensorT<uint32_t>(inData);
    
    // Gray and Sobel are now arrays of 32-bit FLOATS
    vector<float> grayData(width * height, 0.0f);
    auto tensorGray = mgr.tensorT<float>(grayData);

    vector<float> sobelData(width * height, 0.0f);
    auto tensorSobel = mgr.tensorT<float>(sobelData);


    // 4. Wrap OpenCV Mats directly around the Kompute GPU memory (Zero-Copy) (UPDATED FOR FLOATS)
    Mat gpuMappedRGBA(height, width, CV_8UC4, tensorIn->data());
    
    // Tell OpenCV the final output is a 32-bit float (CV_32FC1)
    Mat gpuMappedSobel(height, width, CV_32FC1, tensorSobel->data());

    // 5. Load Shaders and Create Algorithms
    auto graySpirv = load_spirv("grayscale.spv");
    auto sobelSpirv = load_spirv("sobel.spv");

// Calculate workgroups here so they can be passed into the algorithm
    kp::Workgroup workgroups = { (uint32_t)ceil(width / 16.0), (uint32_t)ceil(height / 16.0), 1 };

    // Explicitly declare Memory vectors to satisfy the new template typings
    vector<shared_ptr<kp::Memory>> paramsGray = {tensorIn, tensorGray, tensorDims};
    auto algoGray = mgr.algorithm(paramsGray, graySpirv, workgroups);

    vector<shared_ptr<kp::Memory>> paramsSobel = {tensorGray, tensorSobel, tensorDims};
    auto algoSobel = mgr.algorithm(paramsSobel, sobelSpirv, workgroups);

    // 6. Pre-record the execution sequence
    auto sequence = mgr.sequence();
    sequence->record<kp::OpSyncDevice>({tensorIn})           // Step 1: Flush OpenCV frame to GPU
            ->record<kp::OpAlgoDispatch>(algoGray)           // Step 2: Grayscale pass
            ->record<kp::OpAlgoDispatch>(algoSobel)          // Step 3: Sobel pass
            ->record<kp::OpSyncLocal>({tensorSobel});        // Step 4: Fetch result back to OpenCV

    Mat cpuFrame;
    cout << "Starting GPU-accelerated video loop. Press ESC to exit.\n";

    int frame_count = 0;
    // 7. The Main Video Loop
    while (frame_count < total_frames) {
        cap.read(cpuFrame);
        if (cpuFrame.empty()) break;

        // Convert the BGR camera frame to RGBA, writing it directly into the GPU's memory space
        cvtColor(cpuFrame, gpuMappedRGBA, COLOR_BGR2RGBA);

        // Tell the GPU to execute the pre-recorded sequence (Gray -> Sobel)
        sequence->eval();

        // The result is instantly available in gpuMappedSobel
        imshow("GPU Accelerated Sobel", gpuMappedSobel);
	    frame_count++;
        if (cv::waitKey(1) == 27) break; // ESC key
    }
    
    cout << "Loop finished. Calculating FPS..." << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = hrono::duration_cast<chrono::milliseconds>(stop - start);
    float duration_secs = (float)duration.count()/1000;
    cout << "Averate FPS: " << frame_count / duration_secs << endl;

    cout << "Destroying window..." << endl;
    cap.release();
    cout << "Destroying window..." << endl;
    destroyAllWindows();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        cerr << "Incorrect usage - use: edge_detector_final [video_path]" << endl;
        return EXIT_FAILURE;
    }
    string videoPath = argv[1];

    try {
        process_video_vulkan(videoPath);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
