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

    cout << "Video initialized. Input: " << width << "x" << height << endl;
    
    // 2. Initialize the Vulkan Compute Manager
    kp::Manager mgr;

    // Create the Dimensions Tensor (NEW)
    // We sync this to the device immediately since it never changes
    vector<uint32_t> dims = { (uint32_t)width, (uint32_t)height };
    auto tensorDims = mgr.tensor(dims.data(), dims.size(), sizeof(uint32_t), kp::Tensor::TensorDataTypes::eUnsignedInt);
    mgr.sequence()->record<kp::OpSyncDevice>({tensorDims})->eval();

    // 3. Allocate Host-Visible Tensors (UPDATED FOR FLOATS)
    // tensorIn is still 4 bytes (RGBA bytes packed into uint32_t)
    auto tensorIn = mgr.tensor(nullptr, width * height, sizeof(uint32_t), kp::Tensor::TensorDataTypes::eUnsignedInt);
    
    // Gray and Sobel are now arrays of 32-bit FLOATS
    auto tensorGray = mgr.tensor(nullptr, width * height, sizeof(float), kp::Tensor::TensorDataTypes::eFloat);
    auto tensorSobel = mgr.tensor(nullptr, width * height, sizeof(float), kp::Tensor::TensorDataTypes::eFloat);

    // 4. Wrap OpenCV Mats directly around the Kompute GPU memory (Zero-Copy) (UPDATED FOR FLOATS)
    Mat gpuMappedRGBA(height, width, CV_8UC4, tensorIn->data());
    
    // Tell OpenCV the final output is a 32-bit float (CV_32FC1)
    Mat gpuMappedSobel(height, width, CV_32FC1, tensorSobel->data());

    // 5. Load Shaders and Create Algorithms
    auto graySpirv = load_spirv("grayscale.spv");
    auto sobelSpirv = load_spirv("sobel.spv");

    auto algoGray = mgr.algorithm({tensorIn, tensorGray, tensorDims}, graySpirv);
    auto algoSobel = mgr.algorithm({tensorGray, tensorSobel, tensorDims}, sobelSpirv);

    // Calculate how many 16x16 workgroups are needed to cover the image
    kp::Workgroup workgroups = { (uint32_t)ceil(width / 16.0), (uint32_t)ceil(height / 16.0), 1 };

    // 6. Pre-record the execution sequence
    auto sequence = mgr.sequence();
    sequence->record<kp::OpAlgoDispatch>(algoGray, workgroups)   // Step A: Grayscale the whole frame
            ->record<kp::OpAlgoDispatch>(algoSobel, workgroups); // Step B: Sobel the grayscale frame

    Mat cpuFrame;
    cout << "Starting GPU-accelerated video loop. Press ESC to exit.\n";

    // 7. The Main Video Loop
    while (true) {
        cap.read(cpuFrame);
        if (cpuFrame.empty()) break;

        // Convert the BGR camera frame to RGBA, writing it directly into the GPU's memory space
        cv::cvtColor(cpuFrame, gpuMappedRGBA, cv::COLOR_BGR2RGBA);

        // Tell the GPU to execute the pre-recorded sequence (Gray -> Sobel)
        sequence->eval();

        // The result is instantly available in gpuMappedSobel
        cv::imshow("GPU Accelerated Sobel", gpuMappedSobel);

        if (cv::waitKey(1) == 27) break; // ESC key
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    float duration_secs = (float)duration.count()/1000;
    cout << "Averate FPS: " << frame_count / duration_secs << endl;

    cap.release();
    cv::destroyAllWindows();
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
