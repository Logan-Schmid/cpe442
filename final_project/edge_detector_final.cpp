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

    // 3. Allocate Host-Visible GPU Memory
    auto tensorIn = mgr.tensor(nullptr, width * height * 4, sizeof(uint8_t), kp::Tensor::TensorDataTypes::eUnsignedInt);  //need 4 channels (RGBA) for the frame input
    auto tensorGray = mgr.tensor(nullptr, width * height, sizeof(uint8_t), kp::Tensor::TensorDataTypes::eUnsignedInt);
    auto tensorSobel = mgr.tensor(nullptr, width * height, sizeof(uint8_t), kp::Tensor::TensorDataTypes::eUnsignedInt);

    // 4. Wrap OpenCV Mats directly around the Kompute GPU memory (Zero-Copy)
    Mat gpuMappedRGBA(height, width, CV_8UC4, tensorIn->data());
    Mat gpuMappedSobel(height, width, CV_8UC1, tensorSobel->data());

    // 5. Load Shaders and Create Algorithms
    auto graySpirv = load_spirv("grayscale.spv");
    auto sobelSpirv = load_spirv("sobel.spv");

    auto algoGray = mgr.algorithm({tensorIn, tensorGray}, graySpirv);
    auto algoSobel = mgr.algorithm({tensorGray, tensorSobel}, sobelSpirv);

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
