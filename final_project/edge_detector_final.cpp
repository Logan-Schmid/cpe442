#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>

#include "vulkan_display.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace cv;
using namespace std;

// Helper to load SPIR-V files
vector<uint32_t> load_spirv(const string& filename)
{
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Failed to open SPIR-V file: " + filename);
    }
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, ios::beg);
    vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

void process_video_vulkan(const string& videoPath)
{
    VideoCapture cap;
    if (videoPath == "0") {
        cap.open(0);
    }
    else {
        cap.open(videoPath);
    }

    if (!cap.isOpened()) {
        throw runtime_error("Error: Could not open video.");
    }

    Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        throw runtime_error("Error: Video file is empty.");
    }

    const uint32_t width = static_cast<uint32_t>(firstFrame.cols);
    const uint32_t height = static_cast<uint32_t>(firstFrame.rows);

    if (width < 3 || height < 3) {
        throw runtime_error("Error: Input frame must be at least 3x3 for Sobel.");
    }

    const uint32_t outWidth = width - 2;
    const uint32_t outHeight = height - 2;

    cout << "Video initialized. Input: " << width << "x" << height
         << " | Output: " << outWidth << "x" << outHeight << endl;

    VulkanDisplay display(outWidth, outHeight);

    // Use the same Vulkan objects for Kompute so compute output stays on the same GPU/device.
    kp::Manager mgr(display.getInstance(), display.getPhysicalDevice(), display.getDevice());

    vector<uint32_t> inInit(width * height, 0);
    vector<uint32_t> outInit(outWidth * outHeight, 0);
    vector<uint32_t> dims = { width, height };

    auto tensorIn = mgr.tensorT<uint32_t>(inInit, kp::Memory::MemoryTypes::eDeviceAndHost);
    auto tensorOut = mgr.tensorT<uint32_t>(outInit, kp::Memory::MemoryTypes::eDevice);
    auto tensorDims = mgr.tensorT<uint32_t>(dims, kp::Memory::MemoryTypes::eDeviceAndHost);

    auto memIn = static_pointer_cast<kp::Memory>(tensorIn);
    auto memOut = static_pointer_cast<kp::Memory>(tensorOut);
    auto memDims = static_pointer_cast<kp::Memory>(tensorDims);

    vector<shared_ptr<kp::Memory>> params = {memIn, memOut, memDims};

    const vector<uint32_t> spirv = load_spirv("edge_detector.spv");
    const kp::Workgroup workgroups = { (outWidth + 15) / 16, (outHeight + 15) / 16, 1 };

    auto algorithm = mgr.algorithm(
        params,
        spirv,
        workgroups,
        vector<float>(),
        vector<float>());

    vector<shared_ptr<kp::Memory>> dimParam = {memDims};
    mgr.sequence()->record<kp::OpSyncDevice>(dimParam)->eval();

    auto seq = mgr.sequence();
    vector<shared_ptr<kp::Memory>> inParam = {memIn};
    seq->record<kp::OpSyncDevice>(inParam)
       ->record<kp::OpAlgoDispatch>(algorithm);

    auto outputBuffer = memOut->getPrimaryBuffer();
    if (!outputBuffer) {
        throw runtime_error("Failed to access output Vulkan buffer from Kompute tensor.");
    }

    Mat frame = firstFrame;
    Mat inputRGBA;

    cout << "Rendering with Vulkan swapchain. Close window or press Ctrl+C to exit." << endl;

    const int64 streamStart = getTickCount();
    int totalFramesProcessed = 0;
    const size_t inputBytes = static_cast<size_t>(width) * height * sizeof(uint32_t);

    while (true) {
        display.pollEvents();
        if (display.shouldClose()) {
            break;
        }

        if (frame.cols != static_cast<int>(width) || frame.rows != static_cast<int>(height)) {
            throw runtime_error("Error: Frame resolution changed during processing.");
        }

        cvtColor(frame, inputRGBA, COLOR_BGR2RGBA);
        if (!inputRGBA.isContinuous()) {
            inputRGBA = inputRGBA.clone();
        }

        memcpy(tensorIn->data(), inputRGBA.data, inputBytes);

        // Compute completes on-GPU, then the same GPU buffer is copied into swapchain image.
        seq->eval();
        display.presentFromBuffer(*outputBuffer, outWidth, outHeight);

        totalFramesProcessed++;

        cap >> frame;
        if (frame.empty()) {
            break;
        }
    }

    const double totalElapsed = (getTickCount() - streamStart) / getTickFrequency();
    if (totalElapsed > 0.0) {
        const double averageFps = totalFramesProcessed / totalElapsed;
        cout << "Average FPS: " << averageFps << endl;
    }

    cap.release();
}

int main(int argc, char** argv)
{
    string videoPath = "0";
    if (argc == 2) {
        videoPath = argv[1];
    }
    else if (argc > 2) {
        cerr << "Incorrect usage - use: edge_detector_final [video_path]" << endl;
        return EXIT_FAILURE;
    }

    try {
        process_video_vulkan(videoPath);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
