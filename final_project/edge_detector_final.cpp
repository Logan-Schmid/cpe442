#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <stdexcept>

using namespace cv;
using namespace std;

// Helper to load SPIR-V files
vector<uint32_t> load_spirv(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Failed to open SPIR-V file: " + filename);
    }
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<uint32_t> buffer(size / 4);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void process_video_vulkan(const string& videoPath) {
    VideoCapture cap;
    if (videoPath == "0") cap.open(0);
    else cap.open(videoPath);

    if (!cap.isOpened()) throw runtime_error("Error: Could not open video.");

    Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) throw runtime_error("Error: Video file is empty.");

    // 1. Calculate Input vs Output Dimensions
    uint32_t width = firstFrame.cols;
    uint32_t height = firstFrame.rows;

    if (width < 3 || height < 3) {
        throw runtime_error("Error: Input frame must be at least 3x3 for Sobel.");
    }
    
    // Unpadded (Valid) convolution shrinks the output by 1 pixel on every side (2 pixels total)
    uint32_t outWidth = width - 2;
    uint32_t outHeight = height - 2;
    
    cout << "Video initialized. Input: " << width << "x" << height 
         << " | Output: " << outWidth << "x" << outHeight << endl;

    // 2. Initialize Kompute Manager
    kp::Manager mgr; 

    // 3. Create Tensors
    // Use host-visible + coherent device memory on Pi to reduce staging overhead.
    constexpr int kBufferCount = 2;
    vector<uint32_t> inInit(width * height, 0);
    vector<float> outInit(outWidth * outHeight, 0.0f);
    vector<uint32_t> dims = { width, height };

    auto tensorDims = mgr.tensorT<uint32_t>(
        dims, kp::Memory::MemoryTypes::eDeviceAndHost);
    auto memDims = static_pointer_cast<kp::Memory>(tensorDims);

    vector<shared_ptr<kp::TensorT<uint32_t>>> tensorInSlots;
    vector<shared_ptr<kp::TensorT<float>>> tensorSobelSlots;
    vector<shared_ptr<kp::Algorithm>> algorithmSlots;
    vector<shared_ptr<kp::Sequence>> sequenceSlots;
    vector<bool> slotInFlight(kBufferCount, false);

    tensorInSlots.reserve(kBufferCount);
    tensorSobelSlots.reserve(kBufferCount);
    algorithmSlots.reserve(kBufferCount);
    sequenceSlots.reserve(kBufferCount);

    // 4. Load Compiled Shader
    vector<uint32_t> spirv = load_spirv("edge_detector.spv");

    // 5. Build Algorithm and Sequence
    // A. Define the workgroups FIRST
    kp::Workgroup workgroups = { (outWidth + 15) / 16, (outHeight + 15) / 16, 1 };

    // Sync dimensions to the GPU once
    vector<shared_ptr<kp::Memory>> dimParam = {memDims};
    mgr.sequence()->record<kp::OpSyncDevice>(dimParam)->eval();

    // Build double-buffered algorithms/sequences for async overlap
    for (int i = 0; i < kBufferCount; ++i) {
        auto tensorIn = mgr.tensorT<uint32_t>(
            inInit, kp::Memory::MemoryTypes::eDeviceAndHost);
        auto tensorSobel = mgr.tensorT<float>(
            outInit, kp::Memory::MemoryTypes::eDeviceAndHost);
        auto memIn = static_pointer_cast<kp::Memory>(tensorIn);
        auto memSobel = static_pointer_cast<kp::Memory>(tensorSobel);
        vector<shared_ptr<kp::Memory>> params = {memIn, memSobel, memDims};

        auto algorithm = mgr.algorithm(
            params, spirv, workgroups, std::vector<float>(), std::vector<float>());

        auto seq = mgr.sequence();
        vector<shared_ptr<kp::Memory>> inParam = {memIn};
        vector<shared_ptr<kp::Memory>> outParam = {memSobel};
        seq->record<kp::OpSyncDevice>(inParam)
           ->record<kp::OpAlgoDispatch>(algorithm)
           ->record<kp::OpSyncLocal>(outParam);

        tensorInSlots.push_back(tensorIn);
        tensorSobelSlots.push_back(tensorSobel);
        algorithmSlots.push_back(algorithm);
        sequenceSlots.push_back(seq);
    }

    // 6. Processing Loop
    Mat frame = firstFrame;
    Mat inputRGBA;
    cout << "Press 'ESC' or 'q' to exit..." << endl;

    // FPS Tracking Variables
    int64 streamStart = getTickCount();
    int totalFramesProcessed = 0;
    const size_t inputBytes = static_cast<size_t>(width) * height * sizeof(uint32_t);

    while (true) {
        if (frame.cols != static_cast<int>(width) || frame.rows != static_cast<int>(height)) {
            throw runtime_error("Error: Frame resolution changed during processing.");
        }

        const int slot = totalFramesProcessed % kBufferCount;
        if (slotInFlight[slot]) {
            sequenceSlots[slot]->evalAwait();
            slotInFlight[slot] = false;

            Mat result(outHeight, outWidth, CV_32FC1, tensorSobelSlots[slot]->data());
            imshow("GPU Processed Video (Unpadded)", result);
            char key = static_cast<char>(waitKey(1));
            if (key == 27 || key == 'q') {
                break;
            }
        }

        // Convert to RGBA
        cvtColor(frame, inputRGBA, COLOR_BGR2RGBA);
        if (!inputRGBA.isContinuous()) {
            inputRGBA = inputRGBA.clone();
        }

        // Copy this frame into the selected slot, then launch async GPU work.
        memcpy(tensorInSlots[slot]->data(), inputRGBA.data, inputBytes);
        sequenceSlots[slot]->evalAsync();
        slotInFlight[slot] = true;

        totalFramesProcessed++;

        cap >> frame;
        if (frame.empty()) break; 
    }

    for (int i = 0; i < kBufferCount; ++i) {
        if (slotInFlight[i]) {
            sequenceSlots[i]->evalAwait();
            slotInFlight[i] = false;

            Mat result(outHeight, outWidth, CV_32FC1, tensorSobelSlots[i]->data());
            imshow("GPU Processed Video (Unpadded)", result);
            char key = static_cast<char>(waitKey(1));
            if (key == 27 || key == 'q') {
                break;
            }
        }
    }

    double totalElapsed = (getTickCount() - streamStart) / getTickFrequency();
    if (totalElapsed > 0.0) {
        double averageFps = totalFramesProcessed / totalElapsed;
        cout << "Average FPS: " << averageFps << endl;
    }

    cap.release();
    destroyAllWindows();
}

int main(int argc, char** argv) {
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
