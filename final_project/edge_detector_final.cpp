#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>
#include <vector>
#include <fstream>
#include <iostream>

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
    
    // Unpadded (Valid) convolution shrinks the output by 1 pixel on every side (2 pixels total)
    uint32_t outWidth = width - 2;
    uint32_t outHeight = height - 2;
    
    cout << "Video initialized. Input: " << width << "x" << height 
         << " | Output: " << outWidth << "x" << outHeight << endl;

    // 2. Initialize Kompute Manager
    kp::Manager mgr; 

    // 3. Create Tensors
    // Initialize host vectors with zeros to allocate the correct amount of space
    vector<uint32_t> inBuffer(width * height, 0);
    auto tensorIn = mgr.tensorT<uint32_t>(inBuffer);
    
    // The output tensor is strictly sized to the new, smaller dimensions
    vector<float> outBuffer(outWidth * outHeight, 0.0f);
    auto tensorSobel = mgr.tensorT<float>(outBuffer);
    
    // We pass the INPUT dimensions to the shader
    vector<uint32_t> dims = { width, height };
    auto tensorDims = mgr.tensorT<uint32_t>(dims);

    // Group them for the algorithm (TensorT safely upcasts to Tensor)
    vector<shared_ptr<kp::Tensor>> params = {tensorIn, tensorSobel, tensorDims};

    // 4. Load Compiled Shader
    vector<uint32_t> spirv = load_spirv("edge_detector.spv");

    // 5. Build Algorithm and Sequence
    // A. Define the workgroups FIRST
    kp::Workgroup workgroups = { (uint32_t)ceil(outWidth / 16.0), (uint32_t)ceil(outHeight / 16.0), 1 };

    // B. Pass the workgroups INTO the algorithm initialization
    auto algorithm = mgr.algorithm(params, spirv, workgroups, std::vector<float>(), std::vector<float>());

    // Sync dimensions to the GPU once
    mgr.sequence()->record<kp::OpTensorSyncDevice>({tensorDims})->eval();

    // Pre-record the processing loop sequence
    auto seq = mgr.sequence();
    seq->record<kp::OpTensorSyncDevice>({tensorIn})
       // C. The workgroup is now baked into the algorithm, so OpAlgoDispatch only needs the algorithm
       ->record<kp::OpAlgoDispatch>(algorithm) 
       ->record<kp::OpTensorSyncLocal>({tensorSobel});

    // 6. Processing Loop
    Mat frame = firstFrame;
    Mat inputRGBA;
    cout << "Press 'ESC' or 'q' to exit..." << endl;

    // FPS Tracking Variables
    int64 timerStart = getTickCount();
    int framesProcessed = 0;
    double currentFps = 0.0;

    while (true) {
        // Convert to RGBA
        cvtColor(frame, inputRGBA, COLOR_BGR2RGBA);

        // Copy the full input frame into tensorIn
        memcpy(tensorIn->data(), inputRGBA.data, width * height * 4);

        // Execute GPU workload
        seq->eval();

        // Wrap the output tensor data in a Mat sized exactly to outWidth and outHeight
        Mat result(outHeight, outWidth, CV_32FC1, tensorSobel->data());

        // FPS Calculation and Overlay
        framesProcessed++;
        double timeElapsed = (getTickCount() - timerStart) / getTickFrequency();
        
        // Update the FPS counter every 1.0 seconds
        if (timeElapsed >= 1.0) {
            currentFps = framesProcessed / timeElapsed;
            framesProcessed = 0;
            timerStart = getTickCount();
        }
        
        // Draw the FPS on the top-left corner of the frame
        // Using Scalar(1.0) because our image pixels are floats from 0.0 to 1.0
        string fpsText = "FPS: " + to_string((int)currentFps);
        putText(result, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(1.0), 2);

        imshow("GPU Processed Video (Unpadded)", result);

        char key = (char)waitKey(1);
        if (key == 27 || key == 'q') break;

        cap >> frame;
        if (frame.empty()) break; 
    }

    cap.release();
    destroyAllWindows();
}

int main(int argc, char** argv) {
    string videoPath = "0"; 
    if (argc == 2) {
        videoPath = argv[1];
    }
    else {
        cerr << "Incorrect usage - use via: 'edge_detector_final [video_path]" << endl;
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