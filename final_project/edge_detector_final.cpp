#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

// Helper to load SPIR-V files
vector<uint32_t> load_spirv(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<uint32_t> buffer(size / 4);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void process_image_vulkan(Mat& inputBGR) {
    // 1. Prepare Data: GPU prefers 4-channel RGBA for alignment
    Mat inputRGBA;
    cvtColor(inputBGR, inputRGBA, COLOR_BGR2RGBA);
    
    int width = inputRGBA.cols;
    int height = inputRGBA.rows;

    // 2. Initialize Kompute Manager (selects the Pi 4 VideoCore VI)
    kp::Manager mgr; 

    // 3. Create Tensors
    // Tensor 0: Input Image (RGBA)
    // Tensor 1: Grayscale Output (1-channel)
    // Tensor 2: Sobel Output (1-channel)
    auto tensorIn = mgr.tensor(inputRGBA.data, width * height * 4, sizeof(uchar), kp::Tensor::TensorDataTypes::eUnsignedInt);
    auto tensorGray = mgr.tensor(nullptr, width * height, sizeof(uchar), kp::Tensor::TensorDataTypes::eUnsignedInt);
    auto tensorSobel = mgr.tensor(nullptr, width * height, sizeof(uchar), kp::Tensor::TensorDataTypes::eUnsignedInt);

    vector<shared_ptr<kp::Tensor>> paramsGray = {tensorIn, tensorGray};
    vector<shared_ptr<kp::Tensor>> paramsSobel = {tensorGray, tensorSobel};

    // 4. Load Compiled Shaders
    vector<uint32_t> graySpirv = load_spirv("grayscale.spv");
    vector<uint32_t> sobelSpirv = load_spirv("sobel.spv");

    // 5. Build and Run Sequence
    // We define a sequence: Sync data to GPU -> Run Gray -> Run Sobel -> Sync back
    auto algorithmGray = mgr.algorithm(paramsGray, graySpirv);
    auto algorithmSobel = mgr.algorithm(paramsSobel, sobelSpirv);

    // Calculate workgroups (we used 16x16 in the shader)
    kp::Workgroup workgroups = { (uint32_t)ceil(width / 16.0), (uint32_t)ceil(height / 16.0), 1 };

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>({tensorIn}) // Upload to GPU
        ->record<kp::OpAlgoDispatch>(algorithmGray, workgroups)
        ->record<kp::OpAlgoDispatch>(algorithmSobel, workgroups)
        ->record<kp::OpTensorSyncLocal>({tensorSobel}) // Download result
        ->eval();

    // 6. Map result back to OpenCV
    Mat result(height, width, CV_8UC1, tensorSobel->data());
    imshow("GPU Processed Edges", result);
    waitKey(0);
}