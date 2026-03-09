#include <opencv2/opencv.hpp>
#include <kompute/Kompute.hpp>
#include <vector>
#include <fstream>

// Helper to load SPIR-V files
std::vector<uint32_t> load_spirv(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / 4);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void process_image_vulkan(cv::Mat& inputBGR) {
    // 1. Prepare Data: GPU prefers 4-channel RGBA for alignment
    cv::Mat inputRGBA;
    cv::cvtColor(inputBGR, inputRGBA, cv::COLOR_BGR2RGBA);
    
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

    std::vector<std::shared_ptr<kp::Tensor>> paramsGray = {tensorIn, tensorGray};
    std::vector<std::shared_ptr<kp::Tensor>> paramsSobel = {tensorGray, tensorSobel};

    // 4. Load Compiled Shaders
    std::vector<uint32_t> graySpirv = load_spirv("grayscale.spv");
    std::vector<uint32_t> sobelSpirv = load_spirv("sobel.spv");

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
    cv::Mat result(height, width, CV_8UC1, tensorSobel->data());
    cv::imshow("GPU Processed Edges", result);
    cv::waitKey(0);
}