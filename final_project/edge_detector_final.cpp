#include <opencv2/opencv.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <kompute/Kompute.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

class VulkanDisplay
{
public:
    VulkanDisplay(uint32_t width, uint32_t height)
        : mWindowWidth(width), mWindowHeight(height)
    {
        try {
            initWindow();
            initVulkan();
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    ~VulkanDisplay()
    {
        cleanup();
    }

    shared_ptr<vk::Instance> getInstance() const { return mInstance; }
    shared_ptr<vk::PhysicalDevice> getPhysicalDevice() const { return mPhysicalDevice; }
    shared_ptr<vk::Device> getDevice() const { return mDevice; }

    bool shouldClose() const
    {
        return mWindow && glfwWindowShouldClose(mWindow);
    }

    void pollEvents() const
    {
        glfwPollEvents();
    }

    void presentFromBuffer(const vk::Buffer& sourceBuffer, uint32_t sourceWidth, uint32_t sourceHeight)
    {
        const array<vk::Fence, 1> inFlightFences = { mInFlightFence };
        mDevice->waitForFences(inFlightFences, VK_TRUE, numeric_limits<uint64_t>::max());
        mDevice->resetFences(inFlightFences);

        uint32_t imageIndex = 0;
        const vk::Result acquireResult = mDevice->acquireNextImageKHR(
            mSwapchain,
            numeric_limits<uint64_t>::max(),
            mImageAvailableSemaphore,
            vk::Fence(),
            &imageIndex);

        if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
            return;
        }
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) {
            throw runtime_error("Failed to acquire swapchain image.");
        }

        vk::CommandBuffer cmd = mCommandBuffers.at(imageIndex);
        cmd.reset(vk::CommandBufferResetFlags());

        const vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(beginInfo);

        vk::BufferMemoryBarrier bufferBarrier{};
        bufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        bufferBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.buffer = sourceBuffer;
        bufferBarrier.offset = 0;
        bufferBarrier.size = VK_WHOLE_SIZE;

        vk::ImageMemoryBarrier toTransferBarrier{};
        toTransferBarrier.srcAccessMask = vk::AccessFlags();
        toTransferBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        toTransferBarrier.oldLayout = vk::ImageLayout::eUndefined;
        toTransferBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.image = mSwapchainImages.at(imageIndex);
        toTransferBarrier.subresourceRange = vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

        const array<vk::MemoryBarrier, 0> noMemoryBarriers{};
        const array<vk::BufferMemoryBarrier, 1> bufferBarriers{bufferBarrier};
        const array<vk::ImageMemoryBarrier, 1> toTransferBarriers{toTransferBarrier};
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags(),
            noMemoryBarriers,
            bufferBarriers,
            toTransferBarriers);

        const vk::ClearColorValue clearColor(array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        const vk::ImageSubresourceRange clearRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        cmd.clearColorImage(
            mSwapchainImages.at(imageIndex),
            vk::ImageLayout::eTransferDstOptimal,
            clearColor,
            clearRange);

        const vk::Extent3D copyExtent(
            min(sourceWidth, mSwapchainExtent.width),
            min(sourceHeight, mSwapchainExtent.height),
            1);

        vk::BufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource = vk::ImageSubresourceLayers(
            vk::ImageAspectFlagBits::eColor,
            0,
            0,
            1);
        copyRegion.imageOffset = vk::Offset3D(0, 0, 0);
        copyRegion.imageExtent = copyExtent;

        cmd.copyBufferToImage(
            sourceBuffer,
            mSwapchainImages.at(imageIndex),
            vk::ImageLayout::eTransferDstOptimal,
            copyRegion);

        vk::ImageMemoryBarrier toPresentBarrier{};
        toPresentBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        toPresentBarrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        toPresentBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        toPresentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        toPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toPresentBarrier.image = mSwapchainImages.at(imageIndex);
        toPresentBarrier.subresourceRange = vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

        const array<vk::BufferMemoryBarrier, 0> noBufferBarriers{};
        const array<vk::ImageMemoryBarrier, 1> toPresentBarriers{toPresentBarrier};
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::DependencyFlags(),
            noMemoryBarriers,
            noBufferBarriers,
            toPresentBarriers);

        cmd.end();

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &mImageAvailableSemaphore;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &mRenderFinishedSemaphore;

        mQueue.submit(submitInfo, mInFlightFence);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &mRenderFinishedSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &mSwapchain;
        presentInfo.pImageIndices = &imageIndex;

        const vk::Result presentResult = mQueue.presentKHR(presentInfo);
        if (presentResult != vk::Result::eSuccess &&
            presentResult != vk::Result::eSuboptimalKHR &&
            presentResult != vk::Result::eErrorOutOfDateKHR) {
            throw runtime_error("Failed to present swapchain image.");
        }
    }

private:
    static vk::SurfaceFormatKHR chooseSurfaceFormat(const vector<vk::SurfaceFormatKHR>& formats)
    {
        for (const auto& format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Unorm &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }
        return formats.at(0);
    }

    static vk::PresentModeKHR choosePresentMode(const vector<vk::PresentModeKHR>& modes)
    {
        for (const auto& mode : modes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                return mode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    static vk::Extent2D chooseExtent(
        const vk::SurfaceCapabilitiesKHR& capabilities,
        uint32_t requestedWidth,
        uint32_t requestedHeight)
    {
        if (capabilities.currentExtent.width != numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        vk::Extent2D extent{};
        extent.width = clamp(
            requestedWidth,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        extent.height = clamp(
            requestedHeight,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
        return extent;
    }

    void initWindow()
    {
        if (!glfwInit()) {
            throw runtime_error("Failed to initialize GLFW.");
        }
        if (!glfwVulkanSupported()) {
            throw runtime_error("GLFW reports Vulkan is not supported.");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        mWindow = glfwCreateWindow(
            static_cast<int>(mWindowWidth),
            static_cast<int>(mWindowHeight),
            "GPU Processed Video (Vulkan)",
            nullptr,
            nullptr);

        if (!mWindow) {
            throw runtime_error("Failed to create GLFW window.");
        }
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDeviceAndQueueFamily();
        createDeviceAndQueue();
        createSwapchain();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0) {
            throw runtime_error("Failed to query required GLFW Vulkan instance extensions.");
        }

        vector<const char*> instanceExtensions(
            glfwExtensions,
            glfwExtensions + glfwExtensionCount);

        const vk::ApplicationInfo appInfo(
            "cpe442-edge-display",
            1,
            "cpe442",
            1,
            VK_API_VERSION_1_0);

        const vk::InstanceCreateInfo instanceInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0,
            nullptr,
            static_cast<uint32_t>(instanceExtensions.size()),
            instanceExtensions.data());

        mInstance = make_shared<vk::Instance>(vk::createInstance(instanceInfo));
    }

    void createSurface()
    {
        VkSurfaceKHR cSurface = VK_NULL_HANDLE;
        const VkResult result = glfwCreateWindowSurface(
            static_cast<VkInstance>(*mInstance),
            mWindow,
            nullptr,
            &cSurface);

        if (result != VK_SUCCESS) {
            throw runtime_error("Failed to create Vulkan window surface.");
        }

        mSurface = cSurface;
    }

    void pickPhysicalDeviceAndQueueFamily()
    {
        const vector<vk::PhysicalDevice> physicalDevices = mInstance->enumeratePhysicalDevices();
        if (physicalDevices.empty()) {
            throw runtime_error("No Vulkan physical devices found.");
        }

        for (const auto& candidate : physicalDevices) {
            const vector<vk::ExtensionProperties> deviceExtensions =
                candidate.enumerateDeviceExtensionProperties();

            bool hasSwapchainExtension = false;
            for (const auto& ext : deviceExtensions) {
                if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                    hasSwapchainExtension = true;
                    break;
                }
            }
            if (!hasSwapchainExtension) {
                continue;
            }

            const vector<vk::QueueFamilyProperties> queueFamilies =
                candidate.getQueueFamilyProperties();

            for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
                const bool hasCompute =
                    static_cast<bool>(queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute);
                const bool hasGraphics =
                    static_cast<bool>(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics);
                const bool supportsPresent = candidate.getSurfaceSupportKHR(i, mSurface);

                if (!hasCompute || !hasGraphics || !supportsPresent) {
                    continue;
                }

                const vector<vk::SurfaceFormatKHR> formats =
                    candidate.getSurfaceFormatsKHR(mSurface);
                const vector<vk::PresentModeKHR> modes =
                    candidate.getSurfacePresentModesKHR(mSurface);

                if (formats.empty() || modes.empty()) {
                    continue;
                }

                mPhysicalDevice = make_shared<vk::PhysicalDevice>(candidate);
                mQueueFamilyIndex = i;
                return;
            }
        }

        throw runtime_error("Failed to find a suitable Vulkan device/queue family for compute + present.");
    }

    void createDeviceAndQueue()
    {
        const float queuePriority = 1.0f;

        const vk::DeviceQueueCreateInfo queueInfo(
            vk::DeviceQueueCreateFlags(),
            mQueueFamilyIndex,
            1,
            &queuePriority);

        const array<const char*, 1> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        const vk::DeviceCreateInfo deviceInfo(
            vk::DeviceCreateFlags(),
            1,
            &queueInfo,
            0,
            nullptr,
            static_cast<uint32_t>(deviceExtensions.size()),
            deviceExtensions.data(),
            nullptr);

        mDevice = make_shared<vk::Device>(mPhysicalDevice->createDevice(deviceInfo));
        mQueue = mDevice->getQueue(mQueueFamilyIndex, 0);
    }

    void createSwapchain()
    {
        const vk::SurfaceCapabilitiesKHR capabilities =
            mPhysicalDevice->getSurfaceCapabilitiesKHR(mSurface);
        const vector<vk::SurfaceFormatKHR> formats =
            mPhysicalDevice->getSurfaceFormatsKHR(mSurface);
        const vector<vk::PresentModeKHR> presentModes =
            mPhysicalDevice->getSurfacePresentModesKHR(mSurface);

        const vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
        const vk::PresentModeKHR presentMode = choosePresentMode(presentModes);
        const vk::Extent2D extent = chooseExtent(capabilities, mWindowWidth, mWindowHeight);

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        const vk::SwapchainCreateInfoKHR swapchainInfo(
            vk::SwapchainCreateFlagsKHR(),
            mSurface,
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1,
            vk::ImageUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode,
            VK_TRUE,
            vk::SwapchainKHR());

        mSwapchain = mDevice->createSwapchainKHR(swapchainInfo);
        mSwapchainImages = mDevice->getSwapchainImagesKHR(mSwapchain);
        if (mSwapchainImages.empty()) {
            throw runtime_error("Swapchain returned zero images.");
        }

        mSwapchainExtent = extent;
    }

    void createCommandPool()
    {
        const vk::CommandPoolCreateInfo poolInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            mQueueFamilyIndex);

        mCommandPool = mDevice->createCommandPool(poolInfo);
    }

    void createCommandBuffers()
    {
        const vk::CommandBufferAllocateInfo allocInfo(
            mCommandPool,
            vk::CommandBufferLevel::ePrimary,
            static_cast<uint32_t>(mSwapchainImages.size()));

        mCommandBuffers = mDevice->allocateCommandBuffers(allocInfo);
    }

    void createSyncObjects()
    {
        const vk::SemaphoreCreateInfo semaphoreInfo;
        mImageAvailableSemaphore = mDevice->createSemaphore(semaphoreInfo);
        mRenderFinishedSemaphore = mDevice->createSemaphore(semaphoreInfo);

        const vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        mInFlightFence = mDevice->createFence(fenceInfo);
    }

    void cleanup()
    {
        if (mDevice) {
            mDevice->waitIdle();

            if (mInFlightFence) {
                mDevice->destroyFence(mInFlightFence);
            }
            if (mImageAvailableSemaphore) {
                mDevice->destroySemaphore(mImageAvailableSemaphore);
            }
            if (mRenderFinishedSemaphore) {
                mDevice->destroySemaphore(mRenderFinishedSemaphore);
            }
            if (mCommandPool) {
                mDevice->destroyCommandPool(mCommandPool);
            }
            if (mSwapchain) {
                mDevice->destroySwapchainKHR(mSwapchain);
            }
        }

        if (mInstance && mSurface) {
            mInstance->destroySurfaceKHR(mSurface);
        }

        if (mDevice) {
            mDevice->destroy();
        }
        if (mInstance) {
            mInstance->destroy();
        }

        if (mWindow) {
            glfwDestroyWindow(mWindow);
            mWindow = nullptr;
        }
        glfwTerminate();
    }

private:
    uint32_t mWindowWidth = 0;
    uint32_t mWindowHeight = 0;

    GLFWwindow* mWindow = nullptr;

    shared_ptr<vk::Instance> mInstance;
    shared_ptr<vk::PhysicalDevice> mPhysicalDevice;
    shared_ptr<vk::Device> mDevice;

    vk::SurfaceKHR mSurface;
    uint32_t mQueueFamilyIndex = 0;
    vk::Queue mQueue;

    vk::SwapchainKHR mSwapchain;
    vk::Extent2D mSwapchainExtent;
    vector<vk::Image> mSwapchainImages;

    vk::CommandPool mCommandPool;
    vector<vk::CommandBuffer> mCommandBuffers;

    vk::Semaphore mImageAvailableSemaphore;
    vk::Semaphore mRenderFinishedSemaphore;
    vk::Fence mInFlightFence;
};

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
