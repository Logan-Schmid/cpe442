#pragma once

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <memory>
#include <vector>

class VulkanDisplay
{
public:
    VulkanDisplay(uint32_t width, uint32_t height);
    ~VulkanDisplay();

    std::shared_ptr<vk::Instance> getInstance() const;
    std::shared_ptr<vk::PhysicalDevice> getPhysicalDevice() const;
    std::shared_ptr<vk::Device> getDevice() const;

    bool shouldClose() const;
    void pollEvents() const;

    void presentFromBuffer(
        const vk::Buffer& sourceBuffer,
        uint32_t sourceWidth,
        uint32_t sourceHeight);

private:
    static vk::SurfaceFormatKHR chooseSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& formats);
    static vk::PresentModeKHR choosePresentMode(
        const std::vector<vk::PresentModeKHR>& modes);
    static vk::Extent2D chooseExtent(
        const vk::SurfaceCapabilitiesKHR& capabilities,
        uint32_t requestedWidth,
        uint32_t requestedHeight);

    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void pickPhysicalDeviceAndQueueFamily();
    void createDeviceAndQueue();
    void createSwapchain();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void cleanup();

    uint32_t mWindowWidth = 0;
    uint32_t mWindowHeight = 0;

    GLFWwindow* mWindow = nullptr;

    std::shared_ptr<vk::Instance> mInstance;
    std::shared_ptr<vk::PhysicalDevice> mPhysicalDevice;
    std::shared_ptr<vk::Device> mDevice;

    vk::SurfaceKHR mSurface;
    uint32_t mQueueFamilyIndex = 0;
    vk::Queue mQueue;

    vk::SwapchainKHR mSwapchain;
    vk::Extent2D mSwapchainExtent;
    std::vector<vk::Image> mSwapchainImages;

    vk::CommandPool mCommandPool;
    std::vector<vk::CommandBuffer> mCommandBuffers;

    vk::Semaphore mImageAvailableSemaphore;
    vk::Semaphore mRenderFinishedSemaphore;
    vk::Fence mInFlightFence;
};
