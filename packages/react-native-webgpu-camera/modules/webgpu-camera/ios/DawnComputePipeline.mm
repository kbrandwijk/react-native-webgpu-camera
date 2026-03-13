#include "DawnComputePipeline.h"

#include "rnskia/RNDawnContext.h"
#include "include/core/SkImage.h"
#include "rnwgpu/api/GPUTexture.h"

#include <jsi/jsi.h>
#include <atomic>
#include <CoreVideo/CoreVideo.h>
#include <CoreVideo/CVPixelBufferIOSurface.h>

using namespace RNSkia;

namespace dawn_pipeline {

struct DawnComputePipeline::Impl {
  wgpu::Device device;
  wgpu::ComputePipeline computePipeline;
  wgpu::BindGroupLayout bindGroupLayout;
  wgpu::Texture outputTexture;
  sk_sp<SkImage> outputImage;

  // Reusable bind group — recreated each frame with new input texture
  wgpu::BindGroup bindGroup;
};

DawnComputePipeline::DawnComputePipeline()
    : _alive(std::make_shared<std::atomic<bool>>(true)) {}

DawnComputePipeline::~DawnComputePipeline() {
  _alive->store(false);
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

bool DawnComputePipeline::setup(const std::string &wgslCode, int width, int height) {
  std::lock_guard<std::mutex> lock(_mutex);

  cleanupLocked();
  _width = width;
  _height = height;
  _impl = new Impl();

  auto &ctx = DawnContext::getInstance();
  _impl->device = ctx.getWGPUDevice();

  if (!_impl->device) {
    printf("[DawnPipeline] No device available from DawnContext\n");
    delete _impl;
    _impl = nullptr;
    return false;
  }

  // Compile shader
  wgpu::ShaderModuleWGSLDescriptor wgslDesc;
  wgslDesc.code = wgslCode.c_str();

  wgpu::ShaderModuleDescriptor shaderDesc;
  shaderDesc.nextInChain = &wgslDesc;
  wgpu::ShaderModule shaderModule = _impl->device.CreateShaderModule(&shaderDesc);

  if (!shaderModule) {
    printf("[DawnPipeline] Failed to create shader module\n");
    delete _impl;
    _impl = nullptr;
    return false;
  }

  // Create compute pipeline
  wgpu::ComputePipelineDescriptor pipelineDesc;
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint = "main";
  _impl->computePipeline = _impl->device.CreateComputePipeline(&pipelineDesc);

  if (!_impl->computePipeline) {
    printf("[DawnPipeline] Failed to create compute pipeline\n");
    delete _impl;
    _impl = nullptr;
    return false;
  }

  _impl->bindGroupLayout = _impl->computePipeline.GetBindGroupLayout(0);

  // Create persistent output texture
  wgpu::TextureDescriptor texDesc;
  texDesc.label = "compute-output";
  texDesc.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
  texDesc.usage = wgpu::TextureUsage::StorageBinding |
                  wgpu::TextureUsage::TextureBinding |
                  wgpu::TextureUsage::CopySrc |
                  wgpu::TextureUsage::RenderAttachment;
  texDesc.dimension = wgpu::TextureDimension::e2D;
  texDesc.mipLevelCount = 1;
  texDesc.sampleCount = 1;

  _impl->outputTexture = _impl->device.CreateTexture(&texDesc);

  if (!_impl->outputTexture) {
    printf("[DawnPipeline] Failed to create output texture\n");
    delete _impl;
    _impl = nullptr;
    return false;
  }

  printf("[DawnPipeline] Setup complete: %dx%d\n", width, height);
  return true;
}

bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer) {
  std::lock_guard<std::mutex> lock(_mutex);

  if (!_impl || !pixelBuffer) return false;

  // Validate pixel buffer dimensions match setup
  int bufWidth = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
  int bufHeight = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
  if (bufWidth != _width || bufHeight != _height) {
    printf("[DawnPipeline] Frame size mismatch: got %dx%d, expected %dx%d\n",
           bufWidth, bufHeight, _width, _height);
    return false;
  }

  // 1. Import camera frame via IOSurface (zero-copy)
  IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
  if (!ioSurface) {
    printf("[DawnPipeline] No IOSurface on pixel buffer\n");
    return false;
  }

  wgpu::SharedTextureMemoryIOSurfaceDescriptor platformDesc;
  platformDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor memDesc = {};
  memDesc.nextInChain = &platformDesc;

  wgpu::SharedTextureMemory sharedMemory =
      _impl->device.ImportSharedTextureMemory(&memDesc);

  if (!sharedMemory) {
    printf("[DawnPipeline] Failed to import SharedTextureMemory\n");
    return false;
  }

  // Create input texture from shared memory
  // Camera is BGRA8Unorm on iOS
  wgpu::TextureDescriptor inputTexDesc;
  inputTexDesc.label = "camera-input";
  inputTexDesc.format = wgpu::TextureFormat::BGRA8Unorm;
  inputTexDesc.dimension = wgpu::TextureDimension::e2D;
  inputTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  inputTexDesc.size = {static_cast<uint32_t>(bufWidth),
                       static_cast<uint32_t>(bufHeight), 1};

  wgpu::Texture inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
  if (!inputTexture) {
    printf("[DawnPipeline] Failed to create input texture from SharedTextureMemory\n");
    return false;
  }

  // Begin access
  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc;
  beginDesc.initialized = true;
  beginDesc.fenceCount = 0;

  bool success = sharedMemory.BeginAccess(inputTexture, &beginDesc);
  if (!success) {
    printf("[DawnPipeline] BeginAccess failed\n");
    return false;
  }

  // From here, we MUST call EndAccess before returning
  bool result = false;

  // 2. Create bind group with input + output textures
  wgpu::BindGroupEntry entries[2];
  entries[0].binding = 0;
  entries[0].textureView = inputTexture.CreateView();
  entries[1].binding = 1;
  entries[1].textureView = _impl->outputTexture.CreateView();

  wgpu::BindGroupDescriptor bgDesc;
  bgDesc.layout = _impl->bindGroupLayout;
  bgDesc.entryCount = 2;
  bgDesc.entries = entries;

  _impl->bindGroup = _impl->device.CreateBindGroup(&bgDesc);
  if (!_impl->bindGroup) {
    printf("[DawnPipeline] Failed to create bind group\n");
    goto end_access;
  }

  {
    // 3. Dispatch compute shader
    wgpu::CommandEncoder encoder = _impl->device.CreateCommandEncoder();
    if (!encoder) {
      printf("[DawnPipeline] Failed to create command encoder\n");
      goto end_access;
    }

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(_impl->computePipeline);
    pass.SetBindGroup(0, _impl->bindGroup);
    pass.DispatchWorkgroups(
        (_width + 15) / 16,
        (_height + 15) / 16);
    pass.End();

    // Fix: store CommandBuffer in a local to avoid address-of-temporary UB
    wgpu::CommandBuffer commands = encoder.Finish();
    _impl->device.GetQueue().Submit(1, &commands);
  }

  {
    // 5. Wrap output texture as SkImage
    auto &ctx = DawnContext::getInstance();
    _impl->outputImage = ctx.MakeImageFromTexture(
        _impl->outputTexture, _width, _height,
        wgpu::TextureFormat::RGBA8Unorm);
    result = _impl->outputImage != nullptr;
  }

end_access:
  // 4. End access to input texture — always called after BeginAccess
  {
    wgpu::SharedTextureMemoryEndAccessState endState = {};
    sharedMemory.EndAccess(inputTexture, &endState);
  }

  return result;
}

void *DawnComputePipeline::getOutputSkImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !_impl->outputImage) return nullptr;
  // Return raw pointer to sk_sp<SkImage> — caller must not free
  return &_impl->outputImage;
}

void *DawnComputePipeline::getOutputTexturePtr() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !_impl->outputTexture) return nullptr;
  // Return pointer to the wgpu::Texture — used by JSI installer to wrap as GPUTexture
  return &_impl->outputTexture;
}

void DawnComputePipeline::cleanup() {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

void DawnComputePipeline::cleanupLocked() {
  if (_impl) {
    _impl->outputImage.reset();
    if (_impl->outputTexture) {
      _impl->outputTexture.Destroy();
    }
    _impl->bindGroup = nullptr;
    _impl->computePipeline = nullptr;
    _impl->bindGroupLayout = nullptr;
    _impl->device = nullptr;
    delete _impl;
    _impl = nullptr;
  }
  _width = 0;
  _height = 0;
}

} // namespace dawn_pipeline

// C interface for Swift
extern "C" {

DawnComputePipelineRef dawn_pipeline_create() {
  return new dawn_pipeline::DawnComputePipeline();
}

void dawn_pipeline_destroy(DawnComputePipelineRef ref) {
  delete static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
}

bool dawn_pipeline_setup(DawnComputePipelineRef ref, const char *wgslCode, int width, int height) {
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
  return pipeline->setup(std::string(wgslCode), width, height);
}

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref, CVPixelBufferRef pixelBuffer) {
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
  return pipeline->processFrame(pixelBuffer);
}

void *dawn_pipeline_get_output_image(DawnComputePipelineRef ref) {
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
  return pipeline->getOutputSkImage();
}

void *dawn_pipeline_get_output_texture(DawnComputePipelineRef ref) {
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
  return pipeline->getOutputTexturePtr();
}

void dawn_pipeline_cleanup(DawnComputePipelineRef ref) {
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);
  pipeline->cleanup();
}

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void *jsiRuntime) {
  auto &runtime = *static_cast<facebook::jsi::Runtime *>(jsiRuntime);
  auto *pipeline = static_cast<dawn_pipeline::DawnComputePipeline *>(ref);

  // Capture a shared_ptr<atomic<bool>> for liveness check — survives pipeline destruction.
  // The lambda may outlive the pipeline (it lives in the JS runtime), so we can't capture
  // the raw pipeline pointer without a guard.
  auto alive = pipeline->getAliveFlag();

  // Install global.__webgpuCamera_getOutputTexture()
  // Returns a GPUTexture JSI host object wrapping the compute output texture.
  // JS then calls Skia.Image.MakeImageFromTexture(texture) to get an SkImage.
  auto getOutputTexture = facebook::jsi::Function::createFromHostFunction(
      runtime,
      facebook::jsi::PropNameID::forAscii(runtime, "__webgpuCamera_getOutputTexture"),
      0,
      [pipeline, alive](facebook::jsi::Runtime &rt,
                 const facebook::jsi::Value &,
                 const facebook::jsi::Value *,
                 size_t) -> facebook::jsi::Value {
        // Check liveness — pipeline may have been destroyed
        if (!alive->load()) {
          return facebook::jsi::Value::null();
        }
        void *texPtr = pipeline->getOutputTexturePtr();
        if (!texPtr) {
          return facebook::jsi::Value::null();
        }
        // texPtr is a pointer to wgpu::Texture — dereference to get the actual texture
        wgpu::Texture texture = *static_cast<wgpu::Texture *>(texPtr);
        auto gpuTexture = std::make_shared<rnwgpu::GPUTexture>(texture, "compute-output");
        return rnwgpu::GPUTexture::create(rt, gpuTexture);
      });

  auto global = runtime.global();
  global.setProperty(runtime, "__webgpuCamera_getOutputTexture", std::move(getOutputTexture));
  printf("[DawnPipeline] JSI bindings installed\n");
}

} // extern "C"
