#include "DawnComputePipeline.h"

#include "rnskia/RNDawnContext.h"
#include "rnskia/RNSkPlatformContext.h"

#include "include/core/SkImage.h"
#include "include/core/SkSurface.h"
#include "include/core/SkCanvas.h"
#include "include/gpu/graphite/BackendTexture.h"
#include "include/gpu/graphite/Surface.h"

#include <jsi/jsi.h>
#include "api/JsiSkImage.h"
#include "api/JsiSkCanvas.h"

#include <CoreVideo/CoreVideo.h>
#include <CoreVideo/CVPixelBufferIOSurface.h>
#include <vector>

#import "SkiaManager.h"

namespace dawn_pipeline {

struct StagingBuffer {
  wgpu::Buffer gpuBuffer;
  wgpu::Buffer staging[2];
  int frameIndex = 0;
  int byteSize = 0;
  int elementSize = 0;
  int count = 0;
  int passIndex = 0;
  bool mapped[2] = {false, false};  // guarded by pipeline mutex
  const void* mappedData[2] = {nullptr, nullptr};
};

struct PassState {
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout bindGroupLayout;
  bool hasOutputBuffer = false;
  int bufferIndex = -1;
};

struct DawnComputePipeline::Impl {
  wgpu::Device device;
  std::vector<PassState> passes;
  wgpu::Texture texA;
  wgpu::Texture texB;
  std::vector<StagingBuffer> buffers;
  bool syncMode = false;
  bool useCanvas = false;
  sk_sp<SkSurface> surface;
  wgpu::Texture* finalTex = nullptr;
  sk_sp<SkImage> outputImage;
  bool imageDirty = true;  // set by processFrame, cleared by getOutputSkImage
};

DawnComputePipeline::DawnComputePipeline()
  : _alive(std::make_shared<std::atomic<bool>>(true)) {}

DawnComputePipeline::~DawnComputePipeline() {
  _alive->store(false);
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

bool DawnComputePipeline::setup(
    const std::vector<std::string>& wgslShaders,
    int width, int height,
    const std::vector<BufferSpec>& bufferSpecs,
    bool useCanvas, bool sync) {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();

  _width = width;
  _height = height;
  _impl = new Impl();

  auto& ctx = RNSkia::DawnContext::getInstance();
  _impl->device = ctx.getWGPUDevice();
  _impl->syncMode = sync;
  _impl->useCanvas = useCanvas;

  // Create ping-pong textures
  wgpu::TextureDescriptor texDesc{};
  texDesc.size = {(uint32_t)width, (uint32_t)height, 1};
  texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
  texDesc.usage = wgpu::TextureUsage::StorageBinding |
                  wgpu::TextureUsage::TextureBinding |
                  wgpu::TextureUsage::CopySrc |
                  wgpu::TextureUsage::RenderAttachment;
  texDesc.dimension = wgpu::TextureDimension::e2D;
  texDesc.mipLevelCount = 1;
  texDesc.sampleCount = 1;

  texDesc.label = "PingPong A";
  _impl->texA = _impl->device.CreateTexture(&texDesc);
  texDesc.label = "PingPong B";
  _impl->texB = _impl->device.CreateTexture(&texDesc);

  if (!_impl->texA || !_impl->texB) {
    cleanupLocked();
    return false;
  }

  // Compile shader passes
  for (size_t i = 0; i < wgslShaders.size(); i++) {
    PassState pass;

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = wgslShaders[i].c_str();

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc;

    auto shaderModule = _impl->device.CreateShaderModule(&smDesc);
    if (!shaderModule) {
      printf("[DawnPipeline] Failed to create shader module for pass %zu\n", i);
      cleanupLocked();
      return false;
    }

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";

    pass.pipeline = _impl->device.CreateComputePipeline(&cpDesc);
    if (!pass.pipeline) {
      printf("[DawnPipeline] Failed to create compute pipeline for pass %zu\n", i);
      cleanupLocked();
      return false;
    }

    pass.bindGroupLayout = pass.pipeline.GetBindGroupLayout(0);
    _impl->passes.push_back(std::move(pass));
  }

  // Create staging buffers for readback
  _impl->buffers.resize(bufferSpecs.size());
  for (size_t i = 0; i < bufferSpecs.size(); i++) {
    auto& spec = bufferSpecs[i];
    auto& sb = _impl->buffers[i];

    sb.passIndex = spec.passIndex;
    sb.elementSize = spec.elementSize;
    sb.count = spec.count;
    sb.byteSize = spec.elementSize * spec.count;

    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = sb.byteSize;
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufDesc.label = "OutputBuffer";
    sb.gpuBuffer = _impl->device.CreateBuffer(&bufDesc);

    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = sb.byteSize;
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    stagingDesc.label = "StagingA";
    sb.staging[0] = _impl->device.CreateBuffer(&stagingDesc);
    stagingDesc.label = "StagingB";
    sb.staging[1] = _impl->device.CreateBuffer(&stagingDesc);

    if (spec.passIndex >= 0 && spec.passIndex < (int)_impl->passes.size()) {
      _impl->passes[spec.passIndex].hasOutputBuffer = true;
      _impl->passes[spec.passIndex].bufferIndex = (int)i;
    }
  }

  // Determine final output texture
  _impl->finalTex = (wgslShaders.size() % 2 != 0) ? &_impl->texA : &_impl->texB;

  // Create SkSurface wrapping the final compute output texture for canvas drawing.
  // Skia Graphite draws directly onto this GPU texture — no extra composition needed.
  if (useCanvas) {
    // Get a Recorder* via a temporary offscreen surface (getRecorder() is private)
    auto tempSurface = ctx.MakeOffscreen(1, 1);
    auto* recorder = tempSurface->recorder();

    auto backendTex = skgpu::graphite::BackendTextures::MakeDawn(
      _impl->finalTex->Get());
    _impl->surface = SkSurfaces::WrapBackendTexture(
      recorder, backendTex,
      kRGBA_8888_SkColorType, nullptr, nullptr);
  }

  printf("[DawnPipeline] Multi-pass setup complete: %zu passes, %dx%d\n",
         wgslShaders.size(), width, height);
  return true;
}

bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !pixelBuffer) return false;

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto& device = _impl->device;

  IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
  if (!ioSurface) return false;

  wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
  ioDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor sharedDesc{};
  sharedDesc.nextInChain = &ioDesc;

  auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
  if (!sharedMemory) return false;

  // Create input texture with RGBA view format compatibility
  wgpu::TextureFormat viewFormats[] = {wgpu::TextureFormat::RGBA8Unorm};
  wgpu::TextureDescriptor inputTexDesc{};
  inputTexDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
  inputTexDesc.format = wgpu::TextureFormat::BGRA8Unorm;
  inputTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  inputTexDesc.dimension = wgpu::TextureDimension::e2D;
  inputTexDesc.mipLevelCount = 1;
  inputTexDesc.sampleCount = 1;
  inputTexDesc.label = "CameraInput";
  inputTexDesc.viewFormatCount = 1;
  inputTexDesc.viewFormats = viewFormats;

  auto inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
  if (!inputTexture) return false;

  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
  beginDesc.initialized = true;
  sharedMemory.BeginAccess(inputTexture, &beginDesc);

  // RGBA view override for pass 0
  wgpu::TextureViewDescriptor inputViewDesc{};
  inputViewDesc.format = wgpu::TextureFormat::RGBA8Unorm;
  auto inputView = inputTexture.CreateView(&inputViewDesc);

  auto encoder = device.CreateCommandEncoder();

  wgpu::TextureView readView = inputView;
  bool writeToA = true;

  for (size_t i = 0; i < _impl->passes.size(); i++) {
    auto& pass = _impl->passes[i];

    wgpu::Texture& writeTex = writeToA ? _impl->texA : _impl->texB;
    auto writeView = writeTex.CreateView();

    std::vector<wgpu::BindGroupEntry> entries;

    wgpu::BindGroupEntry entry0{};
    entry0.binding = 0;
    entry0.textureView = readView;
    entries.push_back(entry0);

    wgpu::BindGroupEntry entry1{};
    entry1.binding = 1;
    entry1.textureView = writeView;
    entries.push_back(entry1);

    if (pass.hasOutputBuffer && pass.bufferIndex >= 0) {
      auto& sb = _impl->buffers[pass.bufferIndex];
      wgpu::BindGroupEntry entry2{};
      entry2.binding = 2;
      entry2.buffer = sb.gpuBuffer;
      entry2.size = sb.byteSize;
      entries.push_back(entry2);
    }

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = pass.bindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    auto bindGroup = device.CreateBindGroup(&bgDesc);

    auto computePass = encoder.BeginComputePass();
    computePass.SetPipeline(pass.pipeline);
    computePass.SetBindGroup(0, bindGroup);
    computePass.DispatchWorkgroups((_width + 15) / 16, (_height + 15) / 16);
    computePass.End();

    readView = writeView;
    writeToA = !writeToA;
  }

  // Copy output buffers to staging
  for (auto& sb : _impl->buffers) {
    int stagingIdx = sb.frameIndex % 2;

    if (sb.mapped[stagingIdx]) {
      sb.staging[stagingIdx].Unmap();
      sb.mapped[stagingIdx] = false;
      sb.mappedData[stagingIdx] = nullptr;
    }

    encoder.CopyBufferToBuffer(sb.gpuBuffer, 0, sb.staging[stagingIdx], 0, sb.byteSize);
  }

  auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  // Async map staging buffers
  for (auto& sb : _impl->buffers) {
    int stagingIdx = sb.frameIndex % 2;

    sb.staging[stagingIdx].MapAsync(
      wgpu::MapMode::Read, 0, sb.byteSize,
      wgpu::CallbackMode::AllowProcessEvents,
      [&sb, stagingIdx](wgpu::MapAsyncStatus status, wgpu::StringView) {
        if (status == wgpu::MapAsyncStatus::Success) {
          sb.mappedData[stagingIdx] = sb.staging[stagingIdx].GetConstMappedRange(0, sb.byteSize);
          sb.mapped[stagingIdx] = true;
        }
      }
    );

    sb.frameIndex++;
  }

  // Sync mode: tick until all maps complete
  if (_impl->syncMode) {
    for (auto& sb : _impl->buffers) {
      int stagingIdx = (sb.frameIndex - 1) % 2;
      while (!sb.mapped[stagingIdx]) {
        device.Tick();
      }
    }
  }

  // Track which texture holds the final output (image created lazily on consumer thread)
  _impl->finalTex = writeToA ? &_impl->texB : &_impl->texA;
  _impl->imageDirty = true;

  // Cleanup IOSurface access
  wgpu::SharedTextureMemoryEndAccessState endState{};
  sharedMemory.EndAccess(inputTexture, &endState);

  return true;
}

const void* DawnComputePipeline::readBuffer(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return nullptr;

  auto& sb = _impl->buffers[bufferIndex];
  int readIdx = sb.frameIndex % 2;
  if (!sb.mapped[readIdx]) return nullptr;
  return sb.mappedData[readIdx];
}

int DawnComputePipeline::getBufferByteSize(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return 0;
  return _impl->buffers[bufferIndex].byteSize;
}

void* DawnComputePipeline::getSkSurface() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;
  return &_impl->surface;
}

void DawnComputePipeline::flushCanvas() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !_impl->surface) return;

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto recording = _impl->surface->recorder()->snap();
  if (recording) {
    ctx.submitRecording(recording.get());
  }

  if (_impl->finalTex) {
    _impl->outputImage = ctx.MakeImageFromTexture(
      *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm
    );
  }
}

void* DawnComputePipeline::getOutputSkImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !_impl->finalTex) return nullptr;

  // Recreate SkImage only when processFrame has produced new output.
  // Created on the calling (UI) thread's recorder so it's renderable.
  if (_impl->imageDirty) {
    auto& ctx = RNSkia::DawnContext::getInstance();
    _impl->outputImage = ctx.MakeImageFromTexture(
      *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm);
    _impl->imageDirty = false;
  }
  return &_impl->outputImage;
}

void DawnComputePipeline::cleanup() {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

void DawnComputePipeline::cleanupLocked() {
  if (!_impl) return;

  for (auto& sb : _impl->buffers) {
    for (int j = 0; j < 2; j++) {
      if (sb.mapped[j]) {
        sb.staging[j].Unmap();
        sb.mapped[j] = false;
        sb.mappedData[j] = nullptr;
      }
    }
  }

  _impl->outputImage.reset();
  _impl->surface.reset();
  _impl->texA = nullptr;
  _impl->texB = nullptr;
  _impl->passes.clear();
  _impl->buffers.clear();

  delete _impl;
  _impl = nullptr;
}

} // namespace dawn_pipeline

// ========== C interface ==========

extern "C" {

DawnComputePipelineRef dawn_pipeline_create() {
  return new dawn_pipeline::DawnComputePipeline();
}

void dawn_pipeline_destroy(DawnComputePipelineRef ref) {
  delete static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
}

bool dawn_pipeline_setup_multipass(
    DawnComputePipelineRef ref,
    const char** shaders, int shaderCount,
    int width, int height,
    const int* bufferSpecsFlat, int bufferCount,
    bool useCanvas, bool sync) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);

  std::vector<std::string> wgslShaders;
  for (int i = 0; i < shaderCount; i++) {
    wgslShaders.push_back(shaders[i]);
  }

  std::vector<dawn_pipeline::DawnComputePipeline::BufferSpec> specs;
  for (int i = 0; i < bufferCount; i++) {
    dawn_pipeline::DawnComputePipeline::BufferSpec s;
    s.passIndex = bufferSpecsFlat[i * 3 + 0];
    s.elementSize = bufferSpecsFlat[i * 3 + 1];
    s.count = bufferSpecsFlat[i * 3 + 2];
    specs.push_back(s);
  }

  return pipeline->setup(wgslShaders, width, height, specs, useCanvas, sync);
}

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref, CVPixelBufferRef pixelBuffer) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->processFrame(pixelBuffer);
}

const void* dawn_pipeline_read_buffer(DawnComputePipelineRef ref, int index) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->readBuffer(index);
}

int dawn_pipeline_get_buffer_byte_size(DawnComputePipelineRef ref, int index) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->getBufferByteSize(index);
}

void* dawn_pipeline_get_sk_surface(DawnComputePipelineRef ref) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->getSkSurface();
}

void dawn_pipeline_flush_canvas(DawnComputePipelineRef ref) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  pipeline->flushCanvas();
}

void* dawn_pipeline_get_output_image(DawnComputePipelineRef ref) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->getOutputSkImage();
}

void dawn_pipeline_cleanup(DawnComputePipelineRef ref) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  pipeline->cleanup();
}

// ========== CameraStreamHostObject ==========

class CameraStreamHostObject : public facebook::jsi::HostObject {
public:
  CameraStreamHostObject(
    dawn_pipeline::DawnComputePipeline* pipeline,
    std::shared_ptr<std::atomic<bool>> alive,
    std::shared_ptr<RNSkia::RNSkPlatformContext> platformContext)
    : _pipeline(pipeline), _alive(alive), _platformContext(platformContext) {}

  facebook::jsi::Value get(
      facebook::jsi::Runtime& runtime,
      const facebook::jsi::PropNameID& name) override {
    auto propName = name.utf8(runtime);

    if (propName == "nextImage") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto* imgPtr = static_cast<sk_sp<SkImage>*>(
            _pipeline->getOutputSkImage());
          if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

          auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
            _platformContext, *imgPtr);
          return facebook::jsi::Object::createFromHostObject(rt, hostObj);
        });
    }

    if (propName == "readBuffer") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 1,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value* args,
               size_t count) -> facebook::jsi::Value {
          if (!_alive->load() || count < 1) return facebook::jsi::Value::null();

          int index = (int)args[0].asNumber();
          const void* data = _pipeline->readBuffer(index);
          if (!data) return facebook::jsi::Value::null();

          int byteSize = _pipeline->getBufferByteSize(index);
          if (byteSize <= 0) return facebook::jsi::Value::null();

          auto arrayBuffer = rt.global()
            .getPropertyAsFunction(rt, "ArrayBuffer")
            .callAsConstructor(rt, byteSize)
            .asObject(rt)
            .getArrayBuffer(rt);
          memcpy(arrayBuffer.data(rt), data, byteSize);

          return arrayBuffer;
        });
    }

    if (propName == "getCanvas") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto* surfPtr = static_cast<sk_sp<SkSurface>*>(
            _pipeline->getSkSurface());
          if (!surfPtr || !*surfPtr) return facebook::jsi::Value::null();

          SkCanvas* canvas = (*surfPtr)->getCanvas();
          if (!canvas) return facebook::jsi::Value::null();

          auto hostObj = std::make_shared<RNSkia::JsiSkCanvas>(
            _platformContext, canvas);
          return facebook::jsi::Object::createFromHostObject(rt, hostObj);
        });
    }

    if (propName == "flushCanvas") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::undefined();
          _pipeline->flushCanvas();
          return facebook::jsi::Value::undefined();
        });
    }

    if (propName == "dispose") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [](facebook::jsi::Runtime&,
           const facebook::jsi::Value&,
           const facebook::jsi::Value*,
           size_t) -> facebook::jsi::Value {
          return facebook::jsi::Value::undefined();
        });
    }

    return facebook::jsi::Value::undefined();
  }

private:
  dawn_pipeline::DawnComputePipeline* _pipeline;
  std::shared_ptr<std::atomic<bool>> _alive;
  std::shared_ptr<RNSkia::RNSkPlatformContext> _platformContext;
};

// ========== JSI Installation ==========

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  auto& runtime = *static_cast<facebook::jsi::Runtime*>(jsiRuntime);

  std::shared_ptr<RNSkia::RNSkManager> skManager = [SkiaManager latestActiveSkManager];
  if (!skManager) {
    printf("[DawnPipeline] WARNING: SkiaManager not available\n");
    return;
  }
  auto platformContext = skManager->getPlatformContext();
  auto alive = pipeline->alive();

  auto nextImageFn = facebook::jsi::Function::createFromHostFunction(
    runtime,
    facebook::jsi::PropNameID::forAscii(runtime, "__webgpuCamera_nextImage"),
    0,
    [pipeline, alive, platformContext](
      facebook::jsi::Runtime& rt,
      const facebook::jsi::Value&,
      const facebook::jsi::Value*,
      size_t) -> facebook::jsi::Value {
      if (!alive->load()) return facebook::jsi::Value::null();

      auto* imgPtr = static_cast<sk_sp<SkImage>*>(pipeline->getOutputSkImage());
      if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

      auto hostObj = std::make_shared<RNSkia::JsiSkImage>(platformContext, *imgPtr);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_nextImage", std::move(nextImageFn));

  auto createStreamFn = facebook::jsi::Function::createFromHostFunction(
    runtime,
    facebook::jsi::PropNameID::forAscii(runtime, "__webgpuCamera_createStream"),
    0,
    [pipeline, alive, platformContext](
      facebook::jsi::Runtime& rt,
      const facebook::jsi::Value&,
      const facebook::jsi::Value*,
      size_t) -> facebook::jsi::Value {
      auto hostObj = std::make_shared<CameraStreamHostObject>(pipeline, alive, platformContext);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_createStream", std::move(createStreamFn));

  printf("[DawnPipeline] JSI bindings installed (nextImage + createStream)\n");
}

} // extern "C"
