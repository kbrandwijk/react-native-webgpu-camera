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
  wgpu::BindGroup cachedBindGroup;  // cached for passes 1+ (static ping-pong textures)
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
  bool rawCamera = false;  // true when 0 shaders, no canvas, no buffers
  wgpu::Texture canvasTex;  // separate texture for Skia draws — not touched by compute
  sk_sp<SkSurface> surface;
  wgpu::Texture* finalTex = nullptr;
  sk_sp<SkImage> outputImage;
  sk_sp<SkImage> compositedImage;  // set by flushCanvas, consumed by next nextImage call

  // FPS tracking — counts processFrame completions per second
  int frameCount = 0;
  double lastFpsTime = 0;
  int currentFps = 0;

  // Generation counter — increments each processFrame completion
  int generation = 0;

  // Per-step timing (ms) — updated every frame, logged every second
  double t_lockWait = 0;
  double t_import = 0;
  double t_bindGroup = 0;
  double t_compute = 0;
  double t_buffers = 0;
  double t_makeImage = 0;
  double t_total = 0;
  double t_wall = 0;  // wall time including lock wait
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

  // TODO: Raw camera path (0 shaders, no canvas, no buffers) needs texture
  // lifetime management — IOSurface texture is freed when processFrame returns.
  // For now, always use the compute path with auto-inserted passthrough shader.

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

  // Auto-insert passthrough shader when no user shaders but canvas/buffers needed.
  // Handles BGRA→RGBA conversion and gives onFrame a canvas to draw on.
  static const std::string kPassthroughWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  textureStore(outputTex, vec2i(id.xy), textureLoad(inputTex, vec2i(id.xy), 0));
}
)";

  auto effectiveShaders = wgslShaders;
  if (effectiveShaders.empty()) {
    effectiveShaders.push_back(kPassthroughWGSL);
  }

  // Compile shader passes
  for (size_t i = 0; i < effectiveShaders.size(); i++) {
    PassState pass;

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = effectiveShaders[i].c_str();

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
  _impl->finalTex = (effectiveShaders.size() % 2 != 0) ? &_impl->texA : &_impl->texB;

  // Cache bind groups for passes 1+ (static ping-pong textures).
  // Pass 0 reads from the camera input which changes every frame — can't cache.
  for (size_t i = 1; i < _impl->passes.size(); i++) {
    auto& pass = _impl->passes[i];
    // Pass i reads from the output of pass i-1, writes to the next ping-pong texture.
    // Odd passes read texA write texB, even passes read texB write texA.
    bool readFromA = (i % 2 != 0);
    wgpu::Texture& readTex = readFromA ? _impl->texA : _impl->texB;
    wgpu::Texture& writeTex = readFromA ? _impl->texB : _impl->texA;

    std::vector<wgpu::BindGroupEntry> entries;
    wgpu::BindGroupEntry entry0{};
    entry0.binding = 0;
    entry0.textureView = readTex.CreateView();
    entries.push_back(entry0);

    wgpu::BindGroupEntry entry1{};
    entry1.binding = 1;
    entry1.textureView = writeTex.CreateView();
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
    pass.cachedBindGroup = _impl->device.CreateBindGroup(&bgDesc);
  }

  // Create a separate texture for Skia canvas draws, isolated from compute output.
  // processFrame writes to finalTex; Skia draws go on canvasTex to avoid races.
  if (useCanvas) {
    wgpu::TextureDescriptor canvasTexDesc{};
    canvasTexDesc.size = {(uint32_t)width, (uint32_t)height, 1};
    canvasTexDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    canvasTexDesc.usage = wgpu::TextureUsage::TextureBinding |
                          wgpu::TextureUsage::CopySrc |
                          wgpu::TextureUsage::CopyDst |
                          wgpu::TextureUsage::RenderAttachment;
    canvasTexDesc.label = "CanvasTex";
    _impl->canvasTex = _impl->device.CreateTexture(&canvasTexDesc);

    auto tempSurface = ctx.MakeOffscreen(1, 1);
    auto* recorder = tempSurface->recorder();

    auto backendTex = skgpu::graphite::BackendTextures::MakeDawn(
      _impl->canvasTex.Get());
    _impl->surface = SkSurfaces::WrapBackendTexture(
      recorder, backendTex,
      kRGBA_8888_SkColorType, nullptr, nullptr);
  }

  printf("[DawnPipeline] Setup complete: %zu passes, %zu buffers, %dx%d%s\n",
         effectiveShaders.size(), bufferSpecs.size(), width, height,
         useCanvas ? " +canvas" : "");
  return true;
}

bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer) {
  double tWallStart = CACurrentMediaTime();

  // Brief lock just to check _impl and grab stable references
  Impl* impl;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_impl || !pixelBuffer) return false;
    impl = _impl;
  }
  double tAfterLock = CACurrentMediaTime();

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto& device = impl->device;

  double tStart = tAfterLock;

  // ── All GPU work below — no mutex needed ──

  IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
  if (!ioSurface) return false;

  wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
  ioDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor sharedDesc{};
  sharedDesc.nextInChain = &ioDesc;

  auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
  if (!sharedMemory) return false;

  // Create input texture from camera IOSurface (BGRA on iOS)
  wgpu::TextureDescriptor inputTexDesc{};
  inputTexDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
  inputTexDesc.format = wgpu::TextureFormat::BGRA8Unorm;
  inputTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  inputTexDesc.dimension = wgpu::TextureDimension::e2D;
  inputTexDesc.mipLevelCount = 1;
  inputTexDesc.sampleCount = 1;
  inputTexDesc.label = "CameraInput";

  auto inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
  if (!inputTexture) return false;

  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
  beginDesc.initialized = true;
  sharedMemory.BeginAccess(inputTexture, &beginDesc);

  double tAfterImport = CACurrentMediaTime();

  // ── Raw camera path: no compute, just wrap the BGRA texture as SkImage ──
  if (impl->rawCamera) {
    auto outputImage = ctx.MakeImageFromTexture(
      inputTexture, _width, _height, wgpu::TextureFormat::BGRA8Unorm);

    wgpu::SharedTextureMemoryEndAccessState endState{};
    sharedMemory.EndAccess(inputTexture, &endState);

    // Brief lock to publish results
    std::lock_guard<std::mutex> lock(_mutex);
    impl->outputImage = std::move(outputImage);
    impl->frameCount++;
    impl->generation++;
    auto now = CACurrentMediaTime();
    if (impl->lastFpsTime == 0) {
      impl->lastFpsTime = now;
    } else if (now - impl->lastFpsTime >= 1.0) {
      impl->currentFps = (int)(impl->frameCount / (now - impl->lastFpsTime));
      impl->frameCount = 0;
      impl->lastFpsTime = now;
    }
    return true;
  }

  // ── Compute path ──

  // Process pending async callbacks from previous frame (e.g. MapAsync).
  // AllowProcessEvents callbacks fire during ProcessEvents(), not Tick().
  if (!impl->syncMode && !impl->buffers.empty()) {
    ctx.getWGPUInstance().ProcessEvents();
  }

  double tAfterBindGroup = tAfterImport; // updated after loop

  auto encoder = device.CreateCommandEncoder();

  for (size_t i = 0; i < impl->passes.size(); i++) {
    auto& pass = impl->passes[i];

    wgpu::BindGroup bindGroup;
    if (i == 0) {
      // Pass 0: camera input changes every frame — build bind group per frame
      bool writeToA = true;
      wgpu::Texture& writeTex = writeToA ? impl->texA : impl->texB;

      std::vector<wgpu::BindGroupEntry> entries;
      wgpu::BindGroupEntry entry0{};
      entry0.binding = 0;
      entry0.textureView = inputTexture.CreateView();
      entries.push_back(entry0);

      wgpu::BindGroupEntry entry1{};
      entry1.binding = 1;
      entry1.textureView = writeTex.CreateView();
      entries.push_back(entry1);

      if (pass.hasOutputBuffer && pass.bufferIndex >= 0) {
        auto& sb = impl->buffers[pass.bufferIndex];
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
      bindGroup = device.CreateBindGroup(&bgDesc);
    } else {
      // Passes 1+: cached bind group (static ping-pong textures)
      bindGroup = pass.cachedBindGroup;
    }

    auto computePass = encoder.BeginComputePass();
    computePass.SetPipeline(pass.pipeline);
    computePass.SetBindGroup(0, bindGroup);
    computePass.DispatchWorkgroups((_width + 15) / 16, (_height + 15) / 16);
    computePass.End();
  }

  tAfterBindGroup = CACurrentMediaTime();

  // Submit compute work
  auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  double tAfterCompute = CACurrentMediaTime();

  // Copy output buffers to staging in a separate submission
  if (!impl->buffers.empty()) {
    auto copyEncoder = device.CreateCommandEncoder();

    for (auto& sb : impl->buffers) {
      int stagingIdx = sb.frameIndex % 2;

      if (sb.mapped[stagingIdx]) {
        sb.staging[stagingIdx].Unmap();
        sb.mapped[stagingIdx] = false;
        sb.mappedData[stagingIdx] = nullptr;
      }

      copyEncoder.CopyBufferToBuffer(sb.gpuBuffer, 0, sb.staging[stagingIdx], 0, sb.byteSize);
    }

    auto copyCommands = copyEncoder.Finish();
    device.GetQueue().Submit(1, &copyCommands);

    // Map staging buffers for readback
    for (auto& sb : impl->buffers) {
      int stagingIdx = sb.frameIndex % 2;

      if (impl->syncMode) {
        auto instance = ctx.getWGPUInstance();
        auto future = sb.staging[stagingIdx].MapAsync(
          wgpu::MapMode::Read, 0, sb.byteSize,
          wgpu::CallbackMode::WaitAnyOnly,
          [&sb, stagingIdx](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
              sb.mappedData[stagingIdx] = sb.staging[stagingIdx].GetConstMappedRange(0, sb.byteSize);
              sb.mapped[stagingIdx] = true;
            }
          }
        );
        instance.WaitAny(future, UINT64_MAX);
      } else {
        // Async readback: AllowProcessEvents callback fires on next ProcessEvents() call.
        // Data arrives one frame behind — imperceptible for overlays.
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
      }

      sb.frameIndex++;
    }
  }

  double tAfterBuffers = CACurrentMediaTime();

  // Determine final output texture and create SkImage
  bool finalIsA = (impl->passes.size() % 2 != 0);
  auto outputImage = ctx.MakeImageFromTexture(
    *(finalIsA ? &impl->texA : &impl->texB), _width, _height, wgpu::TextureFormat::RGBA8Unorm);

  double tAfterMakeImage = CACurrentMediaTime();

  // Cleanup IOSurface access
  wgpu::SharedTextureMemoryEndAccessState endState{};
  sharedMemory.EndAccess(inputTexture, &endState);

  // ── Brief lock to publish results ──
  {
    std::lock_guard<std::mutex> lock(_mutex);
    impl->finalTex = finalIsA ? &impl->texA : &impl->texB;
    impl->outputImage = std::move(outputImage);

    // Per-step timing (ms)
    impl->t_lockWait = (tAfterLock - tWallStart) * 1000.0;
    impl->t_import = (tAfterImport - tStart) * 1000.0;
    impl->t_bindGroup = (tAfterBindGroup - tAfterImport) * 1000.0;
    impl->t_compute = (tAfterCompute - tAfterBindGroup) * 1000.0;
    impl->t_buffers = (tAfterBuffers - tAfterCompute) * 1000.0;
    impl->t_makeImage = (tAfterMakeImage - tAfterBuffers) * 1000.0;
    impl->t_total = (tAfterMakeImage - tStart) * 1000.0;
    impl->t_wall = (tAfterMakeImage - tWallStart) * 1000.0;

    // FPS tracking + generation counter
    impl->frameCount++;
    impl->generation++;
    auto now = CACurrentMediaTime();
    if (impl->lastFpsTime == 0) {
      impl->lastFpsTime = now;
    } else if (now - impl->lastFpsTime >= 1.0) {
      impl->currentFps = (int)(impl->frameCount / (now - impl->lastFpsTime));
      impl->frameCount = 0;
      impl->lastFpsTime = now;
    }
  }

  return true;
}

const void* DawnComputePipeline::readBuffer(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return nullptr;

  auto& sb = _impl->buffers[bufferIndex];
  // Double-buffer read: frameIndex points past the last write.
  // The Tick at the start of frame N+1 completes the map from frame N.
  // After Tick + unmap/write/map cycle, the *other* staging buffer holds completed data.
  // Try both — return whichever is mapped.
  for (int i = 0; i < 2; i++) {
    if (sb.mapped[i] && sb.mappedData[i]) return sb.mappedData[i];
  }
  return nullptr;
}

int DawnComputePipeline::getBufferByteSize(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return 0;
  return _impl->buffers[bufferIndex].byteSize;
}

int DawnComputePipeline::pipelineFps() const {
  if (!_impl) return 0;
  return _impl->currentFps;
}

int DawnComputePipeline::generation() const {
  if (!_impl) return 0;
  return _impl->generation;
}

double DawnComputePipeline::metricLockWait() const { return _impl ? _impl->t_lockWait : 0; }
double DawnComputePipeline::metricImport() const { return _impl ? _impl->t_import : 0; }
double DawnComputePipeline::metricBindGroup() const { return _impl ? _impl->t_bindGroup : 0; }
double DawnComputePipeline::metricCompute() const { return _impl ? _impl->t_compute : 0; }
double DawnComputePipeline::metricBuffers() const { return _impl ? _impl->t_buffers : 0; }
double DawnComputePipeline::metricMakeImage() const { return _impl ? _impl->t_makeImage : 0; }
double DawnComputePipeline::metricTotal() const { return _impl ? _impl->t_total : 0; }
double DawnComputePipeline::metricWall() const { return _impl ? _impl->t_wall : 0; }

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
    ctx.submitRecording(recording.get(), skgpu::graphite::SyncToCpu::kNo);
  }

  if (_impl->finalTex) {
    // Store as compositedImage so the next nextImage() call returns this
    // instead of outputImage (which processFrame may overwrite on the camera thread).
    _impl->compositedImage = ctx.MakeImageFromTexture(
      *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm
    );
  }
}

DawnComputePipeline::FrameData DawnComputePipeline::beginFrame() {
  FrameData fd;
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return fd;

  // Process pending async buffer maps
  if (!_impl->syncMode && !_impl->buffers.empty()) {
    auto& ctx = RNSkia::DawnContext::getInstance();
    ctx.getWGPUInstance().ProcessEvents();
  }

  // Image
  if (_impl->compositedImage) {
    _impl->outputImage = std::move(_impl->compositedImage);
    _impl->compositedImage = nullptr;
  }
  if (_impl->outputImage) {
    fd.image = &_impl->outputImage;
  }

  // Buffers
  fd.bufferCount = std::min((int)_impl->buffers.size(), 8);
  for (int bi = 0; bi < fd.bufferCount; bi++) {
    auto& sb = _impl->buffers[bi];
    fd.bufferByteSizes[bi] = sb.byteSize;
    fd.bufferData[bi] = nullptr;
    for (int si = 0; si < 2; si++) {
      if (sb.mapped[si] && sb.mappedData[si]) {
        fd.bufferData[bi] = sb.mappedData[si];
        break;
      }
    }
  }

  // Canvas surface
  if (_impl->surface) {
    fd.surface = &_impl->surface;
  }

  // FPS + generation + metrics — all in the same lock
  fd.pipelineFps = _impl->currentFps;
  fd.generation = _impl->generation;
  fd.metricLockWait = _impl->t_lockWait;
  fd.metricImport = _impl->t_import;
  fd.metricBindGroup = _impl->t_bindGroup;
  fd.metricCompute = _impl->t_compute;
  fd.metricBuffers = _impl->t_buffers;
  fd.metricMakeImage = _impl->t_makeImage;
  fd.metricTotal = _impl->t_total;
  fd.metricWall = _impl->t_wall;

  return fd;
}

void* DawnComputePipeline::flushCanvasAndGetImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;

  if (_impl->surface && _impl->finalTex && _impl->canvasTex) {
    auto& ctx = RNSkia::DawnContext::getInstance();

    // Copy compute output → canvasTex so Skia draws composite on a stable snapshot.
    // This prevents the race where processFrame overwrites finalTex mid-draw.
    auto encoder = _impl->device.CreateCommandEncoder();
    wgpu::TexelCopyTextureInfo src{};
    src.texture = *_impl->finalTex;
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = _impl->canvasTex;
    wgpu::Extent3D extent = {(uint32_t)_width, (uint32_t)_height, 1};
    encoder.CopyTextureToTexture(&src, &dst, &extent);
    auto commands = encoder.Finish();
    _impl->device.GetQueue().Submit(1, &commands);

    // Flush Skia draws (recorded by onFrame) onto canvasTex
    auto recording = _impl->surface->recorder()->snap();
    if (recording) {
      ctx.submitRecording(recording.get(), skgpu::graphite::SyncToCpu::kNo);
    }

    // Wrap canvasTex (compute + Skia draws) as the output image
    _impl->outputImage = ctx.MakeImageFromTexture(
      _impl->canvasTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm
    );
  }

  if (!_impl->outputImage) return nullptr;
  return &_impl->outputImage;
}

void* DawnComputePipeline::getOutputSkImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;

  // Prefer composited image (from flushCanvas) — consume it so it's one-shot
  if (_impl->compositedImage) {
    _impl->outputImage = std::move(_impl->compositedImage);
    _impl->compositedImage = nullptr;
  }

  if (!_impl->outputImage) return nullptr;
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

    if (propName == "flushCanvasAndGetImage") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto* imgPtr = static_cast<sk_sp<SkImage>*>(
            _pipeline->flushCanvasAndGetImage());
          if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

          auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
            _platformContext, *imgPtr);
          return facebook::jsi::Object::createFromHostObject(rt, hostObj);
        });
    }

    if (propName == "beginFrame") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto fd = _pipeline->beginFrame();

          auto result = facebook::jsi::Object(rt);

          // image
          if (fd.image) {
            auto* imgPtr = static_cast<sk_sp<SkImage>*>(fd.image);
            if (imgPtr && *imgPtr) {
              auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
                _platformContext, *imgPtr);
              result.setProperty(rt, "image",
                facebook::jsi::Object::createFromHostObject(rt, hostObj));
            }
          }

          // canvas
          if (fd.surface) {
            auto* surfPtr = static_cast<sk_sp<SkSurface>*>(fd.surface);
            if (surfPtr && *surfPtr) {
              SkCanvas* canvas = (*surfPtr)->getCanvas();
              if (canvas) {
                auto hostObj = std::make_shared<RNSkia::JsiSkCanvas>(
                  _platformContext, canvas);
                result.setProperty(rt, "canvas",
                  facebook::jsi::Object::createFromHostObject(rt, hostObj));
              }
            }
          }

          // buffers as array
          auto bufsArr = facebook::jsi::Array(rt, fd.bufferCount);
          for (int i = 0; i < fd.bufferCount; i++) {
            if (fd.bufferData[i] && fd.bufferByteSizes[i] > 0) {
              auto arrayBuffer = rt.global()
                .getPropertyAsFunction(rt, "ArrayBuffer")
                .callAsConstructor(rt, fd.bufferByteSizes[i])
                .asObject(rt)
                .getArrayBuffer(rt);
              memcpy(arrayBuffer.data(rt), fd.bufferData[i], fd.bufferByteSizes[i]);
              bufsArr.setValueAtIndex(rt, i, std::move(arrayBuffer));
            } else {
              bufsArr.setValueAtIndex(rt, i, facebook::jsi::Value::null());
            }
          }
          result.setProperty(rt, "buffers", std::move(bufsArr));

          // FPS + generation + metrics
          result.setProperty(rt, "pipelineFps", fd.pipelineFps);
          result.setProperty(rt, "generation", fd.generation);

          auto m = facebook::jsi::Object(rt);
          m.setProperty(rt, "lockWait", fd.metricLockWait);
          m.setProperty(rt, "import", fd.metricImport);
          m.setProperty(rt, "bindGroup", fd.metricBindGroup);
          m.setProperty(rt, "compute", fd.metricCompute);
          m.setProperty(rt, "buffers", fd.metricBuffers);
          m.setProperty(rt, "makeImage", fd.metricMakeImage);
          m.setProperty(rt, "total", fd.metricTotal);
          m.setProperty(rt, "wall", fd.metricWall);
          result.setProperty(rt, "metrics", std::move(m));

          return result;
        });
    }

    if (propName == "pipelineFps") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime&,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value(0);
          return facebook::jsi::Value(_pipeline->pipelineFps());
        });
    }

    if (propName == "generation") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime&,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value(0);
          return facebook::jsi::Value(_pipeline->generation());
        });
    }

    if (propName == "metrics") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();
          auto obj = facebook::jsi::Object(rt);
          obj.setProperty(rt, "lockWait", _pipeline->metricLockWait());
          obj.setProperty(rt, "import", _pipeline->metricImport());
          obj.setProperty(rt, "bindGroup", _pipeline->metricBindGroup());
          obj.setProperty(rt, "compute", _pipeline->metricCompute());
          obj.setProperty(rt, "buffers", _pipeline->metricBuffers());
          obj.setProperty(rt, "makeImage", _pipeline->metricMakeImage());
          obj.setProperty(rt, "total", _pipeline->metricTotal());
          obj.setProperty(rt, "wall", _pipeline->metricWall());
          return obj;
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
