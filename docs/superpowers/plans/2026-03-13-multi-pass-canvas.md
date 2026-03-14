# Multi-Pass Compute + frame.canvas Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend useGPUFrameProcessor with multi-pass shader chaining, real Skia canvas drawing on processed frames, and GPU buffer readback.

**Architecture:** Replace the single-pipeline DawnComputePipeline with a multi-pass engine using ping-pong textures. Add double-buffered staging buffers for GPU→CPU readback. Expose SkSurface-backed canvas via JSI for Skia draws in the worklet. Update the TypeScript hook to support both shorthand (callback) and object form (pipeline + onFrame).

**Tech Stack:** C++ (Dawn/WebGPU, Skia Graphite), Objective-C++ (bridge), Swift (Expo module), TypeScript (React hooks, Reanimated worklets)

**Spec:** `docs/superpowers/specs/2026-03-13-multi-pass-canvas-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/react-native-webgpu-camera/src/types.ts` | Modify | Add `BufferHandle`, `RenderFrame`, `ProcessorConfig`, `TypedArrayConstructor`, update `ProcessorFrame`, add overloads |
| `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` | Modify | Multi-pass capture proxy, object form handling, onFrame worklet, buffer resolution |
| `packages/react-native-webgpu-camera/src/index.ts` | Modify | Re-export new types |
| `modules/webgpu-camera/ios/DawnComputePipeline.h` | Modify | New C interface for multi-pass setup, buffer readback, canvas |
| `modules/webgpu-camera/ios/DawnComputePipeline.mm` | Modify | Multi-pipeline, ping-pong textures, staging buffers, SkSurface, updated CameraStreamHostObject |
| `modules/webgpu-camera/ios/DawnPipelineBridge.h` | Modify | Updated bridge interface |
| `modules/webgpu-camera/ios/DawnPipelineBridge.mm` | Modify | Bridge multi-pass setup |
| `modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Modify | Call new setup function |
| `apps/example/src/app/index.tsx` | Modify | Demo multi-pass + canvas |

Note: All native file paths below are relative to `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/` unless fully qualified.

---

## Chunk 1: TypeScript Types

### Task 1: Update shared types

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/types.ts`

- [ ] **Step 1: Update types.ts with multi-pass types**

Replace the entire file:

```typescript
import type { SharedValue } from 'react-native-reanimated';
import type { SkImage, SkCanvas } from '@shopify/react-native-skia';

// useCamera

export interface CameraConfig {
  device: 'back' | 'front';
  width: number;
  height: number;
  fps: number;
}

export interface CameraHandle {
  /** True once camera is producing frames */
  isReady: boolean;
  /** Camera frame dimensions (passed through from config) */
  width: number;
  height: number;
  fps: number;
}

// useGPUFrameProcessor

/** JSI host object shared across Reanimated runtimes */
export interface CameraStream {
  nextImage(): SkImage | null;
  readBuffer(index: number): ArrayBuffer | null;
  getCanvas(): SkCanvas | null;
  flushCanvas(): void;
  dispose(): void;
}

/** Union of all typed array constructors — used for buffer output type inference */
export type TypedArrayConstructor =
  | typeof Float32Array | typeof Float64Array
  | typeof Int8Array | typeof Int16Array | typeof Int32Array
  | typeof Uint8Array | typeof Uint16Array | typeof Uint32Array
  | typeof Uint8ClampedArray;

/** Opaque handle returned by runShader in pipeline — resolved to live data in onFrame */
declare const __bufferBrand: unique symbol;
export type BufferHandle<T> = T & { readonly [__bufferBrand]: never };

/** Setup-time frame interface — used inside pipeline callback */
export interface ProcessorFrame {
  /** Run a compute shader — output feeds into next pass or becomes final frame */
  runShader(wgsl: string): void;
  /** Run a compute shader with buffer output — returns a handle resolved per-frame */
  runShader<T extends TypedArrayConstructor>(
    wgsl: string,
    options: { output: T; count: number },
  ): BufferHandle<InstanceType<T>>;

  /** Skia canvas targeting the current pass's output texture.
   *  Only valid inside `pipeline` for between-pass draws.
   *  Draws are recorded once at setup time and replayed every frame (static).
   *  For per-frame dynamic draws, use `onFrame`'s `RenderFrame.canvas`. */
  canvas: SkCanvas;

  /** Current frame dimensions */
  width: number;
  height: number;
}

/** Per-frame render interface — used inside onFrame callback */
export interface RenderFrame {
  /** Skia canvas targeting the final compute output texture */
  canvas: SkCanvas;
  /** Current frame dimensions */
  width: number;
  height: number;
}

/** Strips BufferHandle brand and adds | null for each value */
export type NullableBuffers<B> = {
  [K in keyof B]: B[K] extends BufferHandle<infer U> ? U | null : B[K] | null;
};

/** Configuration for the object form of useGPUFrameProcessor */
export interface ProcessorConfig<B extends Record<string, any>> {
  /** When true, onFrame blocks until current frame's compute + readback completes.
   *  Default false: onFrame receives most recent available data (may be 1 frame behind). */
  sync?: boolean;

  /** Runs once at setup. Declares shader chain and buffer outputs.
   *  Return value maps buffer names to handles for use in onFrame. */
  pipeline: (frame: ProcessorFrame) => B;

  /** Runs every display frame on UI thread.
   *  Receives resolved buffer data and a canvas for Skia draws. */
  onFrame?: (
    frame: RenderFrame,
    buffers: NullableBuffers<B>,
  ) => void;
}

export type FrameProcessor = (frame: ProcessorFrame) => void;

export interface GPUFrameProcessorResult {
  /** Latest processed frame as SkImage — drive a Skia Canvas with this.
   *  The hook owns disposal — do NOT call dispose() on this value. */
  currentFrame: SharedValue<SkImage | null>;
  /** Non-null if shader compilation or pipeline setup failed. */
  error: string | null;
}

export type { SharedValue };

// Global JSI bindings installed by the native pipeline
declare global {
  function __webgpuCamera_nextImage(): SkImage | null;
  function __webgpuCamera_createStream(): CameraStream;
}
```

- [ ] **Step 2: Update index.ts to export new types**

Add the new types to the exports in `packages/react-native-webgpu-camera/src/index.ts`:

```typescript
// react-native-webgpu-camera

// Hooks
export { useCamera } from './useCamera';
export { useGPUFrameProcessor } from './useGPUFrameProcessor';

// Types
export type {
  CameraConfig,
  CameraHandle,
  CameraStream,
  ProcessorFrame,
  RenderFrame,
  FrameProcessor,
  GPUFrameProcessorResult,
  ProcessorConfig,
  BufferHandle,
  NullableBuffers,
  TypedArrayConstructor,
} from './types';

// Native module (advanced usage)
export { default as WebGPUCameraModule } from '../modules/webgpu-camera/src/WebGPUCameraModule';
export type { FrameDimensions } from '../modules/webgpu-camera/src/WebGPUCameraModule';
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/src/types.ts packages/react-native-webgpu-camera/src/index.ts
git commit -m "feat: add multi-pass types — BufferHandle, RenderFrame, ProcessorConfig"
```

---

## Chunk 2: Native Multi-Pass Compute

### Task 2: Update DawnComputePipeline header

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h`

- [ ] **Step 1: Rewrite header with multi-pass C interface**

Replace the entire file. The key changes:
- `dawn_pipeline_setup` replaced by `dawn_pipeline_setup_multipass`
- New structs for buffer specs
- New functions for buffer readback and canvas

```cpp
#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <CoreVideo/CVPixelBuffer.h>

namespace dawn_pipeline {

class DawnComputePipeline {
public:
  DawnComputePipeline();
  ~DawnComputePipeline();

  // Multi-pass setup — compiles N shaders, allocates ping-pong textures,
  // staging buffers for readback, and optional SkSurface for canvas.
  struct BufferSpec {
    int passIndex;     // which pass produces this buffer
    int elementSize;   // bytes per element (e.g. 4 for float32)
    int count;         // number of elements
  };

  bool setup(const std::vector<std::string>& wgslShaders,
             int width, int height,
             const std::vector<BufferSpec>& bufferSpecs,
             bool useCanvas, bool sync);

  // Process a camera frame through all passes
  bool processFrame(CVPixelBufferRef pixelBuffer);

  // Buffer readback — returns pointer to mapped staging buffer data
  // Returns nullptr if no data available yet
  const void* readBuffer(int bufferIndex) const;
  int getBufferByteSize(int bufferIndex) const;

  // Canvas — get/flush SkSurface for Skia draws
  void* getSkSurface(); // returns sk_sp<SkSurface>*
  void flushCanvas();

  // Output
  void* getOutputSkImage(); // returns sk_sp<SkImage>*

  void cleanup();

  int width() const { return _width; }
  int height() const { return _height; }
  std::shared_ptr<std::atomic<bool>> alive() const { return _alive; }

private:
  void cleanupLocked();

  struct Impl;
  Impl* _impl = nullptr;
  std::mutex _mutex;
  int _width = 0;
  int _height = 0;
  std::shared_ptr<std::atomic<bool>> _alive;
};

} // namespace dawn_pipeline

// C interface for Swift/ObjC bridge
typedef void* DawnComputePipelineRef;

extern "C" {
  DawnComputePipelineRef dawn_pipeline_create();
  void dawn_pipeline_destroy(DawnComputePipelineRef ref);

  // Multi-pass setup
  // shaders: array of WGSL strings (null-terminated C strings)
  // shaderCount: number of shaders
  // bufferSpecs: flat array of [passIndex, elementSize, count] triples
  // bufferCount: number of buffers
  bool dawn_pipeline_setup_multipass(
    DawnComputePipelineRef ref,
    const char** shaders, int shaderCount,
    int width, int height,
    const int* bufferSpecs, int bufferCount,
    bool useCanvas, bool sync);

  bool dawn_pipeline_process_frame(DawnComputePipelineRef ref,
                                    CVPixelBufferRef pixelBuffer);

  // Buffer readback
  const void* dawn_pipeline_read_buffer(DawnComputePipelineRef ref, int index);
  int dawn_pipeline_get_buffer_byte_size(DawnComputePipelineRef ref, int index);

  // Canvas
  void* dawn_pipeline_get_sk_surface(DawnComputePipelineRef ref);
  void dawn_pipeline_flush_canvas(DawnComputePipelineRef ref);

  // Output
  void* dawn_pipeline_get_output_image(DawnComputePipelineRef ref);

  void dawn_pipeline_cleanup(DawnComputePipelineRef ref);

  // JSI installation
  void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime);
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h
git commit -m "feat: multi-pass DawnComputePipeline header with buffer readback and canvas"
```

---

### Task 3: Rewrite DawnComputePipeline implementation — Impl struct and setup

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

This is the largest task. The implementation file needs a complete rewrite of the Impl struct, setup(), processFrame(), and CameraStreamHostObject. Due to the size, this task covers the Impl struct and setup(). Task 4 covers processFrame(). Task 5 covers buffer readback, canvas, and CameraStreamHostObject.

- [ ] **Step 1: Rewrite file — includes, Impl struct, constructor/destructor, setup**

Replace the entire file. Here is the first section (includes through setup):

```cpp
#include "DawnComputePipeline.h"

// Skia / Dawn
#include "RNDawnContext.h"
#include "RNSkPlatformContext.h"

#include "include/core/SkImage.h"
#include "include/core/SkSurface.h"
#include "include/core/SkCanvas.h"
#include "include/gpu/graphite/BackendTexture.h"
#include "include/gpu/graphite/Surface.h"

// JSI
#include <jsi/jsi.h>
#include "JsiSkImage.h"
#include "JsiSkCanvas.h"
#include "RNSkiaModule.h"

#include <CoreVideo/CoreVideo.h>
#include <vector>

namespace dawn_pipeline {

// --- Staging buffer for GPU→CPU readback ---
struct StagingBuffer {
  wgpu::Buffer gpuBuffer;       // storage buffer bound to compute shader
  wgpu::Buffer staging[2];      // double-buffered staging for async map
  int frameIndex = 0;           // alternates 0/1
  int byteSize = 0;             // total byte size of buffer
  int elementSize = 0;          // bytes per element
  int count = 0;                // number of elements
  int passIndex = 0;            // which pass writes to this buffer
  std::atomic<bool> mapped[2] = {false, false};
  const void* mappedData[2] = {nullptr, nullptr};
};

// --- Per-pass pipeline state ---
struct PassState {
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout bindGroupLayout;
  bool hasOutputBuffer = false;
  int bufferIndex = -1; // index into StagingBuffer array
};

struct DawnComputePipeline::Impl {
  wgpu::Device device;

  // Shader passes
  std::vector<PassState> passes;

  // Ping-pong textures (RGBA8Unorm)
  wgpu::Texture texA;
  wgpu::Texture texB;

  // Buffer readback
  std::vector<StagingBuffer> buffers;
  bool syncMode = false;

  // Canvas (SkSurface backed by output texture)
  bool useCanvas = false;
  sk_sp<SkSurface> surface;

  // Final output — finalTex points to whichever ping-pong texture the last pass wrote to
  wgpu::Texture* finalTex = nullptr;
  sk_sp<SkImage> outputImage;
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

  // --- Create ping-pong textures ---
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

  // --- Compile shader passes ---
  for (size_t i = 0; i < wgslShaders.size(); i++) {
    PassState pass;

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = wgslShaders[i].c_str();

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc;

    auto shaderModule = _impl->device.CreateShaderModule(&smDesc);
    if (!shaderModule) {
      cleanupLocked();
      return false;
    }

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";

    pass.pipeline = _impl->device.CreateComputePipeline(&cpDesc);
    if (!pass.pipeline) {
      cleanupLocked();
      return false;
    }

    pass.bindGroupLayout = pass.pipeline.GetBindGroupLayout(0);
    _impl->passes.push_back(std::move(pass));
  }

  // --- Create staging buffers for readback ---
  _impl->buffers.resize(bufferSpecs.size());
  for (size_t i = 0; i < bufferSpecs.size(); i++) {
    auto& spec = bufferSpecs[i];
    auto& sb = _impl->buffers[i];

    sb.passIndex = spec.passIndex;
    sb.elementSize = spec.elementSize;
    sb.count = spec.count;
    sb.byteSize = spec.elementSize * spec.count;

    // GPU storage buffer (bound to compute shader at @binding(2))
    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = sb.byteSize;
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufDesc.label = "OutputBuffer";
    sb.gpuBuffer = _impl->device.CreateBuffer(&bufDesc);

    // Double-buffered staging for async map
    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = sb.byteSize;
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    stagingDesc.label = "StagingA";
    sb.staging[0] = _impl->device.CreateBuffer(&stagingDesc);
    stagingDesc.label = "StagingB";
    sb.staging[1] = _impl->device.CreateBuffer(&stagingDesc);

    // Link pass to buffer
    if (spec.passIndex >= 0 && spec.passIndex < (int)_impl->passes.size()) {
      _impl->passes[spec.passIndex].hasOutputBuffer = true;
      _impl->passes[spec.passIndex].bufferIndex = (int)i;
    }
  }

  // --- Determine final output texture (same logic as processFrame ping-pong) ---
  // After N passes, writeToA toggles N times from true: final write goes to texA if odd, texB if even
  _impl->finalTex = (wgslShaders.size() % 2 != 0) ? &_impl->texA : &_impl->texB;

  // --- Create persistent SkSurface for canvas if needed (spec: created once at setup) ---
  if (useCanvas) {
    skgpu::graphite::BackendTexture backendTex =
      skgpu::graphite::BackendTextures::MakeDawn(_impl->finalTex->Get());
    _impl->surface = SkSurfaces::WrapBackendTexture(
      ctx.getRecorder(), backendTex,
      kRGBA_8888_SkColorType, nullptr, nullptr
    );
  }

  return true;
}
```

This step continues in Task 4 (processFrame) and Task 5 (readback, canvas, JSI).

- [ ] **Step 2: Commit partial implementation**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: multi-pass DawnComputePipeline — Impl struct and setup"
```

---

### Task 4: Implement processFrame for multi-pass

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

- [ ] **Step 1: Add processFrame method**

Append after setup() in the same file:

```cpp
bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !pixelBuffer) return false;

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto& device = _impl->device;

  // --- Import camera frame as Dawn texture (zero-copy via IOSurface) ---
  IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
  if (!ioSurface) return false;

  wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
  ioDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor sharedDesc{};
  sharedDesc.nextInChain = &ioDesc;

  auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
  if (!sharedMemory) return false;

  // Create input texture from shared memory (BGRA8Unorm — native camera format)
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

  // BeginAccess must be called AFTER CreateTexture
  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
  beginDesc.initialized = true;
  sharedMemory.BeginAccess(inputTexture, &beginDesc);

  // Create input texture view with RGBA8Unorm format override for pass 0
  // This handles the BGRA→RGBA swizzle transparently
  wgpu::TextureViewDescriptor inputViewDesc{};
  inputViewDesc.format = wgpu::TextureFormat::RGBA8Unorm;
  auto inputView = inputTexture.CreateView(&inputViewDesc);

  // --- Command encoder for all compute passes ---
  auto encoder = device.CreateCommandEncoder();

  // Ping-pong state: track which texture is "read" and which is "write"
  // Pass 0: read = camera input, write = texA
  // Pass 1: read = texA, write = texB
  // Pass 2: read = texB, write = texA
  // ...

  wgpu::TextureView readView = inputView; // first pass reads camera
  bool writeToA = true; // first pass writes to A

  for (size_t i = 0; i < _impl->passes.size(); i++) {
    auto& pass = _impl->passes[i];

    wgpu::Texture& writeTex = writeToA ? _impl->texA : _impl->texB;
    auto writeView = writeTex.CreateView();

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries;

    // @binding(0) = input texture
    wgpu::BindGroupEntry entry0{};
    entry0.binding = 0;
    entry0.textureView = readView;
    entries.push_back(entry0);

    // @binding(1) = output texture
    wgpu::BindGroupEntry entry1{};
    entry1.binding = 1;
    entry1.textureView = writeView;
    entries.push_back(entry1);

    // @binding(2) = output storage buffer (optional)
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

    // Dispatch compute
    auto computePass = encoder.BeginComputePass();
    computePass.SetPipeline(pass.pipeline);
    computePass.SetBindGroup(0, bindGroup);
    computePass.DispatchWorkgroups(
      (_width + 15) / 16,
      (_height + 15) / 16
    );
    computePass.End();

    // Next pass reads from what we just wrote
    readView = writeView;
    writeToA = !writeToA;
  }

  // --- Copy output buffers to staging for readback ---
  for (auto& sb : _impl->buffers) {
    int stagingIdx = sb.frameIndex % 2;

    // Unmap previous mapping if still mapped
    if (sb.mapped[stagingIdx].load()) {
      sb.staging[stagingIdx].Unmap();
      sb.mapped[stagingIdx].store(false);
      sb.mappedData[stagingIdx] = nullptr;
    }

    encoder.CopyBufferToBuffer(
      sb.gpuBuffer, 0,
      sb.staging[stagingIdx], 0,
      sb.byteSize
    );
  }

  // --- Submit ---
  auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  // --- Async map staging buffers ---
  for (auto& sb : _impl->buffers) {
    int stagingIdx = sb.frameIndex % 2;

    sb.staging[stagingIdx].MapAsync(
      wgpu::MapMode::Read, 0, sb.byteSize,
      wgpu::CallbackMode::AllowProcessEvents,
      [&sb, stagingIdx](wgpu::MapAsyncStatus status, wgpu::StringView) {
        if (status == wgpu::MapAsyncStatus::Success) {
          sb.mappedData[stagingIdx] = sb.staging[stagingIdx].GetConstMappedRange(0, sb.byteSize);
          sb.mapped[stagingIdx].store(true);
        }
      }
    );

    sb.frameIndex++;
  }

  // If sync mode, tick the device until all maps complete
  if (_impl->syncMode) {
    for (auto& sb : _impl->buffers) {
      int stagingIdx = (sb.frameIndex - 1) % 2;
      while (!sb.mapped[stagingIdx].load()) {
        device.Tick();
      }
    }
  }

  // --- Wrap final output as SkImage ---
  // The last pass wrote to the texture that !writeToA points to
  // (writeToA was toggled after the last pass, so the actual output is the opposite)
  _impl->finalTex = writeToA ? &_impl->texB : &_impl->texA;
  _impl->outputImage = ctx.MakeImageFromTexture(
    *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm
  );

  // --- Cleanup IOSurface access ---
  wgpu::SharedTextureMemoryEndAccessState endState{};
  sharedMemory.EndAccess(inputTexture, &endState);

  return true;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: multi-pass processFrame with ping-pong textures and buffer readback"
```

---

### Task 5: Buffer readback, canvas, cleanup, C interface, and CameraStreamHostObject

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

- [ ] **Step 1: Add readback, canvas, cleanup, and C interface methods**

Append after processFrame():

```cpp
const void* DawnComputePipeline::readBuffer(int bufferIndex) const {
  // No lock needed — we read from the previously-mapped staging buffer
  // which is stable while the current frame is being computed
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return nullptr;

  auto& sb = _impl->buffers[bufferIndex];
  // Read from the PREVIOUS frame's staging buffer (already mapped).
  // frameIndex was incremented after the last processFrame, so frameIndex % 2
  // points to the next write target — which is the one mapped two frames ago.
  int readIdx = sb.frameIndex % 2;
  if (!sb.mapped[readIdx].load()) return nullptr;
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
  auto recording = ctx.getRecorder()->snap();
  if (recording) {
    skgpu::graphite::InsertRecordingInfo insertInfo{};
    insertInfo.fRecording = recording.get();
    ctx.fGraphiteContext->insertRecording(insertInfo);
    ctx.fGraphiteContext->submit(skgpu::graphite::SyncToCpu::kNo);
  }

  // Re-wrap final texture as SkImage to include canvas draws
  if (_impl->finalTex) {
    _impl->outputImage = ctx.MakeImageFromTexture(
      *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA8Unorm
    );
  }
}

void* DawnComputePipeline::getOutputSkImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;
  return &_impl->outputImage;
}

void DawnComputePipeline::cleanup() {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

void DawnComputePipeline::cleanupLocked() {
  if (!_impl) return;

  // Unmap any mapped staging buffers
  for (auto& sb : _impl->buffers) {
    for (int j = 0; j < 2; j++) {
      if (sb.mapped[j].load()) {
        sb.staging[j].Unmap();
        sb.mapped[j].store(false);
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

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref,
                                  CVPixelBufferRef pixelBuffer) {
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

} // extern "C"
```

- [ ] **Step 2: Add updated CameraStreamHostObject and JSI installation**

Append after the C interface:

```cpp
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

          // Create ArrayBuffer with copy of mapped data
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

extern "C" {

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  auto& runtime = *static_cast<facebook::jsi::Runtime*>(jsiRuntime);

  // Get Skia platform context for creating JsiSkImage / JsiSkCanvas
  auto skiaModule = RNSkia::RNSkiaModule::getSkiaModule(runtime);
  auto platformContext = skiaModule->getSkManager()->getPlatformContext();

  auto alive = pipeline->alive();

  // __webgpuCamera_nextImage() — rAF fallback path
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

      auto* imgPtr = static_cast<sk_sp<SkImage>*>(
        pipeline->getOutputSkImage());
      if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

      auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
        platformContext, *imgPtr);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_nextImage", std::move(nextImageFn));

  // __webgpuCamera_createStream() — returns CameraStreamHostObject
  auto createStreamFn = facebook::jsi::Function::createFromHostFunction(
    runtime,
    facebook::jsi::PropNameID::forAscii(runtime, "__webgpuCamera_createStream"),
    0,
    [pipeline, alive, platformContext](
      facebook::jsi::Runtime& rt,
      const facebook::jsi::Value&,
      const facebook::jsi::Value*,
      size_t) -> facebook::jsi::Value {
      auto hostObj = std::make_shared<CameraStreamHostObject>(
        pipeline, alive, platformContext);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_createStream", std::move(createStreamFn));
}

} // extern "C"
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: buffer readback, canvas, CameraStreamHostObject, C interface, JSI"
```

> **Note:** Canvas draws *between* shader passes (SkPicture replay, command encoder splits) are described in the spec but deferred from this initial implementation. The current processFrame submits all compute passes in a single command encoder — optimal performance. Between-pass canvas support will be added in a follow-up if needed after validating multi-pass and onFrame canvas on device.

---

## Chunk 3: Bridge, Swift Module, and Hook

### Task 6: Update DawnPipelineBridge

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`

- [ ] **Step 1: Update bridge header**

Replace `DawnPipelineBridge.h`:

```objc
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>

@interface DawnPipelineBridge : NSObject

- (BOOL)setupMultiPassWithShaders:(nonnull NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(nonnull NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync;

- (BOOL)processFrame:(nonnull CVPixelBufferRef)pixelBuffer;
- (void)cleanup;
- (void)installJSIBindings:(nonnull id)expoRuntime;

@end
```

- [ ] **Step 2: Update bridge implementation**

Replace `DawnPipelineBridge.mm`:

```objcpp
#import "DawnPipelineBridge.h"
#include "DawnComputePipeline.h"
#import <ExpoModulesJSI/EXJavaScriptRuntime.h>

@implementation DawnPipelineBridge {
  DawnComputePipelineRef _pipeline;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _pipeline = dawn_pipeline_create();
  }
  return self;
}

- (void)dealloc {
  if (_pipeline) {
    dawn_pipeline_destroy(_pipeline);
    _pipeline = nullptr;
  }
}

- (BOOL)setupMultiPassWithShaders:(NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync {
  if (!_pipeline) return NO;

  // Convert NSArray<NSString*> to C string array
  int shaderCount = (int)shaders.count;
  std::vector<const char*> cShaders(shaderCount);
  std::vector<std::string> shaderStorage(shaderCount); // keep strings alive
  for (int i = 0; i < shaderCount; i++) {
    shaderStorage[i] = [shaders[i] UTF8String];
    cShaders[i] = shaderStorage[i].c_str();
  }

  // Convert buffer specs to flat int array [passIndex, elementSize, count, ...]
  int bufferCount = (int)bufferSpecs.count;
  std::vector<int> flatSpecs(bufferCount * 3);
  for (int i = 0; i < bufferCount; i++) {
    NSArray<NSNumber *> *spec = bufferSpecs[i];
    flatSpecs[i * 3 + 0] = [spec[0] intValue]; // passIndex
    flatSpecs[i * 3 + 1] = [spec[1] intValue]; // elementSize
    flatSpecs[i * 3 + 2] = [spec[2] intValue]; // count
  }

  return dawn_pipeline_setup_multipass(
    _pipeline,
    cShaders.data(), shaderCount,
    width, height,
    flatSpecs.data(), bufferCount,
    useCanvas, sync
  );
}

- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer {
  if (!_pipeline) return NO;
  return dawn_pipeline_process_frame(_pipeline, pixelBuffer);
}

- (void)cleanup {
  if (!_pipeline) return;
  dawn_pipeline_cleanup(_pipeline);
}

- (void)installJSIBindings:(id)expoRuntime {
  if (!_pipeline) return;
  EXJavaScriptRuntime *runtime = (EXJavaScriptRuntime *)expoRuntime;
  facebook::jsi::Runtime *jsiRuntime = [runtime get];
  if (!jsiRuntime) return;
  dawn_pipeline_install_jsi(_pipeline, jsiRuntime);
}

@end
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h \
      packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git commit -m "feat: DawnPipelineBridge updated for multi-pass setup"
```

---

### Task 7: Update WebGPUCameraModule.swift

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Replace setupComputePipeline with setupMultiPassPipeline**

In `WebGPUCameraModule.swift`, **delete** the existing `Function("setupComputePipeline")` block (lines 78-93) and replace it with:

```swift
Function("setupMultiPassPipeline") { (config: [String: Any]) -> Bool in
  guard let shaders = config["shaders"] as? [String],
        let width = config["width"] as? Int,
        let height = config["height"] as? Int else {
    print("[WebGPUCamera] setupMultiPassPipeline: missing required config fields")
    return false
  }

  let bufferSpecsRaw = config["buffers"] as? [[Int]] ?? []
  let useCanvas = config["useCanvas"] as? Bool ?? false
  let sync = config["sync"] as? Bool ?? false

  // Convert buffer specs to NSArray<NSArray<NSNumber>>
  let bufferSpecs = bufferSpecsRaw.map { spec in
    spec.map { NSNumber(value: $0) }
  }

  // Clean up any existing bridge before creating a new one
  self.dawnBridge?.cleanup()
  self.dawnBridge = nil

  let bridge = DawnPipelineBridge()
  let ok = bridge.setupMultiPass(
    withShaders: shaders,
    width: Int32(width),
    height: Int32(height),
    bufferSpecs: bufferSpecs,
    useCanvas: useCanvas,
    sync: sync
  )

  if ok {
    self.dawnBridge = bridge
    self.computeSetup = true
    if let runtime = try? self.appContext?.runtime {
      bridge.installJSIBindings(runtime)
    } else {
      print("[WebGPUCamera] WARNING: Could not access runtime for JSI bindings")
    }
    print("[WebGPUCamera] Multi-pass pipeline setup OK: \(shaders.count) passes, \(width)x\(height)")
  } else {
    print("[WebGPUCamera] Multi-pass pipeline setup FAILED")
  }
  return ok
}
```

Also update `cleanupComputePipeline` (lines 95-100) to rename for clarity — but since the cleanup interface hasn't changed, this can stay as-is. The hook will call the same `cleanupComputePipeline` function name.

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: WebGPUCameraModule.setupMultiPassPipeline replacing single-shader setup"
```

---

### Task 8: Update useGPUFrameProcessor hook

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`

- [ ] **Step 1: Rewrite useGPUFrameProcessor with multi-pass support**

Replace the entire file:

```typescript
import { useEffect, useState } from 'react';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import type { SkImage } from '@shopify/react-native-skia';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type {
  CameraHandle,
  CameraStream,
  ProcessorFrame,
  ProcessorConfig,
  FrameProcessor,
  GPUFrameProcessorResult,
  TypedArrayConstructor,
} from './types';

/** Internal: collected shader/buffer info from the capture proxy */
interface CapturedPass {
  wgsl: string;
  buffer?: {
    output: TypedArrayConstructor;
    count: number;
  };
}

/** Buffer metadata for resolving handles in the worklet */
interface BufferMeta {
  name: string;
  ctor: TypedArrayConstructor;
}

/**
 * Runs the pipeline callback with a capture proxy to collect shader chain
 * and buffer declarations. Returns the captured config.
 */
function capturePipeline<B extends Record<string, any>>(
  pipelineFn: (frame: ProcessorFrame) => B,
  width: number,
  height: number,
): { passes: CapturedPass[]; bufferMetas: BufferMeta[]; hasCanvas: boolean } {
  const passes: CapturedPass[] = [];
  const bufferMetas: BufferMeta[] = [];
  let hasCanvas = false;

  const captureFrame: ProcessorFrame = {
    runShader(wgsl: string, options?: { output: TypedArrayConstructor; count: number }) {
      const pass: CapturedPass = { wgsl };
      if (options) {
        pass.buffer = { output: options.output, count: options.count };
        // Placeholder name — will be replaced with user-facing key after capture
        bufferMetas.push({ name: `__buf_${bufferMetas.length}`, ctor: options.output });
      }
      passes.push(pass);
      return {} as any;
    },
    canvas: new Proxy({} as any, {
      get(_, prop) {
        // Only flag actual draw calls, not property existence checks
        if (typeof prop === 'string' && prop.startsWith('draw')) {
          hasCanvas = true;
        }
        return () => {};
      },
    }),
    width,
    height,
  };

  let returnedHandles: B | undefined;
  try {
    returnedHandles = pipelineFn(captureFrame);
  } catch {
    // Processor may reference worklet-only APIs during capture — safe to ignore
  }

  // Map returned handle keys to buffer indices
  // pipeline returns { detections: handle } — keys become buffer names for onFrame
  if (returnedHandles) {
    const keys = Object.keys(returnedHandles);
    for (let i = 0; i < Math.min(keys.length, bufferMetas.length); i++) {
      bufferMetas[i].name = keys[i];
    }
  }

  return { passes, bufferMetas, hasCanvas };
}

/**
 * Build the native pipeline config from captured passes.
 */
function buildNativeConfig(
  passes: CapturedPass[],
  width: number,
  height: number,
  useCanvas: boolean,
  sync: boolean,
) {
  const shaders = passes.map((p) => p.wgsl);
  const buffers: [number, number, number][] = [];

  passes.forEach((pass, passIndex) => {
    if (pass.buffer) {
      const elementSize = pass.buffer.output.BYTES_PER_ELEMENT ?? 4;
      buffers.push([passIndex, elementSize, pass.buffer.count]);
    }
  });

  return { shaders, width, height, buffers, useCanvas, sync };
}

/**
 * Typed array constructor lookup — these are available as globals in the worklet runtime.
 * We store the BYTES_PER_ELEMENT at setup and use it to select the right constructor at runtime.
 */
function wrapBuffer(arrayBuffer: ArrayBuffer, bytesPerElement: number): ArrayBufferView {
  'worklet';
  switch (bytesPerElement) {
    case 8: return new Float64Array(arrayBuffer);
    case 4: return new Float32Array(arrayBuffer); // Float32Array, Int32Array, Uint32Array all 4 bytes
    case 2: return new Uint16Array(arrayBuffer);
    case 1: return new Uint8Array(arrayBuffer);
    default: return new Float32Array(arrayBuffer);
  }
}

export function useGPUFrameProcessor(
  camera: CameraHandle,
  processorOrConfig: FrameProcessor | ProcessorConfig<any>,
): GPUFrameProcessorResult {
  const [error, setError] = useState<string | null>(null);
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImage | null>(null);

  // Determine form at module scope (not in worklet)
  const isObjectForm = typeof processorOrConfig !== 'function' && 'pipeline' in processorOrConfig;
  const onFrameFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).onFrame
    : undefined;
  const sync = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).sync ?? false
    : false;
  const pipelineFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).pipeline
    : (processorOrConfig as FrameProcessor);

  // Buffer metadata shared with the worklet via shared values
  // (Reanimated shared values are accessible from worklets, unlike refs)
  const bufferCount = useSharedValue(0);
  const bufferNames = useSharedValue<string[]>([]);
  const bufferBytesPerElement = useSharedValue<number[]>([]);
  const hasOnFrame = useSharedValue(false);

  // Setup compute pipeline when camera is ready
  useEffect(() => {
    if (!camera.isReady) return;

    // Capture shader chain and buffer declarations
    const { passes, bufferMetas, hasCanvas } = capturePipeline(
      pipelineFn as (frame: ProcessorFrame) => any,
      camera.width,
      camera.height,
    );

    if (passes.length === 0) {
      setError('No shader provided — call frame.runShader(wgslCode) in your processor');
      return;
    }

    // Store buffer metadata in shared values for worklet access
    bufferCount.value = bufferMetas.length;
    bufferNames.value = bufferMetas.map((m) => m.name);
    bufferBytesPerElement.value = bufferMetas.map((m) => m.ctor.BYTES_PER_ELEMENT ?? 4);
    hasOnFrame.value = !!onFrameFn;

    // Determine if canvas is used (between passes OR in onFrame)
    const useCanvas = hasCanvas || (isObjectForm && !!onFrameFn);

    // Build and send native config
    const nativeConfig = buildNativeConfig(
      passes, camera.width, camera.height, useCanvas, sync,
    );

    const ok = WebGPUCameraModule.setupMultiPassPipeline(nativeConfig);
    if (!ok) {
      setError('Multi-pass pipeline setup failed');
      return;
    }
    setError(null);

    // Create stream host object — shared across Reanimated runtimes
    stream.value = globalThis.__webgpuCamera_createStream();

    return () => {
      currentFrame.value?.dispose();
      currentFrame.value = null;
      stream.value = null;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, [camera.isReady, camera.width, camera.height, camera.fps]);

  // Frame callback — runs on Reanimated UI thread every display frame
  useFrameCallback(() => {
    'worklet';
    const s = stream.value;
    if (!s) return;

    const img = s.nextImage();
    if (!img) return;

    if (hasOnFrame.value && onFrameFn) {
      // --- Object form with onFrame ---

      // Resolve buffers by index, using shared value metadata
      const count = bufferCount.value;
      const names = bufferNames.value;
      const bpe = bufferBytesPerElement.value;
      const buffers: Record<string, any> = {};

      for (let i = 0; i < count; i++) {
        const buf = s.readBuffer(i);
        buffers[names[i]] = buf !== null ? wrapBuffer(buf, bpe[i]) : null;
      }

      // Get canvas for Skia draws on the output texture
      const canvas = s.getCanvas();

      const renderFrame = {
        canvas: canvas!,
        width: img.width(),
        height: img.height(),
      };

      // Call user's onFrame
      onFrameFn(renderFrame, buffers as any);

      // If canvas was used, flush draws and get composited output
      if (canvas) {
        s.flushCanvas();
        // flushCanvas re-wraps the output texture as a new SkImage.
        // nextImage now returns the composited result (compute + canvas draws).
        const composited = s.nextImage();
        if (composited) {
          currentFrame.value?.dispose();
          currentFrame.value = composited;
          // Dispose the pre-canvas image (we used it for dimensions only)
          img.dispose();
        } else {
          // Fallback: no composited image, use raw compute output
          currentFrame.value?.dispose();
          currentFrame.value = img;
        }
      } else {
        // No canvas draws — use the compute output directly
        currentFrame.value?.dispose();
        currentFrame.value = img;
      }
    } else {
      // --- Shorthand form (no onFrame) ---
      currentFrame.value?.dispose();
      currentFrame.value = img;
    }
  });

  return { currentFrame, error };
}
```

**Key design decisions in this implementation:**
- Buffer metadata (names, BYTES_PER_ELEMENT) is stored in Reanimated shared values, not React refs — shared values are accessible from worklets, refs are not.
- The `onFrameFn` reference is captured in the closure at component scope (stable reference). If the user passes an inline function, they should memoize it or accept re-setup.
- `wrapBuffer` is a worklet function that selects the typed array constructor based on `BYTES_PER_ELEMENT`. This avoids passing constructor references across runtimes.
- After `flushCanvas()`, the native side re-wraps the final texture as a new SkImage, so the subsequent `nextImage()` returns the composited result. If no canvas was used, we skip the flush and use the raw compute output.
- Canvas-between-passes (SkPicture replay) is deferred — see note after Task 5.

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: useGPUFrameProcessor multi-pass, object form, onFrame worklet"
```

---

## Chunk 4: Example App and Verification

### Task 9: Update example app with multi-pass demo

**Files:**
- Modify: `apps/example/src/app/index.tsx`

- [ ] **Step 1: Update example to demonstrate multi-pass**

Replace the example app. Add a third shader option that chains Sobel + Sobel Color (multi-pass):

```typescript
import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage } from '@shopify/react-native-skia';
import { useCamera, useGPUFrameProcessor } from 'react-native-webgpu-camera';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import { SOBEL_COLOR_WGSL } from '@/shaders/sobel-color.wgsl';

const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;

const SHADERS = [
  { name: 'Sobel', wgsl: [SOBEL_WGSL] },
  { name: 'Sobel Color', wgsl: [SOBEL_COLOR_WGSL] },
  { name: 'Multi-pass', wgsl: [SOBEL_WGSL, SOBEL_COLOR_WGSL] },
] as const;

function CameraPreview({ shaderChain }: { shaderChain: readonly string[] }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    for (const wgsl of shaderChain) {
      frame.runShader(wgsl);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <Group transform={[
          { translateX: screenW },
          { rotate: Math.PI / 2 },
        ]}>
          <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Pipeline running' : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [shaderIndex, setShaderIndex] = useState(0);
  const shader = SHADERS[shaderIndex];

  return (
    <View style={styles.container}>
      {isRunning && <CameraPreview key={shader.name} shaderChain={shader.wgsl} />}

      <View style={styles.controls}>
        {isRunning && (
          <Pressable
            style={styles.button}
            onPress={() => setShaderIndex((i) => (i + 1) % SHADERS.length)}
          >
            <Text style={styles.buttonText}>{shader.name}</Text>
          </Pressable>
        )}

        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={() => setIsRunning(!isRunning)}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  statusBar: {
    position: 'absolute', top: 44, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 4, padding: 8,
  },
  statusText: { color: '#aaa', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'center', gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 24,
    paddingHorizontal: 24, paddingVertical: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: { backgroundColor: 'rgba(255,80,80,0.4)', borderColor: 'rgba(255,80,80,0.6)' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
```

- [ ] **Step 2: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: example app with multi-pass shader chaining"
```

---

### Task 10: Build and test on device

- [ ] **Step 1: EAS Build**

This requires a native rebuild since we changed C++/ObjC/Swift files:

```bash
cd apps/example
eas build --platform ios --profile development
```

Expected: Build completes successfully. Watch the build log for:
- No compilation errors in `DawnComputePipeline.mm` or `DawnPipelineBridge.mm`
- No linker errors for missing Dawn/Skia symbols
- Output is a downloadable .ipa or link in the EAS dashboard

- [ ] **Step 2: Test on device**

Install the build on iPhone 16 Pro. Test:
1. Single shader (Sobel) — should work as before
2. Single shader (Sobel Color) — should work as before
3. Multi-pass (Sobel → Sobel Color) — validates ping-pong textures
4. Shader switching — validates cleanup and re-setup
5. Check Xcode GPU profiler or FPS counter to confirm 120fps is maintained in multi-pass mode

Expected: all three modes render correctly at 120fps.

- [ ] **Step 3: Fix any issues found during device testing**

Commit fixes with specific file paths and descriptive messages — do not use `git add -A`.
