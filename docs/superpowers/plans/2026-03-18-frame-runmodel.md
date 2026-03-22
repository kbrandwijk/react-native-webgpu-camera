# frame.runModel() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `frame.runModel()` to the GPU camera pipeline — runs ONNX models on camera frames using the shared Dawn device, async by default, zero CPU pixel copies.

**Architecture:** Model declared as `GPUResource.model(path, opts)` in `resources` block. At setup, native side creates an ONNX `InferenceSession` with WebGPU EP on the shared Dawn device, compiles a resize+normalize compute shader, and pre-allocates GPU buffers. Per-frame, inference runs on a background thread; the pipeline always gets the latest available result (never blocks). Output is a texture handle usable in subsequent `frame.runShader()` passes.

**Tech Stack:** ONNX Runtime (WebGPU EP), Dawn (shared device from Skia Graphite), C++ compute shaders for preprocessing

---

## File Structure

### New Files
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.h` — C++ class: manages ONNX session, resize shader, async inference thread, output texture cache
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.mm` — Implementation

### Modified Files
- `packages/react-native-webgpu-camera/src/GPUResource.ts` — Add `model()` resource constructor
- `packages/react-native-webgpu-camera/src/types.ts` — Add `runModel()` to `ProcessorFrame`, add `ModelResourceHandle` type
- `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` — Handle model resources in capture proxy, send model specs to native, add `runModel` to capture frame
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` — Add model setup/run methods, model state in Impl
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` — Integrate ModelRunner: setup at pipeline init, feed frames, provide latest output texture to subsequent passes
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h` — Add model specs parameter
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm` — Convert model specs from NSDictionary to C++ structs
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` — Pass model specs through to bridge

### Example App
- `apps/example/src/app/index.tsx` — Add depth estimation example using `frame.runModel()`

---

## Task 0: Wire ONNX Runtime headers and framework into the Expo module build

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/webgpu-camera.podspec`

The ModelRunner needs ONNX Runtime headers (`onnxruntime_cxx_api.h`) and the ORT framework at link time. The `onnxruntime-react-native` package already has the vendored framework at `packages/onnxruntime/js/react_native/libs/ios/onnxruntime.framework`. We need to make it available to the webgpu-camera module.

- [ ] **Step 1: Add ONNX Runtime headers to the podspec**

In `webgpu-camera.podspec`, add header search paths for the ONNX Runtime framework headers and add a dependency on onnxruntime-react-native:

```ruby
s.dependency 'onnxruntime-react-native'

# Or if the dependency approach doesn't expose headers, add them directly:
ort_fw = File.join(__dir__, '..', '..', '..', '..', 'onnxruntime', 'js', 'react_native', 'libs', 'ios', 'onnxruntime.framework')
s.pod_target_xcconfig = {
  'HEADER_SEARCH_PATHS' => "\"#{ort_fw}/Headers\"",
  'OTHER_LDFLAGS' => '-framework onnxruntime',
}
```

The exact approach depends on how the existing podspec is structured. Check the current podspec and follow its patterns.

- [ ] **Step 2: Verify ORT headers are available**

Create a minimal test — add `#include "onnxruntime_cxx_api.h"` to ModelRunner.h (created in Task 3) and verify it compiles. If the header search path is wrong, adjust in the podspec.

- [ ] **Step 3: Verify linking works**

The onnxruntime framework symbols must be available at link time. Since both `onnxruntime-react-native` and `webgpu-camera` are pods in the same app, CocoaPods should handle transitive linking. If not, add explicit framework linking to the podspec.

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/webgpu-camera.podspec
git commit -m "build: add ONNX Runtime dependency to webgpu-camera podspec"
```

---

## Task 1: Add `GPUResource.model()` to the JS resource system

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/GPUResource.ts`
- Modify: `packages/react-native-webgpu-camera/src/types.ts`

This task adds the JS-side model resource type. No native changes yet — just the type system and resource declaration.

- [ ] **Step 1: Add ModelOptions interface and model() constructor to GPUResource.ts**

In `GPUResource.ts`, add a `ModelOptions` interface and a `model()` factory function that returns a `ResourceHandle<'model'>`:

```typescript
export interface ModelOptions {
  /** Model input shape, e.g. [1, 3, 518, 518]. Inferred from model if omitted. */
  inputShape?: number[];
  /** ImageNet normalization params. Default: ImageNet standard. */
  normalization?: { mean: [number, number, number]; std: [number, number, number] };
  /** When true, inference blocks the pipeline (for small models). Default: false (async). */
  sync?: boolean;
}

/** Model-specific resource handle — extends ResourceHandle with model options */
export interface ModelResourceHandle extends ResourceHandle<'model'> {
  readonly __modelOptions?: ModelOptions;
}

function model(
  pathOrUrl: string,
  options?: ModelOptions,
): ModelResourceHandle {
  return {
    __resourceType: 'model',
    __handle: -1,
    __fileUri: pathOrUrl,
    __modelOptions: options,
  };
}
```

The base `ResourceHandle` interface is NOT modified — model options stay on the dedicated `ModelResourceHandle` type. This preserves type safety for all existing resource types.

Add `model` to the `GPUResource` export object:

```typescript
export const GPUResource = {
  texture3D,
  texture2D: /* existing */,
  storageBuffer,
  cameraDepth,
  model,
};
```

- [ ] **Step 2: Add `runModel()` to ProcessorFrame in types.ts**

Add the `runModel` method to the `ProcessorFrame` interface (import `ModelResourceHandle`):

```typescript
export interface ProcessorFrame {
  // ... existing runShader overloads ...

  /** Run an ONNX model on the current frame. Returns a texture handle with the model output.
   *  Async by default — returns the latest available result (null if not ready yet). */
  runModel(model: ModelResourceHandle): ResourceHandle<'texture2d'> | null;

  canvas: SkCanvas;
  width: number;
  height: number;
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`
Expected: No new errors (existing errors may exist, just verify no regressions from our changes).

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/src/GPUResource.ts packages/react-native-webgpu-camera/src/types.ts
git commit -m "feat: add GPUResource.model() type and ProcessorFrame.runModel() signature"
```

---

## Task 2: Handle model resources in the capture proxy and native config

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`

This task makes the capture proxy understand `runModel()` calls and include model specs in the native config sent to the bridge.

- [ ] **Step 1: Add CapturedModel interface and modelSpecs collection**

At the top of `useGPUFrameProcessor.ts`, add:

```typescript
/** A model spec to send to native for session creation */
interface CapturedModel {
  path: string;
  inputShape?: number[];
  normalization?: { mean: [number, number, number]; std: [number, number, number] };
  sync: boolean;
  /** Index in the pipeline where this model runs (after pass offset) */
  pipelineIndex: number;
}
```

- [ ] **Step 2: Add `runModel` to the capture proxy**

In `capturePipeline()`, add a `capturedModels: CapturedModel[]` array alongside existing `passes` and `capturedResources`.

Update the return type (line 64-70 of `useGPUFrameProcessor.ts`):

```typescript
): {
  passes: CapturedPass[];
  bufferMetas: BufferMeta[];
  hasCanvas: boolean;
  capturedResources: CapturedResource[];
  capturedModels: CapturedModel[];  // NEW
}
```

Update the return statement and the destructuring at the call site (line 424):

```typescript
const { passes, bufferMetas, hasCanvas, capturedResources, capturedModels } = capturePipeline(...)
```

Add `runModel` to the `captureFrame` proxy object. **Important**: The model output check (`src.modelIndex !== undefined`) must come **before** the existing `src.isTexture` check in the `outputHandleMap` branch, otherwise model outputs would be handled as generic texture outputs:

```typescript
runModel(modelHandle: ResourceHandle<'model'>): ResourceHandle<'texture2d'> | null {
  if (!isResourceHandle(modelHandle) || modelHandle.__resourceType !== 'model') {
    return null;
  }

  const modelIndex = capturedModels.length;
  const pipelineIndex = passes.length; // position in the pass chain

  capturedModels.push({
    path: modelHandle.__fileUri ?? '',
    inputShape: modelHandle.__modelOptions?.inputShape,
    normalization: modelHandle.__modelOptions?.normalization,
    sync: modelHandle.__modelOptions?.sync ?? false,
    pipelineIndex,
  });

  // Create an output handle representing the model's output texture
  const outputHandle = {
    __resourceType: 'texture2d' as const,
    __handle: -1,
    __modelOutput: modelIndex,
  } as any as ResourceHandle<'texture2d'>;

  // Track this handle so subsequent runShader calls can reference it
  outputHandleMap.set(outputHandle, {
    passIndex: -1, // special: model output, not a shader pass
    bufferIndex: -1,
    isTexture: true,
    modelIndex,
  });

  return outputHandle;
},
```

Update the `outputHandleMap` value type to include optional `modelIndex`:
```typescript
const outputHandleMap = new Map<any, {
  passIndex: number;
  bufferIndex: number;
  isTexture: boolean;
  modelIndex?: number;
}>();
```

When a subsequent `runShader()` references a model output handle in its `inputs`, detect it via `outputHandleMap` entry having `modelIndex !== undefined` and emit a special input binding:

```typescript
} else if (outputHandleMap.has(handle)) {
  const src = outputHandleMap.get(handle)!;
  if (src.modelIndex !== undefined) {
    // Model output texture
    pass.inputs.push({
      name,
      bindingIndex: nextBinding,
      type: 'texture2d',
      modelOutput: src.modelIndex,  // new field
    });
    nextBinding++;
    pass.inputs.push({
      name: `${name}_sampler`,
      bindingIndex: nextBinding,
      type: 'sampler',
      modelOutput: src.modelIndex,
    });
    nextBinding++;
  } else if (src.isTexture) {
    // existing texture output path...
```

Add `modelOutput?: number` to the `CapturedInput` interface.

- [ ] **Step 3: Include model specs in buildNativeConfig**

Update the `capturePipeline` return type to include `capturedModels`.

Update `buildNativeConfig` to accept and pass through model specs:

```typescript
function buildNativeConfig(
  passes: CapturedPass[],
  width: number,
  height: number,
  useCanvas: boolean,
  sync: boolean,
  capturedResources: CapturedResource[],
  appleLog: boolean,
  capturedModels: CapturedModel[],  // NEW
) {
  // ... existing code ...

  // Build model specs for native
  const models = capturedModels.map(m => ({
    path: m.path,
    inputShape: m.inputShape ?? [],
    normMean: m.normalization?.mean ?? [0.485, 0.456, 0.406],
    normStd: m.normalization?.std ?? [0.229, 0.224, 0.225],
    sync: m.sync,
    pipelineIndex: m.pipelineIndex + passOffset,
  }));

  // Build model-output input bindings (for shaders that consume model output)
  passes.forEach((pass, passIndex) => {
    if (pass.inputs) {
      for (const inp of pass.inputs) {
        if (inp.modelOutput !== undefined) {
          // Add to passInputs with modelOutput marker
          // Will be handled on native side
        }
      }
    }
  });

  return {
    shaders, width, height, buffers, useCanvas, sync, appleLog,
    resources, passInputs, textureOutputPasses, useDepth,
    models,  // NEW
  };
}
```

Update the call site in `useEffect` to pass `capturedModels`.

- [ ] **Step 4: Pass model specs through setupMultiPassPipeline**

In the `useEffect`, update the `WebGPUCameraModule.setupMultiPassPipeline(nativeConfig)` call — the `nativeConfig` now includes `models`. The native module will need to accept this, which is wired in Task 4.

Add logging:
```typescript
if (nativeConfig.models.length > 0) {
  console.log(`[WebGPUCamera] models:`, JSON.stringify(nativeConfig.models.map(m => ({
    path: m.path, inputShape: m.inputShape, sync: m.sync
  }))));
}
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: capture model resources in pipeline proxy and include in native config"
```

---

## Task 3: Create ModelRunner C++ class

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.h`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.mm`

This is the core native component. It manages an ONNX InferenceSession, a resize+normalize compute shader, and a background inference thread.

- [ ] **Step 1: Create ModelRunner.h**

```cpp
#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>

#include <webgpu/webgpu_cpp.h>

namespace dawn_pipeline {

struct ModelSpec {
  std::string path;
  std::vector<int64_t> inputShape;  // e.g. [1, 3, 518, 518]
  float normMean[3] = {0.485f, 0.456f, 0.406f};
  float normStd[3] = {0.229f, 0.224f, 0.225f};
  bool sync = false;
  int pipelineIndex = -1;  // position in pass chain
};

class ModelRunner {
public:
  ModelRunner(wgpu::Device device, int cameraWidth, int cameraHeight);
  ~ModelRunner();

  /** Load model, create session, compile resize shader, allocate buffers. */
  bool setup(const ModelSpec& spec);

  /** Submit a new frame for inference (called from processFrame thread).
   *  For async models, copies the texture and returns immediately.
   *  For sync models, blocks until inference completes. */
  void submitFrame(wgpu::Texture cameraTexture);

  /** Get the latest output texture view (may be null if no result yet). */
  wgpu::TextureView getOutputView() const;

  /** Get the output dimensions. */
  int outputWidth() const { return _outputW; }
  int outputHeight() const { return _outputH; }

  /** Check if a result is available. */
  bool hasResult() const { return _hasResult.load(); }

  void shutdown();

private:
  void inferenceLoop();
  void runInference();

  wgpu::Device _device;
  int _cameraW, _cameraH;
  int _modelW = 0, _modelH = 0;  // model input spatial dims
  int _outputW = 0, _outputH = 0;

  // Resize + normalize compute shader
  wgpu::ComputePipeline _resizePipeline;
  wgpu::Buffer _modelInputBuffer;  // NCHW float32 buffer for model input

  // ONNX Runtime session (opaque — ort headers only in .mm)
  void* _session = nullptr;       // Ort::Session*
  void* _ioBinding = nullptr;     // Ort::IoBinding*

  // Output
  wgpu::Texture _outputTexture;
  wgpu::TextureView _outputView;
  mutable std::mutex _outputMutex;

  // Async inference thread
  std::thread _inferenceThread;
  std::atomic<bool> _running{false};
  std::atomic<bool> _hasResult{false};
  std::atomic<bool> _hasNewFrame{false};

  // Frame handoff: processFrame writes, inference thread reads
  wgpu::Texture _pendingTexture;
  std::mutex _frameMutex;

  ModelSpec _spec;
};

}  // namespace dawn_pipeline
```

- [ ] **Step 2: Create ModelRunner.mm — setup method**

The setup method:
1. Parses the model path (from `resources/` or documents directory)
2. Gets Dawn device/instance pointers for ONNX WebGPU EP
3. Creates an `Ort::InferenceSession` with WebGPU EP using the shared Dawn device
4. Reads model input/output metadata (shape, type)
5. Compiles a resize+normalize compute shader (camera res → model input size)
6. Allocates the model input GPU buffer (NCHW float32)
7. Creates the output texture (model output dims, R16Float for depth, RGBA for others)
8. For async models, starts the background inference thread

Key code for session creation (uses same pattern as OrtTest.tsx but in C++):

```cpp
#include "ModelRunner.h"
#include "onnxruntime_cxx_api.h"
#include "rnskia/RNDawnContext.h"
#include <dawn/native/DawnNative.h>

namespace dawn_pipeline {

bool ModelRunner::setup(const ModelSpec& spec) {
  _spec = spec;

  // Get Dawn pointers for WebGPU EP
  auto& ctx = RNSkia::DawnContext::getInstance();
  auto device = ctx.getWGPUDevice();
  auto instance = ctx.getWGPUInstance();

  static const DawnProcTable* procs = nullptr;
  if (!procs) {
    static DawnProcTable p = dawn::native::GetProcs();
    procs = &p;
  }

  // Create ONNX Runtime environment (static — must outlive all sessions)
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelRunner");
  Ort::SessionOptions sessionOptions;

  // Configure WebGPU EP with shared Dawn device
  std::unordered_map<std::string, std::string> epOptions;
  epOptions["deviceId"] = "1";
  epOptions["webgpuDevice"] = std::to_string((uintptr_t)device.Get());
  epOptions["webgpuInstance"] = std::to_string((uintptr_t)instance.Get());
  epOptions["dawnProcTable"] = std::to_string((uintptr_t)procs);
  sessionOptions.AppendExecutionProvider("WebGPU", epOptions);

  _session = new Ort::Session(env, spec.path.c_str(), sessionOptions);

  // Read model input/output shapes
  auto* session = static_cast<Ort::Session*>(_session);
  auto inputInfo = session->GetInputTypeInfo(0);
  auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
  auto shape = tensorInfo.GetShape();

  if (!spec.inputShape.empty()) {
    _modelH = (int)spec.inputShape[2];
    _modelW = (int)spec.inputShape[3];
  } else if (shape.size() == 4) {
    _modelH = (int)shape[2];
    _modelW = (int)shape[3];
  }

  // Read output shape to determine output texture size
  auto outputInfo = session->GetOutputTypeInfo(0);
  auto outTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
  auto outShape = outTensorInfo.GetShape();
  // For depth models: [1, 1, H, W] or [1, H, W]
  // For classification: [1, 1000] — not a spatial output
  if (outShape.size() == 4) {
    _outputH = (int)outShape[2];
    _outputW = (int)outShape[3];
  } else if (outShape.size() == 3) {
    _outputH = (int)outShape[1];
    _outputW = (int)outShape[2];
  } else {
    // Non-spatial output (e.g. classification) — use 1x1 texture
    _outputW = 1;
    _outputH = (int)(outShape.back());
  }

  // Compile resize + normalize compute shader
  // ... (Step 3)

  // Allocate model input buffer
  int inputElements = 1 * 3 * _modelH * _modelW;
  wgpu::BufferDescriptor bufDesc{};
  bufDesc.size = inputElements * sizeof(float);
  bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  bufDesc.label = "ModelInputNCHW";
  _modelInputBuffer = _device.CreateBuffer(&bufDesc);

  // Create output texture
  // Phase 1: depth models only (single-channel R16Float).
  // Future: determine format from output shape (3-ch → RGBA16Float, etc.)
  wgpu::TextureDescriptor outTexDesc{};
  outTexDesc.size = {(uint32_t)_outputW, (uint32_t)_outputH, 1};
  outTexDesc.format = wgpu::TextureFormat::R16Float;
  outTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  outTexDesc.label = "ModelOutput";
  _outputTexture = _device.CreateTexture(&outTexDesc);
  _outputView = _outputTexture.CreateView();

  // Start background inference thread for async models
  if (!spec.sync) {
    _running = true;
    _inferenceThread = std::thread(&ModelRunner::inferenceLoop, this);
  }

  NSLog(@"[ModelRunner] Setup complete: model=%s, input=%dx%d, output=%dx%d, sync=%d",
        spec.path.c_str(), _modelW, _modelH, _outputW, _outputH, spec.sync);
  return true;
}

}  // namespace dawn_pipeline
```

Note: The Ort::Env should be a static/long-lived object — we'll make it a static local in setup().

- [ ] **Step 3: Add resize+normalize compute shader**

The shader reads from the camera's RGBA16Float texture and writes NCHW float32 data to a storage buffer, with bilinear interpolation and ImageNet normalization:

```cpp
static const char* kResizeNormalizeWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  modelW: u32,
  modelH: u32,
  meanR: f32, meanG: f32, meanB: f32,
  stdR: f32, stdG: f32, stdB: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.modelW || id.y >= params.modelH) { return; }

  let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(params.modelW), f32(params.modelH));
  let pixel = textureSampleLevel(inputTex, inputSampler, uv, 0.0);

  let r = (pixel.r - params.meanR) / params.stdR;
  let g = (pixel.g - params.meanG) / params.stdG;
  let b = (pixel.b - params.meanB) / params.stdB;

  let idx = id.y * params.modelW + id.x;
  let planeSize = params.modelW * params.modelH;
  output[idx] = r;
  output[planeSize + idx] = g;
  output[2u * planeSize + idx] = b;
}
)";
```

Compile this shader and create the resize pipeline + bind group in `setup()`.

Create a uniform buffer for the normalization params and upload once at setup:

```cpp
struct ResizeParams {
  uint32_t modelW, modelH;
  float meanR, meanG, meanB;
  float stdR, stdG, stdB;
};

ResizeParams params{};
params.modelW = _modelW;
params.modelH = _modelH;
params.meanR = spec.normMean[0];
params.meanG = spec.normMean[1];
params.meanB = spec.normMean[2];
params.stdR = spec.normStd[0];
params.stdG = spec.normStd[1];
params.stdB = spec.normStd[2];

wgpu::BufferDescriptor paramBufDesc{};
paramBufDesc.size = sizeof(ResizeParams);
paramBufDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
_paramBuffer = _device.CreateBuffer(&paramBufDesc);
_device.GetQueue().WriteBuffer(_paramBuffer, 0, &params, sizeof(params));
```

- [ ] **Step 4: Implement submitFrame and async inference loop**

`submitFrame`: copies the current pipeline texture reference for the inference thread.

**Texture lifetime**: `wgpu::Texture` is a reference-counted handle — assigning it to `_pendingTexture` increments the Dawn refcount, keeping the texture alive even if the pipeline ping-pongs to the other texture on the next frame. The inference thread's local copy also holds a ref. This is safe.

```cpp
void ModelRunner::submitFrame(wgpu::Texture cameraTexture) {
  if (_spec.sync) {
    // Sync mode: run resize shader, then inference, block until done
    runResizeShader(cameraTexture);
    runInference();
    return;
  }

  // Async: hand off texture to inference thread
  // wgpu::Texture is refcounted — inference thread holds its own ref
  {
    std::lock_guard<std::mutex> lock(_frameMutex);
    _pendingTexture = cameraTexture;
    _hasNewFrame = true;
  }
}
```

`inferenceLoop`: runs on background thread, processes latest frame:

```cpp
void ModelRunner::inferenceLoop() {
  while (_running) {
    wgpu::Texture frame;
    {
      std::lock_guard<std::mutex> lock(_frameMutex);
      if (!_hasNewFrame) {
        // No new frame — sleep briefly and retry
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      frame = _pendingTexture;
      _hasNewFrame = false;
    }

    runResizeShader(frame);
    runInference();
  }
}
```

`runInference`: runs the ONNX session with IO binding:

```cpp
void ModelRunner::runInference() {
  auto* session = static_cast<Ort::Session*>(_session);

  // IO binding: bind GPU buffer as input, pre-allocated buffer as output
  Ort::IoBinding binding(*session);
  Ort::MemoryInfo gpuMem("WebGPU_Buffer", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  int64_t inputShape[] = {1, 3, _modelH, _modelW};
  // ... bind input tensor from _modelInputBuffer ...
  // ... bind output ...
  // ... session->Run(Ort::RunOptions{nullptr}, binding) ...

  // Copy output buffer → output texture
  // ... update _outputView under _outputMutex ...

  _hasResult = true;
}
```

**Important**: The IO binding approach requires the ONNX Runtime WebGPU EP to expose the GPU buffer pointer. If IO binding is not available in the current ORT build, fall back to CPU tensors:
1. Map the model input buffer to CPU, fill from resize shader output
2. Run session.Run() with CPU tensors
3. Copy output to GPU texture

Document both paths in the code with a compile-time `#ifdef ORT_HAS_IO_BINDING` guard. Start with the CPU tensor fallback (simpler, proven working) and upgrade to IO binding when available.

- [ ] **Step 5: Implement shutdown and destructor**

```cpp
void ModelRunner::shutdown() {
  _running = false;
  if (_inferenceThread.joinable()) {
    _inferenceThread.join();
  }
  if (_session) {
    delete static_cast<Ort::Session*>(_session);
    _session = nullptr;
  }
}

ModelRunner::~ModelRunner() {
  shutdown();
}
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ModelRunner.mm
git commit -m "feat: add ModelRunner — ONNX inference with resize shader and async thread"
```

---

## Task 4: Integrate ModelRunner into DawnComputePipeline

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

This task wires ModelRunner into the existing pipeline so models are set up at init and fed frames during processFrame.

- [ ] **Step 1: Add ModelSpec and model state to DawnComputePipeline.h**

Add to the C interface:

```cpp
// In the C bridge section:
// ModelSpec is passed via the existing resources/passInputs mechanism

// In the DawnComputePipeline class:
struct ModelRunnerState {
  std::unique_ptr<ModelRunner> runner;
  int pipelineIndex = -1;  // which pass position this model occupies
};
```

Add to the `setup()` signature:

```cpp
bool setup(const std::vector<std::string>& wgslShaders,
           int width, int height,
           const std::vector<BufferSpec>& bufferSpecs,
           bool useCanvas, bool sync,
           const std::vector<ResourceSpec>& resources = {},
           const std::vector<PassInputSpec>& passInputs = {},
           const std::vector<int>& textureOutputPasses = {},
           bool appleLog = false,
           bool useDepth = false,
           bool lidarYUV = false,
           const std::vector<ModelSpec>& models = {});  // NEW
```

Add to the C interface:

```cpp
bool dawn_pipeline_setup_multipass(
  DawnComputePipelineRef ref,
  const char** shaders, int shaderCount,
  int width, int height,
  const int* bufferSpecs, int bufferCount,
  bool useCanvas, bool sync, bool appleLog, bool useDepth, bool lidarYUV,
  const void* resources, int resourceCount,
  const void* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount,
  const void* modelSpecs, int modelCount);  // NEW
```

- [ ] **Step 2: Add model state to Impl struct**

In `DawnComputePipeline.mm`, add to the `Impl` struct:

```cpp
// Models
std::vector<std::unique_ptr<ModelRunner>> models;
// Model output textures indexed by model index — used for input binding
```

- [ ] **Step 3: Initialize models in setup()**

After resource upload and before bind group caching, initialize models:

```cpp
// ── Setup model runners ──
for (const auto& modelSpec : models) {
  auto runner = std::make_unique<ModelRunner>(_impl->device, _width, _height);
  if (!runner->setup(modelSpec)) {
    NSLog(@"[DawnPipeline] FAILED to setup model: %s", modelSpec.path.c_str());
    cleanupLocked();
    return false;
  }
  _impl->models.push_back(std::move(runner));
}
```

- [ ] **Step 4: Feed frames to models in processFrame()**

After the compute pass loop completes (after all shader passes dispatch), submit the current pipeline texture to each model:

```cpp
// ── Submit frame to model runners ──
// Models read from the latest pipeline output (after all shader passes)
wgpu::Texture& currentOutput = finalIsA ? impl->texA : impl->texB;
for (auto& model : impl->models) {
  model->submitFrame(currentOutput);
}
```

For passes that consume model output, modify `appendCustomInputEntries()` to handle model output bindings. Add a new `InputBindingType::ModelOutput` or use a convention where `modelOutput >= 0` in the input binding.

In `appendCustomInputEntries`, add handling:

```cpp
// Model output texture
if (ib.modelOutput >= 0 && (size_t)ib.modelOutput < models.size()) {
  auto& model = models[ib.modelOutput];
  if (model->hasResult()) {
    entry.textureView = model->getOutputView();
    entries.push_back(entry);
  }
}
```

- [ ] **Step 5: Clean up models in cleanupLocked()**

```cpp
for (auto& model : _impl->models) {
  model->shutdown();
}
_impl->models.clear();
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: integrate ModelRunner into DawnComputePipeline setup and processFrame"
```

---

## Task 5: Wire model specs through the ObjC++ bridge and Swift module

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Add modelSpecs parameter to DawnPipelineBridge**

In `DawnPipelineBridge.h`, add `modelSpecs` parameter to `setupMultiPassWithShaders:`:

```objc
- (BOOL)setupMultiPassWithShaders:(NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                         appleLog:(BOOL)appleLog
                         useDepth:(BOOL)useDepth
                         lidarYUV:(BOOL)lidarYUV
                        resources:(NSArray<NSDictionary *> *)resources
                       passInputs:(NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(NSArray<NSNumber *> *)textureOutputPasses
                       modelSpecs:(NSArray<NSDictionary *> *)modelSpecs;  // NEW
```

- [ ] **Step 2: Convert model specs in DawnPipelineBridge.mm**

In the implementation, convert `NSArray<NSDictionary *>` to `std::vector<dawn_pipeline::ModelSpec>`:

```objc
// Convert model specs
std::vector<dawn_pipeline::ModelSpec> modelSpecsCpp;
for (NSDictionary *ms in modelSpecs) {
  dawn_pipeline::ModelSpec spec;

  NSString *path = ms[@"path"];
  if ([path hasPrefix:@"file://"]) {
    path = [path substringFromIndex:7];
  }
  spec.path = [path UTF8String];

  NSArray<NSNumber *> *shape = ms[@"inputShape"];
  if (shape) {
    for (NSNumber *n in shape) {
      spec.inputShape.push_back([n longLongValue]);
    }
  }

  NSArray<NSNumber *> *mean = ms[@"normMean"];
  if (mean && mean.count == 3) {
    spec.normMean[0] = [mean[0] floatValue];
    spec.normMean[1] = [mean[1] floatValue];
    spec.normMean[2] = [mean[2] floatValue];
  }

  NSArray<NSNumber *> *std = ms[@"normStd"];
  if (std && std.count == 3) {
    spec.normStd[0] = [std[0] floatValue];
    spec.normStd[1] = [std[1] floatValue];
    spec.normStd[2] = [std[2] floatValue];
  }

  spec.sync = [ms[@"sync"] boolValue];
  spec.pipelineIndex = [ms[@"pipelineIndex"] intValue];

  modelSpecsCpp.push_back(spec);
}
```

Pass to `dawn_pipeline_setup_multipass()`.

- [ ] **Step 3: Pass modelSpecs from Swift module**

In `WebGPUCameraModule.swift`, extract `models` from the pipeline config dictionary and pass to the bridge:

```swift
let modelSpecs = config["models"] as? [[String: Any]] ?? []
// ... pass to bridge.setupMultiPass(..., modelSpecs: modelSpecs)
```

- [ ] **Step 4: Build and verify compilation**

Run: `cd /Users/kim/dev/rn-webgpu-camera && eas build --platform ios --profile development --local 2>&1 | tail -50`

This verifies the whole native chain compiles. Expect compilation to succeed (runtime testing is Task 6).

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: wire model specs through ObjC++ bridge and Swift module"
```

---

## Task 6: Add model input binding type to pass input system

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` (InputBinding)
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` (appendCustomInputEntries)
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm` (parse model output bindings)
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` (emit model output bindings in passInputs)

This connects model outputs to shader inputs through the existing pass input binding system.

- [ ] **Step 1: Add modelOutput field to InputBinding**

In `DawnComputePipeline.h`:

```cpp
struct InputBinding {
  int bindingIndex = 0;
  InputBindingType type;
  int resourceHandle = -1;
  int sourcePass = -1;
  int sourceBuffer = -1;
  int modelOutput = -1;  // NEW: index into models array
};
```

- [ ] **Step 2: Handle model output in appendCustomInputEntries**

In `DawnComputePipeline.mm`, in the `appendCustomInputEntries` method, add handling for `modelOutput`:

```cpp
case InputBindingType::Texture2D:
  // ... existing resource handle and source pass cases ...
  // NEW: model output texture
  if (ib.modelOutput >= 0 && (size_t)ib.modelOutput < models.size()) {
    auto view = models[ib.modelOutput]->getOutputView();
    if (view) {
      entry.textureView = view;
      entries.push_back(entry);
    }
  }
  break;

case InputBindingType::Sampler:
  // ... existing cases ...
  // Model output sampler — use default sampler
  if (ib.modelOutput >= 0) {
    entry.sampler = defaultSampler;
    entries.push_back(entry);
  }
  break;
```

Mark passes with model output inputs as `hasDynamicInputs = true` (model output changes async).

- [ ] **Step 3: Parse modelOutput in bridge conversion**

In `DawnPipelineBridge.mm`, when converting passInputs bindings:

```objc
ib.modelOutput = b[@"modelOutput"] ? [b[@"modelOutput"] intValue] : -1;
```

- [ ] **Step 4: Emit modelOutput in JS passInputs**

In `useGPUFrameProcessor.ts` `buildNativeConfig()`, when processing pass inputs with model outputs:

```typescript
// In the passInputs building loop, when inp.modelOutput is set:
const binding: any = {
  index: inp.bindingIndex,
  type: inp.type,
};
if (inp.modelOutput !== undefined) {
  binding.modelOutput = inp.modelOutput;
} else if (inp.resourceHandle !== undefined) {
  binding.resourceHandle = inp.resourceHandle;
} else if (inp.sourcePass !== undefined) {
  binding.sourcePass = inp.sourcePass + passOffset;
  binding.sourceBuffer = inp.sourceBuffer;
}
```

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: connect model output textures to shader pass inputs"
```

---

## Task 7: End-to-end example — depth estimation with frame.runModel()

**Files:**
- Modify: `apps/example/src/app/index.tsx`

This task adds a working example that uses `frame.runModel()` with a depth estimation model.

- [ ] **Step 1: Add depth model pipeline example**

Add a new shader option or pipeline variant that uses `frame.runModel()`:

```typescript
import { GPUResource } from 'react-native-webgpu-camera';

// In the example app's pipeline config:
const DEPTH_MODEL_URL = 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx';

// Resource declaration
resources: {
  depth: GPUResource.model(DEPTH_MODEL_URL, {
    inputShape: [1, 3, 518, 518],
  }),
},
pipeline: (frame, { depth }) => {
  'worklet';
  const depthMap = frame.runModel(depth);
  if (depthMap) {
    frame.runShader(DEPTH_OVERLAY_WGSL, { inputs: { depth: depthMap } });
  }
},
```

The overlay shader blends the depth map with the camera feed (similar to the existing LiDAR depth shader but sampling from the model output texture):

```wgsl
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let camera = textureLoad(inputTex, vec2i(id.xy), 0);
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let d = textureSampleLevel(depthTex, depthSampler, uv, 0.0).r;

  // Blue → green → yellow colormap
  var color: vec3f;
  if (d < 0.5) {
    color = mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), d * 2.0);
  } else {
    color = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), (d - 0.5) * 2.0);
  }

  let blended = mix(camera.rgb, color, 0.4);
  textureStore(outputTex, vec2i(id.xy), vec4f(blended, 1.0));
}
```

- [ ] **Step 2: Handle model download**

The model needs to be downloaded before the pipeline is created. Add model download logic — either:
- In `GPUResource.model()` setup, check if URL needs download → use fetch + cache to documents dir
- Or have the native side handle download via `NSURLSession`

The simpler approach is JS-side download before pipeline creation, storing the local path:

```typescript
const [modelPath, setModelPath] = useState<string | null>(null);

useEffect(() => {
  (async () => {
    const localPath = `${Paths.document.uri}/depth-anything-v2-small.onnx`;
    const file = new File(localPath);
    if (!file.exists) {
      const resp = await fetch(DEPTH_MODEL_URL);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      file.write(bytes);
    }
    setModelPath(localPath);
  })();
}, []);

// Only create pipeline after model is downloaded
const resources = modelPath ? {
  depth: GPUResource.model(modelPath, { inputShape: [1, 3, 518, 518] }),
} : undefined;
```

- [ ] **Step 3: Test on device**

Build with: `eas build --platform ios --profile development --local`
Deploy to device and verify:
1. Camera feed displays normally
2. After model loads (~1-2s), depth overlay appears
3. Depth map updates at model FPS (lower than camera FPS — that's expected)
4. Camera feed stays smooth at 30/60fps while depth updates at 10-15fps

- [ ] **Step 4: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: add depth estimation example using frame.runModel()"
```

---

## Implementation Notes

### CPU Tensor Fallback (Start Here)

The IO binding path (`Ort::IoBinding` with `WebGPU_Buffer` memory) requires specific ORT WebGPU EP support that may not be fully available. **Start with the CPU tensor fallback**:

1. Resize shader writes NCHW data to a GPU buffer
2. Map buffer to CPU, create `Ort::Value` from the mapped data
3. Run `session.Run()` with CPU tensors
4. Copy float output to GPU texture via `WriteTexture()`

This is proven working (we ran MobileNetV2 this way). The per-frame overhead is ~1ms for a 518×518 model — acceptable since inference itself takes 10-30ms.

### IO Binding Upgrade Path

Once CPU fallback works end-to-end, upgrade to IO binding:
1. Use `Ort::MemoryInfo("WebGPU_Buffer", ...)` for input/output binding
2. Pass GPU buffer pointers directly
3. Zero CPU copies — the resize shader output buffer IS the model input

### Thread Safety

- `submitFrame()` is called from the camera pipeline thread (inside `processFrame`)
- `inferenceLoop()` runs on its own thread
- `getOutputView()` is called from `processFrame` when building bind groups
- All three are synchronized via `_frameMutex` (frame handoff) and `_outputMutex` (result)
- The pipeline mutex in `DawnComputePipeline` is NOT held during model operations

### Model Output as Dynamic Input

Model outputs change asynchronously — passes consuming model output must be marked `hasDynamicInputs = true` so their bind groups are rebuilt every frame (same pattern as LiDAR depth).
