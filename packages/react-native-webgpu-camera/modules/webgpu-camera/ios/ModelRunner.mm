#include "ModelRunner.h"
#include "onnxruntime_cxx_api.h"
#include "rnskia/RNDawnContext.h"
#include <dawn/native/DawnNative.h>

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>

#include <chrono>
#include <cstring>

namespace dawn_pipeline {

// ── Resize params — must be 16-byte aligned for WebGPU uniform buffer ──

struct ResizeParams {
  uint32_t modelW;
  uint32_t modelH;
  uint32_t inputW;   // raw IOSurface width (landscape)
  uint32_t inputH;   // raw IOSurface height (landscape)
  float meanR, meanG, meanB;
  float stdR, stdG, stdB;
  float padValue;    // letterbox pad color (0.447 = 114/255 for YOLO)
  // 12 x 4 bytes = 48 bytes — 16-byte aligned
};

// ── Resize + normalize compute shader ──
// Reads RGBA16Float camera texture, writes NCHW float32 to storage buffer.
// Bilinear interpolation via textureSampleLevel, configurable normalization.

static const char* kResizeNormalizeWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  modelW: u32,
  modelH: u32,
  inputW: u32,
  inputH: u32,
  meanR: f32, meanG: f32, meanB: f32,
  stdR: f32, stdG: f32, stdB: f32,
  padValue: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.modelW || id.y >= params.modelH) { return; }

  // Letterbox: fit portrait source into model input, preserving aspect ratio.
  // inputW/inputH are already portrait dims (set from _cameraW/_cameraH after rotation).
  // inputW = 1080 (portrait width), inputH = 1920 (portrait height).
  let srcW = f32(params.inputW);  // portrait width (1080)
  let srcH = f32(params.inputH);  // portrait height (1920)
  let dstW = f32(params.modelW);
  let dstH = f32(params.modelH);

  // Scale to fit, preserving aspect ratio
  let scale = min(dstW / srcW, dstH / srcH);
  let scaledW = srcW * scale;
  let scaledH = srcH * scale;
  let padX = (dstW - scaledW) * 0.5;
  let padY = (dstH - scaledH) * 0.5;

  let px = f32(id.x);
  let py = f32(id.y);

  var r: f32; var g: f32; var b: f32;

  if (px < padX || px >= padX + scaledW || py < padY || py >= padY + scaledH) {
    // Pad region
    r = (params.padValue - params.meanR) / params.stdR;
    g = (params.padValue - params.meanG) / params.stdG;
    b = (params.padValue - params.meanB) / params.stdB;
  } else {
    // Map to source UV with 90° CW rotation (same as pass 0)
    // pass 0: output(x,y) ← input(y, H-1-x)
    // In UV: uv = (srcY, 1 - srcX)
    let srcX = (px - padX) / scaledW;
    let srcY = (py - padY) / scaledH;
    let uv = vec2f(srcY, 1.0 - srcX);
    let pixel = textureSampleLevel(inputTex, inputSampler, uv, 0.0);
    r = (pixel.r - params.meanR) / params.stdR;
    g = (pixel.g - params.meanG) / params.stdG;
    b = (pixel.b - params.meanB) / params.stdB;
  }

  let idx = id.y * params.modelW + id.x;
  let planeSize = params.modelW * params.modelH;
  output[idx] = r;
  output[planeSize + idx] = g;
  output[2u * planeSize + idx] = b;
}
)";

// ── Constructor / Destructor ──

ModelRunner::ModelRunner(wgpu::Device device, int cameraWidth, int cameraHeight)
    : _device(device), _cameraW(cameraWidth), _cameraH(cameraHeight) {}

ModelRunner::~ModelRunner() {
  NSLog(@"[ModelRunner] ~ModelRunner() destructor called");
  shutdown();
}

// ── Setup ──

bool ModelRunner::setup(const ModelSpec& spec) {
  static int setupCount = 0;
  NSLog(@"[ModelRunner] setup() call #%d for model: %s", ++setupCount, spec.path.c_str());
  _spec = spec;

  // ── 1. Create secondary Dawn device for ORT (own queue + mutex) ──
  auto& ctx = RNSkia::DawnContext::getInstance();
  auto instance = ctx.getWGPUInstance();

  _ortDevice = ctx.createSecondaryDevice();
  if (!_ortDevice) {
    NSLog(@"[ModelRunner] FAILED to create secondary Dawn device");
    return false;
  }
  NSLog(@"[ModelRunner] Secondary Dawn device created (separate queue)");

  static const DawnProcTable* procs = nullptr;
  if (!procs) {
    static DawnProcTable p = dawn::native::GetProcs();
    procs = &p;
  }

  // ── 2. Create ONNX Runtime session with WebGPU EP on secondary device ──
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelRunner");

  try {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // ORT uses the secondary device — its own queue, won't block camera pipeline
    std::unordered_map<std::string, std::string> epOptions;
    epOptions["deviceId"] = "1";
    epOptions["webgpuDevice"] = std::to_string((uintptr_t)_ortDevice.Get());
    epOptions["webgpuInstance"] = std::to_string((uintptr_t)instance.Get());
    epOptions["dawnProcTable"] = std::to_string((uintptr_t)procs);
    epOptions["validationMode"] = "disabled";
    sessionOptions.AppendExecutionProvider("WebGPU", epOptions);

    _session = new Ort::Session(env, spec.path.c_str(), sessionOptions);
    NSLog(@"[ModelRunner] ONNX session created on secondary device: %s", spec.path.c_str());
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "[ModelRunner] FAILED to create ONNX session: %s\n", e.what());
    return false;
  }

  auto* session = static_cast<Ort::Session*>(_session);

  // ── 3. Read model input metadata ──
  Ort::AllocatorWithDefaultOptions allocator;

  size_t inputCount = session->GetInputCount();
  for (size_t i = 0; i < inputCount; i++) {
    auto name = session->GetInputNameAllocated(i, allocator);
    _inputNames.push_back(name.get());
  }

  auto inputInfo = session->GetInputTypeInfo(0);
  auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
  auto shape = tensorInfo.GetShape();

  if (!spec.inputShape.empty() && spec.inputShape.size() == 4) {
    _modelH = (int)spec.inputShape[2];
    _modelW = (int)spec.inputShape[3];
  } else if (shape.size() == 4) {
    // Dynamic dims may be -1; use absolute value or default
    _modelH = (shape[2] > 0) ? (int)shape[2] : 518;
    _modelW = (shape[3] > 0) ? (int)shape[3] : 518;
  } else {
    NSLog(@"[ModelRunner] Unexpected input shape rank: %zu", shape.size());
    return false;
  }

  NSLog(@"[ModelRunner] Model input: %dx%d (from %s)",
        _modelW, _modelH, spec.inputShape.empty() ? "model metadata" : "user spec");

  // ── 4. Read model output metadata ──
  size_t outputCount = session->GetOutputCount();
  for (size_t i = 0; i < outputCount; i++) {
    auto name = session->GetOutputNameAllocated(i, allocator);
    _outputNames.push_back(name.get());
  }

  auto outputInfo = session->GetOutputTypeInfo(0);
  auto outTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
  auto outShape = outTensorInfo.GetShape();

  // Determine output spatial dimensions from shape
  // Depth models: [1, 1, H, W] or [1, H, W]
  // Classification: [1, 1000] — not spatial
  if (outShape.size() == 4) {
    _outputH = (outShape[2] > 0) ? (int)outShape[2] : _modelH;
    _outputW = (outShape[3] > 0) ? (int)outShape[3] : _modelW;
  } else if (outShape.size() == 3) {
    _outputH = (outShape[1] > 0) ? (int)outShape[1] : _modelH;
    _outputW = (outShape[2] > 0) ? (int)outShape[2] : _modelW;
  } else {
    // Non-spatial output (e.g. classification) — 1xN texture
    _outputW = (outShape.back() > 0) ? (int)outShape.back() : 1;
    _outputH = 1;
  }

  NSLog(@"[ModelRunner] Model output: %dx%d (shape rank=%zu)", _outputW, _outputH, outShape.size());

  // ── 5. Compile resize + normalize compute shader ──
  {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = kResizeNormalizeWGSL;

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc;

    auto shaderModule = _ortDevice.CreateShaderModule(&smDesc);
    if (!shaderModule) {
      NSLog(@"[ModelRunner] FAILED to create resize shader module");
      return false;
    }

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";
    cpDesc.label = "ModelResizeNormalize";

    _resizePipeline = _ortDevice.CreateComputePipeline(&cpDesc);
    if (!_resizePipeline) {
      NSLog(@"[ModelRunner] FAILED to create resize compute pipeline");
      return false;
    }

    _resizeBindGroupLayout = _resizePipeline.GetBindGroupLayout(0);
    NSLog(@"[ModelRunner] Resize shader compiled OK");
  }

  // ── 6. Create linear sampler for bilinear resize (on ORT device) ──
  {
    wgpu::SamplerDescriptor sampDesc{};
    sampDesc.magFilter = wgpu::FilterMode::Linear;
    sampDesc.minFilter = wgpu::FilterMode::Linear;
    sampDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    sampDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    sampDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    sampDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
    _resizeSampler = _ortDevice.CreateSampler(&sampDesc);
  }

  // ── 7. Allocate model input buffer on ORT device ──
  {
    uint32_t inputElements = 1 * 3 * _modelH * _modelW;
    uint32_t inputBytes = inputElements * sizeof(float);

    // ORT device: resize shader writes here, ORT reads via IO binding
    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = inputBytes;
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufDesc.label = "ModelInputNCHW";
    _modelInputBuffer = _ortDevice.CreateBuffer(&bufDesc);

    NSLog(@"[ModelRunner] Input buffer: %u elements (%u bytes, on ORT device)", inputElements, inputBytes);
  }

  // ── 8. Upload resize params to uniform buffer (on ORT device) ──
  {
    ResizeParams params{};
    params.modelW = (uint32_t)_modelW;
    params.modelH = (uint32_t)_modelH;
    params.inputW = (uint32_t)_cameraW;  // landscape IOSurface width
    params.inputH = (uint32_t)_cameraH;  // landscape IOSurface height
    params.meanR = spec.normMean[0];
    params.meanG = spec.normMean[1];
    params.meanB = spec.normMean[2];
    params.stdR = spec.normStd[0];
    params.stdG = spec.normStd[1];
    params.stdB = spec.normStd[2];
    params.padValue = 114.0f / 255.0f;  // YOLO standard letterbox pad color

    wgpu::BufferDescriptor paramBufDesc{};
    paramBufDesc.size = sizeof(ResizeParams);
    paramBufDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramBufDesc.label = "ModelResizeParams";
    _paramBuffer = _ortDevice.CreateBuffer(&paramBufDesc);
    _ortDevice.GetQueue().WriteBuffer(_paramBuffer, 0, &params, sizeof(params));
  }

  // ── 9. Create output buffers ──
  {
    _outputElements = (size_t)_outputW * _outputH;
    uint32_t outputBytes = (uint32_t)(_outputElements * sizeof(float));

    // ORT output on secondary device — MapRead for CPU bridge to primary
    wgpu::BufferDescriptor outBufDesc{};
    outBufDesc.size = outputBytes;
    outBufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc
                     | wgpu::BufferUsage::CopyDst;
    outBufDesc.label = "ModelOutputBuffer";
    _ortBuffer = _ortDevice.CreateBuffer(&outBufDesc);

    // Staging buffer on ORT device — pre-allocated, reused every frame
    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = outputBytes;
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    stagingDesc.label = "OrtOutputStaging";
    _stagingBuffer = _ortDevice.CreateBuffer(&stagingDesc);

    // Read buffer on PRIMARY device — shader binds this
    wgpu::BufferDescriptor readDesc{};
    readDesc.size = outputBytes;
    readDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    readDesc.label = "ModelReadBuffer";
    _readBuffer = _device.CreateBuffer(&readDesc);

    NSLog(@"[ModelRunner] Output buffers: %u bytes (dual-device, pre-allocated staging)", outputBytes);

    NSLog(@"[ModelRunner] Output: %dx%d (%u elements, buffer=f32)",
          _outputW, _outputH, (uint32_t)_outputElements);
  }

  // ── 10. Set up IO binding — pre-allocated GPU buffers bound as concrete tensors ──
  // Both buffer pointers are stable for the lifetime of this ModelRunner,
  // so the binding never needs to change between frames.
  try {
    NSLog(@"[ModelRunner] Creating IO binding...");
    auto* gpuMem = new Ort::MemoryInfo("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    _gpuMemInfo = gpuMem;

    auto* binding = new Ort::IoBinding(*session);

    // Bind input: resize shader output buffer
    // Tensors must be heap-allocated — binding holds references, locals would dangle.
    size_t inputElements = 3 * _modelH * _modelW;
    int64_t inputShapeArr[] = {1, 3, (int64_t)_modelH, (int64_t)_modelW};
    auto* inputTensor = new Ort::Value(Ort::Value::CreateTensor(
      *gpuMem,
      (void*)_modelInputBuffer.Get(),
      inputElements * sizeof(float),
      inputShapeArr, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    binding->BindInput(_inputNames[0].c_str(), *inputTensor);
    _boundInputTensor = inputTensor;
    NSLog(@"[ModelRunner] Input bound: %zu elements", inputElements);

    // Bind output: pre-allocated output buffer
    auto outputInfo2 = session->GetOutputTypeInfo(0);
    auto outTensorInfo2 = outputInfo2.GetTensorTypeAndShapeInfo();
    auto outShape2 = outTensorInfo2.GetShape();
    std::vector<int64_t> outputShapeVec;
    for (size_t i = 0; i < outShape2.size(); i++) {
      int64_t d = outShape2[i];
      if (d > 0) {
        outputShapeVec.push_back(d);
      } else if (i == 0) {
        outputShapeVec.push_back(1);  // batch dim
      } else if (i == outShape2.size() - 1) {
        outputShapeVec.push_back(_outputW);
      } else {
        outputShapeVec.push_back(_outputH);
      }
    }
    // Log the resolved shape for debugging
    {
      std::string shapeStr;
      size_t totalElements = 1;
      for (size_t i = 0; i < outputShapeVec.size(); i++) {
        if (i > 0) shapeStr += "x";
        shapeStr += std::to_string(outputShapeVec[i]);
        totalElements *= outputShapeVec[i];
      }
      NSLog(@"[ModelRunner] Output shape resolved: [%s] = %zu elements (buffer has %zu)",
            shapeStr.c_str(), totalElements, _outputElements);
    }

    auto* outputTensor = new Ort::Value(Ort::Value::CreateTensor(
      *gpuMem,
      (void*)_ortBuffer.Get(),
      _outputElements * sizeof(float),
      outputShapeVec.data(), outputShapeVec.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    binding->BindOutput(_outputNames[0].c_str(), *outputTensor);
    _boundOutputTensor = outputTensor;
    NSLog(@"[ModelRunner] Output bound: %zu elements", _outputElements);

    _ioBinding = binding;
    NSLog(@"[ModelRunner] IO binding created — fully pre-allocated, zero-copy GPU path");
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "[ModelRunner] IO binding setup FAILED: %s\n", e.what());
    return false;
  }

  // ── 10. Start background inference thread for async models ──
  if (!spec.sync) {
    _running = true;
    _inferenceThread = std::thread(&ModelRunner::inferenceLoop, this);
    NSLog(@"[ModelRunner] Async inference thread started");
  }

  NSLog(@"[ModelRunner] Setup complete: model=%s, input=%dx%d, output=%dx%d, sync=%d",
        spec.path.c_str(), _modelW, _modelH, _outputW, _outputH, spec.sync);
  return true;
}

// ── Resize shader dispatch ──

void ModelRunner::runResizeShader(IOSurfaceRef ioSurface) {
  if (!_running && !_spec.sync) return;
  if (!ioSurface) return;

  // Import the camera IOSurface on the ORT device — zero-copy shared memory
  wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
  ioDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor sharedDesc{};
  sharedDesc.nextInChain = &ioDesc;

  auto sharedMemory = _ortDevice.ImportSharedTextureMemory(&sharedDesc);
  if (!sharedMemory) {
    static int failCount = 0;
    if (failCount++ < 3) NSLog(@"[ModelRunner] FAILED to import IOSurface on ORT device");
    return;
  }

  // Create texture from shared memory — use raw IOSurface dimensions (unrotated)
  uint32_t ioW = (uint32_t)IOSurfaceGetWidth(ioSurface);
  uint32_t ioH = (uint32_t)IOSurfaceGetHeight(ioSurface);

  wgpu::TextureDescriptor texDesc{};
  texDesc.format = wgpu::TextureFormat::BGRA8Unorm;
  texDesc.dimension = wgpu::TextureDimension::e2D;
  texDesc.usage = wgpu::TextureUsage::TextureBinding;
  texDesc.size = {ioW, ioH, 1};

  auto inputTexture = sharedMemory.CreateTexture(&texDesc);

  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
  beginDesc.initialized = true;
  beginDesc.fenceCount = 0;
  if (!sharedMemory.BeginAccess(inputTexture, &beginDesc)) {
    return;
  }

  // Create bind group and dispatch resize shader
  wgpu::TextureView inputView = inputTexture.CreateView();

  std::vector<wgpu::BindGroupEntry> entries(4);
  entries[0].binding = 0;
  entries[0].textureView = inputView;
  entries[1].binding = 1;
  entries[1].sampler = _resizeSampler;
  entries[2].binding = 2;
  entries[2].buffer = _modelInputBuffer;
  entries[2].size = _modelInputBuffer.GetSize();
  entries[3].binding = 3;
  entries[3].buffer = _paramBuffer;
  entries[3].size = _paramBuffer.GetSize();

  wgpu::BindGroupDescriptor bgDesc{};
  bgDesc.layout = _resizeBindGroupLayout;
  bgDesc.entryCount = entries.size();
  bgDesc.entries = entries.data();

  auto bindGroup = _ortDevice.CreateBindGroup(&bgDesc);
  if (!bindGroup) {
    wgpu::SharedTextureMemoryEndAccessState endState{};
    sharedMemory.EndAccess(inputTexture, &endState);
    return;
  }

  auto encoder = _ortDevice.CreateCommandEncoder();
  auto pass = encoder.BeginComputePass();
  pass.SetPipeline(_resizePipeline);
  pass.SetBindGroup(0, bindGroup);

  uint32_t groupsX = (_modelW + 15) / 16;
  uint32_t groupsY = (_modelH + 15) / 16;
  pass.DispatchWorkgroups(groupsX, groupsY, 1);
  pass.End();

  auto commands = encoder.Finish();
  _ortDevice.GetQueue().Submit(1, &commands);

  // End shared access
  wgpu::SharedTextureMemoryEndAccessState endState{};
  sharedMemory.EndAccess(inputTexture, &endState);

}

// ── GPU-native inference via IO binding ──

void ModelRunner::runInference() {
  if (!_running && !_spec.sync) return;  // shutting down

  auto* session = static_cast<Ort::Session*>(_session);
  auto* binding = static_cast<Ort::IoBinding*>(_ioBinding);
  if (!session || !binding) return;

  try {
    // ── Run inference — all GPU, no CPU data movement ──
    // Input and output are pre-bound at setup — just Run().
    auto startTime = std::chrono::high_resolution_clock::now();

    auto tRun0 = std::chrono::high_resolution_clock::now();
    session->Run(Ort::RunOptions{nullptr}, *binding);
    auto tRun1 = std::chrono::high_resolution_clock::now();

    // ── Bridge ORT output from device B → device A via CPU ──
    // Map _ortBuffer on ORT device (no contention with camera pipeline).
    // Then WriteBuffer to _readBuffer on primary device (tiny ~1MB transfer).
    {
      uint32_t bytes = (uint32_t)(_outputElements * sizeof(float));

      // Create fresh staging buffer per frame (avoids map/unmap lifecycle issues)
      wgpu::BufferDescriptor stagingDesc{};
      stagingDesc.size = bytes;
      stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
      stagingDesc.label = "OrtOutputStaging";
      auto stagingBuf = _ortDevice.CreateBuffer(&stagingDesc);

      auto encoder = _ortDevice.CreateCommandEncoder();
      encoder.CopyBufferToBuffer(_ortBuffer, 0, stagingBuf, 0, bytes);
      auto commands = encoder.Finish();
      _ortDevice.GetQueue().Submit(1, &commands);

      auto instance = RNSkia::DawnContext::getInstance().getWGPUInstance();
      bool mapOk = false;
      auto future = stagingBuf.MapAsync(
        wgpu::MapMode::Read, 0, bytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [&mapOk](wgpu::MapAsyncStatus status, wgpu::StringView) {
          mapOk = (status == wgpu::MapAsyncStatus::Success);
        }
      );
      instance.WaitAny(future, UINT64_MAX);
      if (mapOk) {
        const void* mapped = stagingBuf.GetConstMappedRange(0, bytes);
        _device.GetQueue().WriteBuffer(_readBuffer, 0, mapped, bytes);
        // Keep a CPU copy for JS readback (onFrame)
        {
          std::lock_guard<std::mutex> lock(_outputMutex);
          _cpuOutputData.resize(bytes);
          memcpy(_cpuOutputData.data(), mapped, bytes);
        }
        stagingBuf.Unmap();
      }
    }
    auto tBridge1 = std::chrono::high_resolution_clock::now();
    _hasResult = true;

    double runMs = std::chrono::duration<double, std::milli>(tRun1 - tRun0).count();
    double bridgeMs = std::chrono::duration<double, std::milli>(tBridge1 - tRun1).count();
    double totalMs = runMs + bridgeMs;

    // Model FPS tracking — count completed inferences per second
    _inferenceCount++;
    double now = CACurrentMediaTime();
    if (_lastFpsTime == 0) {
      _lastFpsTime = now;
    } else if (now - _lastFpsTime >= 1.0) {
      double elapsed = now - _lastFpsTime;
      double fps = _inferenceCount / elapsed;
      NSLog(@"[ModelRunner] %.1f fps | run=%.1fms bridge=%.1fms total=%.1fms",
            fps, runMs, bridgeMs, totalMs);
      _inferenceCount = 0;
      _lastFpsTime = now;
    }

  } catch (const Ort::Exception& e) {
    // Build error string with known-safe characters to bypass iOS log redaction
    std::string msg = e.what();
    // Replace any characters that might trigger redaction
    NSMutableString *safe = [NSMutableString stringWithCapacity:msg.size()];
    for (char c : msg) {
      [safe appendFormat:@"%c", c];
    }
    NSLog(@"[ModelRunner] Inference FAILED reason=[%@]", safe);
    // Also write to Documents for retrieval
    static bool wroteError = false;
    if (!wroteError) {
      NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
      NSString *docDir = [paths firstObject];
      NSString *errFile = [docDir stringByAppendingPathComponent:@"model_error.txt"];
      NSString *errStr = [NSString stringWithUTF8String:e.what()];
      [errStr writeToFile:errFile atomically:YES encoding:NSUTF8StringEncoding error:nil];
      NSLog(@"[ModelRunner] Error written to %@", errFile);
      wroteError = true;
    }
  }
}

// ── Frame submission ──

void ModelRunner::submitFrame(wgpu::Texture cameraTexture, IOSurfaceRef ioSurface) {
  if (_spec.sync) {
    runResizeShader(ioSurface);
    runInference();
    return;
  }

  // Async: hand off IOSurface to inference thread.
  // IOSurface is refcounted by the OS — safe to pass across threads.
  {
    std::lock_guard<std::mutex> lock(_frameMutex);
    if (_pendingIOSurface) CFRelease(_pendingIOSurface);
    _pendingIOSurface = (IOSurfaceRef)CFRetain(ioSurface);
    _hasNewFrame = true;
  }
  static int submitLog = 0;
  if (submitLog++ < 5) {
    NSLog(@"[ModelRunner] submitFrame: queued frame %d (ioSurface=%p)", submitLog, ioSurface);
  }
}

// ── Async inference loop (background thread) ──

void ModelRunner::inferenceLoop() {
  NSLog(@"[ModelRunner] Inference thread started");

  while (_running) {
    IOSurfaceRef ioSurface = nullptr;
    {
      std::lock_guard<std::mutex> lock(_frameMutex);
      if (!_hasNewFrame) {
        // No new frame — unlock, sleep briefly, and retry.
      }
    }

    if (!_hasNewFrame.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(_frameMutex);
      if (!_hasNewFrame) continue;
      ioSurface = _pendingIOSurface;
      _pendingIOSurface = nullptr;  // take ownership
      _hasNewFrame = false;
    }

    static int loopLog = 0;
    if (loopLog++ < 5) {
      NSLog(@"[ModelRunner] inferenceLoop: got frame %d (ioSurface=%p)", loopLog, ioSurface);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    // Resize shader imports IOSurface on ORT device — no primary device involvement
    runResizeShader(ioSurface);
    if (ioSurface) CFRelease(ioSurface);
    auto t1 = std::chrono::high_resolution_clock::now();

    runInference();
    auto t2 = std::chrono::high_resolution_clock::now();

    double resizeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double inferMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double totalMs = std::chrono::duration<double, std::milli>(t2 - t0).count();

    if (loopLog <= 5) {
      NSLog(@"[ModelRunner] frame %d: resize=%.1fms run=%.1fms total=%.1fms hasResult=%d",
            loopLog, resizeMs, inferMs, totalMs, _hasResult.load());
    }
  }

  NSLog(@"[ModelRunner] Inference thread stopped");
}

// ── Output access ──

wgpu::TextureView ModelRunner::getOutputView() const {
  std::lock_guard<std::mutex> lock(_outputMutex);
  return _outputView;
}

// ── Shutdown ──

void ModelRunner::shutdown() {
  _running = false;
  if (_inferenceThread.joinable()) {
    _inferenceThread.join();
  }
  if (_ioBinding) {
    delete static_cast<Ort::IoBinding*>(_ioBinding);
    _ioBinding = nullptr;
  }
  if (_boundInputTensor) {
    delete static_cast<Ort::Value*>(_boundInputTensor);
    _boundInputTensor = nullptr;
  }
  if (_boundOutputTensor) {
    delete static_cast<Ort::Value*>(_boundOutputTensor);
    _boundOutputTensor = nullptr;
  }
  if (_gpuMemInfo) {
    delete static_cast<Ort::MemoryInfo*>(_gpuMemInfo);
    _gpuMemInfo = nullptr;
  }
  if (_session) {
    delete static_cast<Ort::Session*>(_session);
    _session = nullptr;
  }

  // Release GPU resources (primary device)
  _readBuffer = nullptr;

  // Release GPU resources (ORT device)
  _resizePipeline = nullptr;
  _resizeBindGroupLayout = nullptr;
  _modelInputBuffer = nullptr;
  _paramBuffer = nullptr;
  _resizeSampler = nullptr;
  _ortBuffer = nullptr;
  _stagingBuffer = nullptr;
  _ortDevice = nullptr;

  // Release IOSurface ref
  if (_pendingIOSurface) {
    CFRelease(_pendingIOSurface);
    _pendingIOSurface = nullptr;
  }

  NSLog(@"[ModelRunner] Shutdown complete");
}

}  // namespace dawn_pipeline
