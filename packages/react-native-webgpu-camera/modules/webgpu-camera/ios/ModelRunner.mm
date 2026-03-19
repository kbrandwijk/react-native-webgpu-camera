#include "ModelRunner.h"
#include "onnxruntime_cxx_api.h"
#include "rnskia/RNDawnContext.h"
#include <dawn/native/DawnNative.h>

#import <Foundation/Foundation.h>

#include <chrono>
#include <cstring>

namespace dawn_pipeline {

// ── Resize params — must be 16-byte aligned for WebGPU uniform buffer ──

struct ResizeParams {
  uint32_t modelW;
  uint32_t modelH;
  float meanR, meanG, meanB;
  float stdR, stdG, stdB;
  // 8 x 4 bytes = 32 bytes — already 16-byte aligned
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

// ── Constructor / Destructor ──

ModelRunner::ModelRunner(wgpu::Device device, int cameraWidth, int cameraHeight)
    : _device(device), _cameraW(cameraWidth), _cameraH(cameraHeight) {}

ModelRunner::~ModelRunner() {
  shutdown();
}

// ── Setup ──

bool ModelRunner::setup(const ModelSpec& spec) {
  _spec = spec;

  // ── 1. Get Dawn pointers for ONNX Runtime WebGPU EP ──
  auto& ctx = RNSkia::DawnContext::getInstance();
  auto device = ctx.getWGPUDevice();
  auto instance = ctx.getWGPUInstance();

  static const DawnProcTable* procs = nullptr;
  if (!procs) {
    static DawnProcTable p = dawn::native::GetProcs();
    procs = &p;
  }

  // ── 2. Create ONNX Runtime session with WebGPU EP ──
  // Ort::Env must be static — it must outlive all sessions
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelRunner");

  try {
    Ort::SessionOptions sessionOptions;

    // Configure WebGPU EP with shared Dawn device
    std::unordered_map<std::string, std::string> epOptions;
    epOptions["deviceId"] = "1";  // non-zero signals external Dawn device
    epOptions["webgpuDevice"] = std::to_string((uintptr_t)device.Get());
    epOptions["webgpuInstance"] = std::to_string((uintptr_t)instance.Get());
    epOptions["dawnProcTable"] = std::to_string((uintptr_t)procs);
    sessionOptions.AppendExecutionProvider("WebGPU", epOptions);

    _session = new Ort::Session(env, spec.path.c_str(), sessionOptions);
    NSLog(@"[ModelRunner] ONNX session created: %s", spec.path.c_str());
  } catch (const Ort::Exception& e) {
    NSLog(@"[ModelRunner] FAILED to create ONNX session: %{public}s", e.what());
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

    auto shaderModule = _device.CreateShaderModule(&smDesc);
    if (!shaderModule) {
      NSLog(@"[ModelRunner] FAILED to create resize shader module");
      return false;
    }

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";
    cpDesc.label = "ModelResizeNormalize";

    _resizePipeline = _device.CreateComputePipeline(&cpDesc);
    if (!_resizePipeline) {
      NSLog(@"[ModelRunner] FAILED to create resize compute pipeline");
      return false;
    }

    _resizeBindGroupLayout = _resizePipeline.GetBindGroupLayout(0);
    NSLog(@"[ModelRunner] Resize shader compiled OK");
  }

  // ── 6. Create linear sampler for bilinear resize ──
  {
    wgpu::SamplerDescriptor sampDesc{};
    sampDesc.magFilter = wgpu::FilterMode::Linear;
    sampDesc.minFilter = wgpu::FilterMode::Linear;
    sampDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    sampDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    sampDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    sampDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
    _resizeSampler = _device.CreateSampler(&sampDesc);
  }

  // ── 7. Allocate model input buffer (NCHW float32) ──
  // This buffer is shared: resize shader writes to it, ORT reads from it via IO binding.
  {
    uint32_t inputElements = 1 * 3 * _modelH * _modelW;
    uint32_t inputBytes = inputElements * sizeof(float);

    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = inputBytes;
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufDesc.label = "ModelInputNCHW";
    _modelInputBuffer = _device.CreateBuffer(&bufDesc);

    NSLog(@"[ModelRunner] Input buffer: %u elements (%u bytes)", inputElements, inputBytes);
  }

  // ── 8. Upload resize params to uniform buffer ──
  {
    ResizeParams params{};
    params.modelW = (uint32_t)_modelW;
    params.modelH = (uint32_t)_modelH;
    params.meanR = spec.normMean[0];
    params.meanG = spec.normMean[1];
    params.meanB = spec.normMean[2];
    params.stdR = spec.normStd[0];
    params.stdG = spec.normStd[1];
    params.stdB = spec.normStd[2];

    wgpu::BufferDescriptor paramBufDesc{};
    paramBufDesc.size = sizeof(ResizeParams);
    paramBufDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramBufDesc.label = "ModelResizeParams";
    _paramBuffer = _device.CreateBuffer(&paramBufDesc);
    _device.GetQueue().WriteBuffer(_paramBuffer, 0, &params, sizeof(params));
  }

  // ── 9. Create output texture ──
  // ORT allocates its own output GPU buffer via IO binding.
  // We copy from that buffer to this texture after inference.
  {
    _outputElements = (size_t)_outputW * _outputH;

    wgpu::TextureDescriptor outTexDesc{};
    outTexDesc.size = {(uint32_t)_outputW, (uint32_t)_outputH, 1};
    outTexDesc.format = wgpu::TextureFormat::R16Float;
    outTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    outTexDesc.dimension = wgpu::TextureDimension::e2D;
    outTexDesc.mipLevelCount = 1;
    outTexDesc.sampleCount = 1;
    outTexDesc.label = "ModelOutput";
    _outputTexture = _device.CreateTexture(&outTexDesc);
    _outputView = _outputTexture.CreateView();

    NSLog(@"[ModelRunner] Output: %dx%d (%u elements, texture=R16Float)",
          _outputW, _outputH, (uint32_t)_outputElements);
  }

  // ── 10. Set up IO binding — GPU buffers bound directly as input/output ──
  // Note: IO binding with pre-allocated GPU tensors requires careful lifetime
  // management. For now, bind output by name only (let ORT allocate) and
  // use the simpler RunWithBinding API. We'll read output from the binding.
  try {
    NSLog(@"[ModelRunner] Creating IO binding...");
    auto* binding = new Ort::IoBinding(*session);

    // Bind output by name + memory info — ORT allocates the output GPU buffer
    auto* gpuMem = new Ort::MemoryInfo("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    _gpuMemInfo = gpuMem;
    binding->BindOutput(_outputNames[0].c_str(), *gpuMem);
    NSLog(@"[ModelRunner] Output bound to WebGPU_Buf by name");

    _ioBinding = binding;
    NSLog(@"[ModelRunner] IO binding created — zero-copy GPU path");
  } catch (const Ort::Exception& e) {
    NSLog(@"[ModelRunner] IO binding setup FAILED: %{public}s", e.what());
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

void ModelRunner::runResizeShader(wgpu::Texture inputTexture) {
  if (!_running && !_spec.sync) return;  // shutting down

  // Create bind group for this frame's input texture
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

  auto bindGroup = _device.CreateBindGroup(&bgDesc);
  if (!bindGroup) {
    NSLog(@"[ModelRunner] FAILED to create resize bind group");
    return;
  }

  // Dispatch compute shader
  auto encoder = _device.CreateCommandEncoder();
  auto pass = encoder.BeginComputePass();
  pass.SetPipeline(_resizePipeline);
  pass.SetBindGroup(0, bindGroup);

  uint32_t groupsX = (_modelW + 15) / 16;
  uint32_t groupsY = (_modelH + 15) / 16;
  pass.DispatchWorkgroups(groupsX, groupsY, 1);
  pass.End();

  auto commands = encoder.Finish();
  _device.GetQueue().Submit(1, &commands);
}

// ── GPU-native inference via IO binding ──

void ModelRunner::runInference() {
  if (!_running && !_spec.sync) return;  // shutting down

  auto* session = static_cast<Ort::Session*>(_session);
  auto* binding = static_cast<Ort::IoBinding*>(_ioBinding);
  if (!session || !binding) return;

  try {
    auto* gpuMem = static_cast<Ort::MemoryInfo*>(_gpuMemInfo);

    // ── Bind input: GPU buffer from resize shader ──
    size_t inputElements = 3 * _modelH * _modelW;
    int64_t inputShapeArr[] = {1, 3, (int64_t)_modelH, (int64_t)_modelW};

    Ort::Value inputTensor = Ort::Value::CreateTensor(
      *gpuMem,
      (void*)_modelInputBuffer.Get(),
      inputElements * sizeof(float),
      inputShapeArr, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    binding->BindInput(_inputNames[0].c_str(), inputTensor);

    // Re-bind output each frame — GetOutputValues() consumes the previous binding
    binding->BindOutput(_outputNames[0].c_str(), *gpuMem);

    // ── Run inference — all GPU, no CPU data movement ──
    auto startTime = std::chrono::high_resolution_clock::now();

    session->Run(Ort::RunOptions{nullptr}, *binding);

    auto endTime = std::chrono::high_resolution_clock::now();
    double inferMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    static std::atomic<int> logCounter{0};
    if (logCounter.fetch_add(1) % 60 == 0) {
      NSLog(@"[ModelRunner] Inference: %.1fms (GPU IO binding)", inferMs);
    }

    // ── Get output from binding and copy to texture ──
    // ORT allocated the output GPU buffer; get it from the binding results.
    auto outputValues = binding->GetOutputValues();
    if (outputValues.empty() || !outputValues[0].IsTensor()) {
      NSLog(@"[ModelRunner] No valid output tensor from binding");
      return;
    }

    // The output tensor data pointer IS the WGPUBuffer handle
    const float* outputBufRaw = outputValues[0].GetTensorData<float>();
    WGPUBuffer outputWgpuBuf = reinterpret_cast<WGPUBuffer>(const_cast<float*>(outputBufRaw));

    auto outInfo = outputValues[0].GetTensorTypeAndShapeInfo();
    size_t outputElements = outInfo.GetElementCount();

    // Copy ORT's output GPU buffer → staging → CPU for f32→f16 conversion
    // TODO: replace with a f32→f16 compute shader for fully zero-copy
    wgpu::Buffer ortOutputBuf = wgpu::Buffer::Acquire(outputWgpuBuf);

    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = outputElements * sizeof(float);
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    stagingDesc.label = "ModelOutputStaging";
    auto stagingBuf = _device.CreateBuffer(&stagingDesc);

    auto encoder = _device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ortOutputBuf, 0, stagingBuf, 0, outputElements * sizeof(float));
    auto commands = encoder.Finish();
    _device.GetQueue().Submit(1, &commands);

    // Sync map on inference thread
    auto& ctx = RNSkia::DawnContext::getInstance();
    bool mapSuccess = false;
    auto future = stagingBuf.MapAsync(
      wgpu::MapMode::Read, 0, outputElements * sizeof(float),
      wgpu::CallbackMode::WaitAnyOnly,
      [&mapSuccess](wgpu::MapAsyncStatus status, wgpu::StringView) {
        mapSuccess = (status == wgpu::MapAsyncStatus::Success);
      }
    );
    ctx.getWGPUInstance().WaitAny(future, UINT64_MAX);

    if (!mapSuccess) {
      NSLog(@"[ModelRunner] FAILED to map output staging buffer");
      return;
    }

    const float* outputData = static_cast<const float*>(
      stagingBuf.GetConstMappedRange(0, outputElements * sizeof(float)));

    // Convert float32 → float16 for R16Float texture
    std::vector<uint16_t> f16Data(outputElements);
    for (size_t i = 0; i < outputElements; i++) {
      float val = outputData[i];
      uint32_t f32;
      std::memcpy(&f32, &val, 4);
      uint32_t sign = (f32 >> 16) & 0x8000;
      int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
      uint32_t mantissa = (f32 >> 13) & 0x3FF;
      if (exponent <= 0) {
        f16Data[i] = (uint16_t)sign;
      } else if (exponent >= 31) {
        f16Data[i] = (uint16_t)(sign | 0x7C00);
      } else {
        f16Data[i] = (uint16_t)(sign | (exponent << 10) | mantissa);
      }
    }
    stagingBuf.Unmap();

    // Upload to output texture
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = _outputTexture;
    dst.mipLevel = 0;
    dst.origin = {0, 0, 0};
    dst.aspect = wgpu::TextureAspect::All;

    wgpu::TexelCopyBufferLayout layout{};
    layout.offset = 0;
    layout.bytesPerRow = (uint32_t)_outputW * sizeof(uint16_t);
    layout.rowsPerImage = (uint32_t)_outputH;

    wgpu::Extent3D extent = {(uint32_t)_outputW, (uint32_t)_outputH, 1};
    _device.GetQueue().WriteTexture(
      &dst, f16Data.data(), f16Data.size() * sizeof(uint16_t), &layout, &extent);

    // Update output view
    {
      std::lock_guard<std::mutex> lock(_outputMutex);
      _outputView = _outputTexture.CreateView();
    }
    _hasResult = true;

    // Model FPS tracking
    _inferenceCount++;
    double now = CACurrentMediaTime();
    if (_lastFpsTime == 0) {
      _lastFpsTime = now;
    } else if (now - _lastFpsTime >= 1.0) {
      int fps = (int)(_inferenceCount / (now - _lastFpsTime));
      NSLog(@"[ModelRunner] %d fps (%.1fms avg)", fps, 1000.0 / std::max(fps, 1));
      _inferenceCount = 0;
      _lastFpsTime = now;
    }

  } catch (const Ort::Exception& e) {
    NSLog(@"[ModelRunner] Inference FAILED: %{public}s", e.what());
  }
}

// ── Frame submission ──

void ModelRunner::submitFrame(wgpu::Texture cameraTexture) {
  if (_spec.sync) {
    // Sync mode: run resize + inference, block until done
    runResizeShader(cameraTexture);
    runInference();
    return;
  }

  // Async: hand off texture to inference thread.
  // wgpu::Texture is refcounted — the inference thread's copy keeps the
  // texture alive even if the pipeline ping-pongs to the other buffer.
  {
    std::lock_guard<std::mutex> lock(_frameMutex);
    _pendingTexture = cameraTexture;
    _hasNewFrame = true;
  }
}

// ── Async inference loop (background thread) ──

void ModelRunner::inferenceLoop() {
  NSLog(@"[ModelRunner] Inference thread started");

  while (_running) {
    wgpu::Texture frame;
    {
      std::lock_guard<std::mutex> lock(_frameMutex);
      if (!_hasNewFrame) {
        // No new frame — unlock, sleep briefly, and retry.
        // 1ms sleep keeps CPU usage minimal while allowing ~30fps inference.
      }
    }

    if (!_hasNewFrame.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(_frameMutex);
      if (!_hasNewFrame) continue;
      frame = _pendingTexture;
      _hasNewFrame = false;
    }

    runResizeShader(frame);
    runInference();
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
  if (_gpuMemInfo) {
    delete static_cast<Ort::MemoryInfo*>(_gpuMemInfo);
    _gpuMemInfo = nullptr;
  }
  if (_session) {
    delete static_cast<Ort::Session*>(_session);
    _session = nullptr;
  }

  // Release GPU resources
  _resizePipeline = nullptr;
  _resizeBindGroupLayout = nullptr;
  _modelInputBuffer = nullptr;
  _paramBuffer = nullptr;
  _resizeSampler = nullptr;
  _outputTexture = nullptr;
  _outputView = nullptr;
  _pendingTexture = nullptr;

  NSLog(@"[ModelRunner] Shutdown complete");
}

}  // namespace dawn_pipeline
