#pragma once

#ifdef __cplusplus

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#include <webgpu/webgpu_cpp.h>
#include <IOSurface/IOSurfaceRef.h>

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

  /** Submit a new camera frame for inference.
   *  Takes the IOSurface so the ORT device can import it directly.
   *  For async models, stores the ref and returns immediately. */
  void submitFrame(wgpu::Texture cameraTexture, IOSurfaceRef ioSurface);

  /** Get the latest output texture view (may be null if no result yet). */
  wgpu::TextureView getOutputView() const;

  /** Get the output buffer for shader binding. Shader reads latest ORT output directly. */
  wgpu::Buffer getOutputBuffer() const { return _readBuffer; }

  /** Get the output dimensions. */
  int outputWidth() const { return _outputW; }
  int outputHeight() const { return _outputH; }

  /** Check if a result is available. */
  bool hasResult() const { return _hasResult.load(); }

  void shutdown();

private:
  void inferenceLoop();
  void runResizeShader(IOSurfaceRef ioSurface);
  void runInference();

  wgpu::Device _device;       // primary device (camera pipeline) — for _readBuffer
  wgpu::Device _ortDevice;    // secondary device (ORT) — own queue + mutex
  int _cameraW, _cameraH;
  int _modelW = 0, _modelH = 0;  // model input spatial dims
  int _outputW = 0, _outputH = 0;

  // Resize + normalize compute shader (runs on _device / primary)
  wgpu::ComputePipeline _resizePipeline;
  wgpu::BindGroupLayout _resizeBindGroupLayout;
  wgpu::Buffer _modelInputBuffer;  // on _device — resize shader output, mapped for CPU read
  wgpu::Buffer _paramBuffer;       // uniform buffer for resize params
  wgpu::Sampler _resizeSampler;    // linear sampler for bilinear resize

  // ORT input buffer on _ortDevice — CPU-written from mapped _modelInputBuffer
  wgpu::Buffer _ortInputBuffer;

  // ONNX Runtime session and IO binding (opaque — ort headers only in .mm)
  void* _session = nullptr;       // Ort::Session*
  void* _ioBinding = nullptr;     // Ort::IoBinding*
  void* _gpuMemInfo = nullptr;    // Ort::MemoryInfo* for WebGPU_Buf
  void* _boundInputTensor = nullptr;   // Ort::Value* — must outlive binding
  void* _boundOutputTensor = nullptr;  // Ort::Value* — must outlive binding

  // ORT output buffer on _ortDevice
  wgpu::Buffer _ortBuffer;
  // Staging buffer on _ortDevice — for MapAsync readback
  wgpu::Buffer _stagingBuffer;
  // Read buffer on _device (primary) — shader binds this, never contends with ORT
  wgpu::Buffer _readBuffer;
  size_t _outputElements = 0;

  // Input/output names (cached from model metadata)
  std::vector<std::string> _inputNames;
  std::vector<std::string> _outputNames;

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
  IOSurfaceRef _pendingIOSurface = nullptr;
  std::mutex _frameMutex;

  ModelSpec _spec;

  // FPS tracking
  int _inferenceCount = 0;
  double _lastFpsTime = 0;
};

}  // namespace dawn_pipeline

#endif // __cplusplus
