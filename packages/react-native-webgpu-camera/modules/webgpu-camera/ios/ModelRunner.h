#pragma once

#ifdef __cplusplus

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

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
   *  For async models, copies the texture ref and returns immediately.
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
  void runResizeShader(wgpu::Texture inputTexture);
  void runInference();

  wgpu::Device _device;
  int _cameraW, _cameraH;
  int _modelW = 0, _modelH = 0;  // model input spatial dims
  int _outputW = 0, _outputH = 0;

  // Resize + normalize compute shader
  wgpu::ComputePipeline _resizePipeline;
  wgpu::BindGroupLayout _resizeBindGroupLayout;
  wgpu::Buffer _modelInputBuffer;  // NCHW float32 storage buffer
  wgpu::Buffer _paramBuffer;       // uniform buffer for resize params
  wgpu::Sampler _resizeSampler;    // linear sampler for bilinear resize

  // Staging buffer for CPU readback of model input (CPU tensor fallback)
  wgpu::Buffer _stagingBuffer;

  // ONNX Runtime session (opaque -- ort headers only in .mm)
  void* _session = nullptr;       // Ort::Session*

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
  wgpu::Texture _pendingTexture;
  std::mutex _frameMutex;

  ModelSpec _spec;

  // FPS tracking
  int _inferenceCount = 0;
  double _lastFpsTime = 0;
};

}  // namespace dawn_pipeline

#endif // __cplusplus
