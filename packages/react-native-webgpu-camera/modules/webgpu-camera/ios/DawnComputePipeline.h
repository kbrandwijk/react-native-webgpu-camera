#pragma once

#ifdef __cplusplus

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <CoreVideo/CVPixelBuffer.h>
#include "ModelRunner.h"

namespace dawn_pipeline {

enum class ResourceType { Texture3D, Texture2D, StorageBuffer, CameraDepth };
enum class ResourceFormat { RGBA8Unorm, RGBA32Float };

struct ResourceSpec {
  ResourceType type;
  ResourceFormat format = ResourceFormat::RGBA8Unorm;
  std::vector<uint8_t> data;  // owns a copy of the upload data
  std::string fileUri;        // alternative: load from file (e.g. .cube LUT)
  int width = 0;
  int height = 0;
  int depth = 0;
};

enum class InputBindingType { Texture3D, Texture2D, Sampler, StorageBufferRead };

struct InputBinding {
  int bindingIndex = 0;
  InputBindingType type;
  int resourceHandle = -1;  // index into resources array
  int sourcePass = -1;      // pass that produced this buffer/texture
  int sourceBuffer = -1;    // global buffer index
  int modelOutput = -1;     // index into models array
};

struct PassInputSpec {
  int passIndex = 0;
  std::vector<InputBinding> bindings;
};

class DawnComputePipeline {
public:
  DawnComputePipeline();
  ~DawnComputePipeline();

  struct BufferSpec {
    int passIndex;
    int elementSize;
    int count;
  };

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
             const std::vector<ModelSpec>& models = {});

  bool processFrame(CVPixelBufferRef pixelBuffer);
  bool processFrame(CVPixelBufferRef pixelBuffer, CVPixelBufferRef depthBuffer);

  const void* readBuffer(int bufferIndex) const;
  int getBufferByteSize(int bufferIndex) const;

  void* getSkSurface();
  void flushCanvas();
  void* flushCanvasAndGetImage();

  void* getOutputSkImage();

  /** Set a WebGPU canvas context ID for direct-to-screen output.
   *  When set, processFrame blits the final compute output to the canvas
   *  surface and presents, skipping MakeImageFromTexture entirely. */
  void setCanvasContextId(int contextId);
  bool presentToCanvas(); // blit compute output to canvas surface (call from JS thread)

  /** Single-lock frame begin: returns image, buffers, canvas, fps, generation, metrics */
  struct FrameData {
    void* image = nullptr;         // sk_sp<SkImage>*
    void* surface = nullptr;       // sk_sp<SkSurface>*
    const void* bufferData[8] = {};
    int bufferByteSizes[8] = {};
    int bufferCount = 0;
    int pipelineFps = 0;
    int generation = 0;
    double metricLockWait = 0;
    double metricImport = 0;
    double metricBindGroup = 0;
    double metricCompute = 0;
    double metricBuffers = 0;
    double metricMakeImage = 0;
    double metricTotal = 0;
    double metricWall = 0;
  };
  FrameData beginFrame();

  void cleanup();

  int width() const { return _width; }
  int height() const { return _height; }
  int pipelineFps() const;
  int generation() const;

  // Per-step timing (ms) from last processFrame
  double metricLockWait() const;
  double metricImport() const;
  double metricBindGroup() const;
  double metricCompute() const;
  double metricBuffers() const;
  double metricMakeImage() const;
  double metricTotal() const;
  double metricWall() const;
  std::shared_ptr<std::atomic<bool>> alive() const { return _alive; }

private:
  void cleanupLocked();

  struct Impl;
  Impl* _impl = nullptr;
  std::mutex _mutex;
  int _width = 0;         // output (portrait) width
  int _height = 0;        // output (portrait) height
  int _inputWidth = 0;    // camera (landscape) width
  int _inputHeight = 0;   // camera (landscape) height
  int _canvasContextId = -1;  // WebGPU canvas context ID for direct output (-1 = disabled)
  std::shared_ptr<std::atomic<bool>> _alive;
};

} // namespace dawn_pipeline

#endif // __cplusplus

// C interface for Swift/ObjC bridge
#ifdef __cplusplus
extern "C" {
#endif

typedef void* DawnComputePipelineRef;

DawnComputePipelineRef dawn_pipeline_create(void);
void dawn_pipeline_destroy(DawnComputePipelineRef ref);

bool dawn_pipeline_setup_multipass(
  DawnComputePipelineRef ref,
  const char** shaders, int shaderCount,
  int width, int height,
  const int* bufferSpecs, int bufferCount,
  bool useCanvas, bool sync, bool appleLog, bool useDepth, bool lidarYUV,
  const void* resources, int resourceCount,
  const void* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount,
  const void* modelSpecs, int modelCount);

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref,
                                  CVPixelBufferRef pixelBuffer);

bool dawn_pipeline_process_frame_with_depth(DawnComputePipelineRef ref,
                                             CVPixelBufferRef pixelBuffer,
                                             CVPixelBufferRef depthBuffer);

const void* dawn_pipeline_read_buffer(DawnComputePipelineRef ref, int index);
int dawn_pipeline_get_buffer_byte_size(DawnComputePipelineRef ref, int index);

void* dawn_pipeline_get_sk_surface(DawnComputePipelineRef ref);
void dawn_pipeline_flush_canvas(DawnComputePipelineRef ref);

void* dawn_pipeline_get_output_image(DawnComputePipelineRef ref);

void dawn_pipeline_cleanup(DawnComputePipelineRef ref);

void dawn_pipeline_set_canvas_context_id(DawnComputePipelineRef ref, int contextId);

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime);

#ifdef __cplusplus
}
#endif
