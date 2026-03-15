#pragma once

#ifdef __cplusplus

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <CoreVideo/CVPixelBuffer.h>

namespace dawn_pipeline {

enum class ResourceType { Texture3D, Texture2D, StorageBuffer };

struct ResourceSpec {
  ResourceType type;
  std::vector<uint8_t> data;  // owns a copy of the upload data
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
             const std::vector<int>& textureOutputPasses = {});

  bool processFrame(CVPixelBufferRef pixelBuffer);

  const void* readBuffer(int bufferIndex) const;
  int getBufferByteSize(int bufferIndex) const;

  void* getSkSurface();
  void flushCanvas();
  void* flushCanvasAndGetImage();

  void* getOutputSkImage();

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
  int _width = 0;
  int _height = 0;
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
  bool useCanvas, bool sync,
  const void* resources, int resourceCount,
  const void* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount);

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref,
                                  CVPixelBufferRef pixelBuffer);

const void* dawn_pipeline_read_buffer(DawnComputePipelineRef ref, int index);
int dawn_pipeline_get_buffer_byte_size(DawnComputePipelineRef ref, int index);

void* dawn_pipeline_get_sk_surface(DawnComputePipelineRef ref);
void dawn_pipeline_flush_canvas(DawnComputePipelineRef ref);

void* dawn_pipeline_get_output_image(DawnComputePipelineRef ref);

void dawn_pipeline_cleanup(DawnComputePipelineRef ref);

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime);

#ifdef __cplusplus
}
#endif
