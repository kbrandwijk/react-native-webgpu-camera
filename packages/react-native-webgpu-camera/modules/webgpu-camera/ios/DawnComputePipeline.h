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

  struct BufferSpec {
    int passIndex;
    int elementSize;
    int count;
  };

  bool setup(const std::vector<std::string>& wgslShaders,
             int width, int height,
             const std::vector<BufferSpec>& bufferSpecs,
             bool useCanvas, bool sync);

  bool processFrame(CVPixelBufferRef pixelBuffer);

  const void* readBuffer(int bufferIndex) const;
  int getBufferByteSize(int bufferIndex) const;

  void* getSkSurface();
  void flushCanvas();

  void* getOutputSkImage();

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

typedef void* DawnComputePipelineRef;

extern "C" {
  DawnComputePipelineRef dawn_pipeline_create();
  void dawn_pipeline_destroy(DawnComputePipelineRef ref);

  bool dawn_pipeline_setup_multipass(
    DawnComputePipelineRef ref,
    const char** shaders, int shaderCount,
    int width, int height,
    const int* bufferSpecs, int bufferCount,
    bool useCanvas, bool sync);

  bool dawn_pipeline_process_frame(DawnComputePipelineRef ref,
                                    CVPixelBufferRef pixelBuffer);

  const void* dawn_pipeline_read_buffer(DawnComputePipelineRef ref, int index);
  int dawn_pipeline_get_buffer_byte_size(DawnComputePipelineRef ref, int index);

  void* dawn_pipeline_get_sk_surface(DawnComputePipelineRef ref);
  void dawn_pipeline_flush_canvas(DawnComputePipelineRef ref);

  void* dawn_pipeline_get_output_image(DawnComputePipelineRef ref);

  void dawn_pipeline_cleanup(DawnComputePipelineRef ref);

  void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime);
}
