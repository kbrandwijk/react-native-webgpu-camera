#pragma once

#ifdef __cplusplus

#include <string>
#include <mutex>
#include <memory>
#include <atomic>
#include <CoreVideo/CVPixelBuffer.h>

// Forward declare to avoid pulling all Dawn headers into Swift bridging
namespace dawn_pipeline {

/// Opaque handle to the native compute pipeline.
/// All GPU work happens on the shared Dawn device from Skia Graphite.
class DawnComputePipeline {
public:
  DawnComputePipeline();
  ~DawnComputePipeline();

  /// Compile WGSL shader and create compute pipeline + output texture.
  /// Call once at setup time. Returns true on success.
  bool setup(const std::string &wgslCode, int width, int height);

  /// Import a camera CVPixelBuffer as a Dawn texture via IOSurface (zero-copy).
  /// Then dispatch the compute shader.
  /// Call from the camera frame callback (native thread).
  bool processFrame(CVPixelBufferRef pixelBuffer);

  /// Get the compute output as an opaque pointer to sk_sp<SkImage>.
  /// The returned pointer is only valid until the next processFrame call.
  /// Returns nullptr if no frame has been processed yet.
  void *getOutputSkImage();

  /// Get the raw output wgpu::Texture pointer (for wrapping as GPUTexture JSI object).
  /// Returns nullptr if pipeline not set up.
  void *getOutputTexturePtr();

  /// Tear down GPU resources.
  void cleanup();

  int getWidth() const { return _width; }
  int getHeight() const { return _height; }

  /// Shared liveness flag — checked by JSI lambda to avoid use-after-free.
  std::shared_ptr<std::atomic<bool>> getAliveFlag() const { return _alive; }

private:
  void cleanupLocked(); // called with _mutex already held

  struct Impl;
  Impl *_impl = nullptr;
  int _width = 0;
  int _height = 0;
  std::mutex _mutex;
  std::shared_ptr<std::atomic<bool>> _alive;
};

} // namespace dawn_pipeline

#endif // __cplusplus

// C interface for Swift interop
#ifdef __cplusplus
extern "C" {
#endif

typedef void *DawnComputePipelineRef;

DawnComputePipelineRef dawn_pipeline_create(void);
void dawn_pipeline_destroy(DawnComputePipelineRef ref);
bool dawn_pipeline_setup(DawnComputePipelineRef ref, const char *wgslCode, int width, int height);
bool dawn_pipeline_process_frame(DawnComputePipelineRef ref, CVPixelBufferRef pixelBuffer);
void *dawn_pipeline_get_output_image(DawnComputePipelineRef ref);
void *dawn_pipeline_get_output_texture(DawnComputePipelineRef ref);
void dawn_pipeline_cleanup(DawnComputePipelineRef ref);

/// Install JSI bindings for the compute pipeline on the given JS runtime.
/// Call once after the pipeline is created. Installs global.__webgpuCamera_getOutputTexture().
void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void *jsiRuntime);

#ifdef __cplusplus
}
#endif
