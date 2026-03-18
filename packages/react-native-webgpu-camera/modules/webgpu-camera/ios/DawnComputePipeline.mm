#include "DawnComputePipeline.h"

#include "rnskia/RNDawnContext.h"
#include "rnskia/RNSkPlatformContext.h"

#include "include/core/SkImage.h"
#include "include/core/SkSurface.h"
#include "include/core/SkCanvas.h"
#include "include/gpu/graphite/BackendTexture.h"
#include "include/gpu/graphite/Surface.h"

#include <jsi/jsi.h>
#include "api/JsiSkImage.h"
#include "api/JsiSkCanvas.h"

#include <CoreVideo/CoreVideo.h>
#include <CoreVideo/CVPixelBufferIOSurface.h>
#include <vector>
#include <fstream>
#include <sstream>

#import <Foundation/Foundation.h>
#import "SkiaManager.h"

namespace dawn_pipeline {

struct StagingBuffer {
  wgpu::Buffer gpuBuffer;
  wgpu::Buffer staging[2];
  int frameIndex = 0;
  int byteSize = 0;
  int elementSize = 0;
  int count = 0;
  int passIndex = 0;
  bool mapped[2] = {false, false};  // guarded by pipeline mutex
  const void* mappedData[2] = {nullptr, nullptr};
};

struct PassState {
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout bindGroupLayout;
  wgpu::BindGroup cachedBindGroup;  // cached for passes 1+ (static ping-pong textures)
  bool hasOutputBuffer = false;
  int bufferIndex = -1;
  bool hasTextureOutput = false;
  bool hasDynamicInputs = false;  // true if pass references per-frame resources (e.g. depth)
};

struct DawnComputePipeline::Impl {
  wgpu::Device device;
  std::vector<PassState> passes;
  wgpu::Texture texA;
  wgpu::Texture texB;
  std::vector<StagingBuffer> buffers;
  bool syncMode = false;
  bool useCanvas = false;
  bool appleLog = false;
  bool rawCamera = false;  // true when 0 shaders, no canvas, no buffers
  wgpu::Texture canvasTex;  // separate texture for Skia draws — not touched by compute
  sk_sp<SkSurface> surface;
  wgpu::Texture* finalTex = nullptr;
  sk_sp<SkImage> outputImage;
  sk_sp<SkImage> compositedImage;  // set by flushCanvas, consumed by next nextImage call

  // FPS tracking — counts processFrame completions per second
  int frameCount = 0;
  double lastFpsTime = 0;
  int currentFps = 0;

  // Generation counter — increments each processFrame completion
  int generation = 0;

  // Per-step timing (ms) — updated every frame, logged every second
  double t_lockWait = 0;
  double t_import = 0;
  double t_bindGroup = 0;
  double t_compute = 0;
  double t_buffers = 0;
  double t_makeImage = 0;
  double t_total = 0;
  double t_wall = 0;  // wall time including lock wait

  // Custom resources
  struct UploadedResource {
    ResourceType type;
    wgpu::Texture texture;
    wgpu::TextureView textureView;
    wgpu::Sampler sampler;
    wgpu::Buffer buffer;
  };
  std::vector<UploadedResource> uploadedResources;

  // Depth support
  bool useDepth = false;
  int depthResourceIndex = -1;
  CVPixelBufferRef currentDepthBuffer = nullptr;  // set per-frame before processFrame
  wgpu::Sampler depthSampler;
  wgpu::Texture depthTexture;   // per-frame
  wgpu::TextureView depthView;  // per-frame

  // Default linear sampler for texture outputs (always created if any custom inputs exist)
  wgpu::Sampler defaultSampler;

  // Per-pass custom input bindings (indexed by pass index)
  std::vector<std::vector<InputBinding>> passInputBindings;

  // Texture outputs from passes (indexed by pass index)
  std::vector<wgpu::Texture> passTextureOutputs;

  // Helper: append custom input binding entries for the given pass into the bind group entry vector.
  // Handles resource textures, samplers, storage buffers, and cross-pass texture outputs.
  void appendCustomInputEntries(
      int passIndex,
      std::vector<wgpu::BindGroupEntry>& entries) const {
    if ((size_t)passIndex >= passInputBindings.size()) return;
    for (const auto& ib : passInputBindings[passIndex]) {
    wgpu::BindGroupEntry entry{};
    entry.binding = ib.bindingIndex;

    switch (ib.type) {
      case InputBindingType::Texture3D:
      case InputBindingType::Texture2D:
        if (ib.resourceHandle >= 0 && ib.resourceHandle == depthResourceIndex && depthView) {
          // Per-frame depth texture from camera
          entry.textureView = depthView;
          entries.push_back(entry);
        } else if (ib.resourceHandle >= 0 && (size_t)ib.resourceHandle < uploadedResources.size()) {
          entry.textureView = uploadedResources[ib.resourceHandle].textureView;
          entries.push_back(entry);
        } else if (ib.sourcePass >= 0 && (size_t)ib.sourcePass < passTextureOutputs.size()
                   && passTextureOutputs[ib.sourcePass]) {
          entry.textureView = passTextureOutputs[ib.sourcePass].CreateView();
          entries.push_back(entry);
        }
        break;

      case InputBindingType::Sampler:
        if (ib.resourceHandle >= 0 && ib.resourceHandle == depthResourceIndex && depthSampler) {
          // Depth sampler for upsampling
          entry.sampler = depthSampler;
        } else if (ib.resourceHandle >= 0 && (size_t)ib.resourceHandle < uploadedResources.size()
            && uploadedResources[ib.resourceHandle].sampler) {
          entry.sampler = uploadedResources[ib.resourceHandle].sampler;
        } else {
          // Fall back to default sampler (for pass texture output samplers)
          entry.sampler = defaultSampler;
        }
        entries.push_back(entry);
        break;

      case InputBindingType::StorageBufferRead:
        if (ib.resourceHandle >= 0 && (size_t)ib.resourceHandle < uploadedResources.size()
            && uploadedResources[ib.resourceHandle].buffer) {
          entry.buffer = uploadedResources[ib.resourceHandle].buffer;
          entry.size = uploadedResources[ib.resourceHandle].buffer.GetSize();
          entries.push_back(entry);
        } else if (ib.sourceBuffer >= 0 && (size_t)ib.sourceBuffer < buffers.size()) {
          entry.buffer = buffers[ib.sourceBuffer].gpuBuffer;
          entry.size = buffers[ib.sourceBuffer].byteSize;
          entries.push_back(entry);
        }
        break;
    }
  }
  }
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
    bool useCanvas, bool sync,
    const std::vector<ResourceSpec>& resources,
    const std::vector<PassInputSpec>& passInputs,
    const std::vector<int>& textureOutputPasses,
    bool appleLog,
    bool useDepth) {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();

  // Camera delivers landscape; output is portrait (rotated 90° CW)
  _inputWidth = width;
  _inputHeight = height;
  _width = height;   // portrait width = landscape height
  _height = width;   // portrait height = landscape width
  _impl = new Impl();

  auto& ctx = RNSkia::DawnContext::getInstance();
  _impl->device = ctx.getWGPUDevice();
  _impl->syncMode = sync;
  _impl->useCanvas = useCanvas;
  _impl->appleLog = appleLog;
  _impl->useDepth = useDepth;

  // Find CameraDepth resource index and create depth sampler
  if (useDepth) {
    for (size_t i = 0; i < resources.size(); i++) {
      if (resources[i].type == ResourceType::CameraDepth) {
        _impl->depthResourceIndex = (int)i;
        break;
      }
    }

    // Create linear sampler for depth upsampling
    wgpu::SamplerDescriptor depthSampDesc{};
    depthSampDesc.magFilter = wgpu::FilterMode::Linear;
    depthSampDesc.minFilter = wgpu::FilterMode::Linear;
    depthSampDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    depthSampDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    depthSampDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    depthSampDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
    _impl->depthSampler = _impl->device.CreateSampler(&depthSampDesc);

    NSLog(@"[DawnPipeline] Depth enabled: resourceIndex=%d\n", _impl->depthResourceIndex);
  }

  // TODO: Raw camera path (0 shaders, no canvas, no buffers) needs texture
  // lifetime management — IOSurface texture is freed when processFrame returns.
  // For now, always use the compute path with auto-inserted passthrough shader.

  // Create ping-pong textures (portrait dimensions — rotation done in pass 0)
  wgpu::TextureDescriptor texDesc{};
  texDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
  texDesc.format = wgpu::TextureFormat::RGBA16Float;
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

  // Auto-insert passthrough shader when no user shaders but canvas/buffers needed.
  // Handles BGRA→RGBA conversion, rotates landscape→portrait (90° CW),
  // and gives onFrame a canvas to draw on.
  static const std::string kPassthroughWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) { return; }

  // 90° CW rotation: output(x, y) ← input(y, inW - 1 - x)
  let inDims = textureDimensions(inputTex);
  let srcCoord = vec2i(vec2u(id.y, inDims.y - 1u - id.x));
  textureStore(outputTex, vec2i(id.xy), textureLoad(inputTex, srcCoord, 0));
}
)";

  // Built-in YUV→RGB shader for Apple Log mode.
  // Converts 10-bit video range YCbCr (BT.2020) to Apple Log encoded RGB.
  // YUV unpack for Apple Log — recovers Apple Log encoded R'G'B' from 10-bit YCbCr.
  // Output is Apple Log encoded RGB — the flat/washed-out look that viewing LUTs expect.
  // Also rotates landscape→portrait (90° CW) so all downstream shaders work in portrait coords.
  static const std::string kYUVtoRGBWGSL = R"(
@group(0) @binding(0) var yPlaneTex: texture_2d<f32>;
@group(0) @binding(1) var uvPlaneTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) { return; }

  // 90° CW rotation: output(x, y) reads from input(y, inH - 1 - x)
  let yDims = textureDimensions(yPlaneTex);
  let srcCoord = vec2u(id.y, yDims.y - 1u - id.x);

  // Derive UV coord from plane dimensions — works for both 4:2:0 and 4:2:2
  let uvDims = textureDimensions(uvPlaneTex);
  let uvCoord = vec2i(vec2u(
    srcCoord.x * uvDims.x / yDims.x,
    srcCoord.y * uvDims.y / yDims.y
  ));

  let yRaw = textureLoad(yPlaneTex, vec2i(srcCoord), 0).r;
  let uvRaw = textureLoad(uvPlaneTex, uvCoord, 0).rg;

  // Video range expansion (10-bit: Y [64..940]/1023, CbCr [64..960]/1023)
  let y = (yRaw - 0.06256) / (0.91887 - 0.06256);
  let cb = (uvRaw.r - 0.06256) / (0.93842 - 0.06256) - 0.5;
  let cr = (uvRaw.g - 0.06256) / (0.93842 - 0.06256) - 0.5;

  // BT.2020 YCbCr -> R'G'B' (Apple Log encoded)
  let r = y + 1.4746 * cr;
  let g = y - 0.16455 * cb - 0.57135 * cr;
  let b = y + 1.8814 * cb;

  textureStore(outputTex, vec2i(id.xy), vec4f(r, g, b, 1.0));
}
)";

  NSLog(@"[DawnPipeline] setup() appleLog=%d, %zu user shaders, input=%dx%d, output=%dx%d (rotated)\n",
         appleLog, wgslShaders.size(), _inputWidth, _inputHeight, _width, _height);

  auto effectiveShaders = wgslShaders;
  if (appleLog) {
    // Auto-insert YUV→RGB (with rotation) as pass 0 before user shaders
    effectiveShaders.insert(effectiveShaders.begin(), kYUVtoRGBWGSL);
  } else {
    // Auto-insert passthrough (with rotation) as pass 0 for sRGB
    // Handles BGRA→RGBA conversion + landscape→portrait rotation
    effectiveShaders.insert(effectiveShaders.begin(), kPassthroughWGSL);
  }

  NSLog(@"[DawnPipeline] %zu effective shaders after prepend\n", effectiveShaders.size());

  // Check Dawn feature support
  if (appleLog) {
    bool hasMultiPlanar = _impl->device.HasFeature(wgpu::FeatureName::DawnMultiPlanarFormats);
    bool hasP010 = _impl->device.HasFeature(wgpu::FeatureName::MultiPlanarFormatP010);
    bool hasP210 = _impl->device.HasFeature(wgpu::FeatureName::MultiPlanarFormatP210);
    bool hasExtUsages = _impl->device.HasFeature(wgpu::FeatureName::MultiPlanarFormatExtendedUsages);
    NSLog(@"[DawnPipeline] Dawn features: MultiPlanar=%d, P010=%d, P210=%d, ExtUsages=%d\n",
           hasMultiPlanar, hasP010, hasP210, hasExtUsages);
  }

  // Compile shader passes
  for (size_t i = 0; i < effectiveShaders.size(); i++) {
    PassState pass;

    NSLog(@"[DawnPipeline] Compiling shader pass %zu (%zu chars)\n", i, effectiveShaders[i].size());

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = effectiveShaders[i].c_str();

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc;

    auto shaderModule = _impl->device.CreateShaderModule(&smDesc);
    if (!shaderModule) {
      NSLog(@"[DawnPipeline] FAILED to create shader module for pass %zu\n", i);
      cleanupLocked();
      return false;
    }

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = "main";

    pass.pipeline = _impl->device.CreateComputePipeline(&cpDesc);
    if (!pass.pipeline) {
      NSLog(@"[DawnPipeline] FAILED to create compute pipeline for pass %zu\n", i);
      cleanupLocked();
      return false;
    }

    pass.bindGroupLayout = pass.pipeline.GetBindGroupLayout(0);
    _impl->passes.push_back(std::move(pass));
    NSLog(@"[DawnPipeline] Pass %zu compiled OK\n", i);
  }

  // Create staging buffers for readback
  _impl->buffers.resize(bufferSpecs.size());
  for (size_t i = 0; i < bufferSpecs.size(); i++) {
    auto& spec = bufferSpecs[i];
    auto& sb = _impl->buffers[i];

    sb.passIndex = spec.passIndex;
    sb.elementSize = spec.elementSize;
    sb.count = spec.count;
    sb.byteSize = spec.elementSize * spec.count;

    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = sb.byteSize;
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufDesc.label = "OutputBuffer";
    sb.gpuBuffer = _impl->device.CreateBuffer(&bufDesc);

    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = sb.byteSize;
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    stagingDesc.label = "StagingA";
    sb.staging[0] = _impl->device.CreateBuffer(&stagingDesc);
    stagingDesc.label = "StagingB";
    sb.staging[1] = _impl->device.CreateBuffer(&stagingDesc);

    if (spec.passIndex >= 0 && spec.passIndex < (int)_impl->passes.size()) {
      _impl->passes[spec.passIndex].hasOutputBuffer = true;
      _impl->passes[spec.passIndex].bufferIndex = (int)i;
    }
  }

  // Determine final output texture
  _impl->finalTex = (effectiveShaders.size() % 2 != 0) ? &_impl->texA : &_impl->texB;

  // ── Upload custom resources ──
  if (!passInputs.empty()) {
    // Create default linear sampler (used for pass texture outputs)
    wgpu::SamplerDescriptor samplerDesc{};
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    samplerDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
    _impl->defaultSampler = _impl->device.CreateSampler(&samplerDesc);
  }

  _impl->uploadedResources.resize(resources.size());
  for (size_t ri = 0; ri < resources.size(); ri++) {
    auto spec = resources[ri];  // mutable copy — fileUri may populate data/dims
    auto& ur = _impl->uploadedResources[ri];
    ur.type = spec.type;

    // CameraDepth resources are bound per-frame from IOSurface — skip GPU upload
    if (spec.type == ResourceType::CameraDepth) continue;

    // Load .cube LUT file if fileUri is set
    if (!spec.fileUri.empty() && spec.data.empty()) {
      NSLog(@"[DawnPipeline] Loading .cube file: %s\n", spec.fileUri.c_str());
      std::ifstream file(spec.fileUri);
      if (!file.is_open()) {
        NSLog(@"[DawnPipeline] FAILED to open file: %s\n", spec.fileUri.c_str());
        cleanupLocked();
        return false;
      }

      int lutSize = 0;
      std::vector<float> values;
      std::string line;
      while (std::getline(file, line)) {
        // Trim
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.empty() || line[0] == '#') continue;

        if (line.rfind("LUT_3D_SIZE", 0) == 0) {
          lutSize = std::stoi(line.substr(11));
          continue;
        }

        // Skip metadata lines (TITLE, DOMAIN_MIN, etc.)
        if (line[0] >= 'A' && line[0] <= 'Z') continue;

        // Parse R G B triplet
        std::istringstream iss(line);
        float r, g, b;
        if (iss >> r >> g >> b) {
          values.push_back(r);
          values.push_back(g);
          values.push_back(b);
          values.push_back(1.0f);  // alpha
        }
      }

      if (lutSize == 0) {
        NSLog(@"[DawnPipeline] .cube file missing LUT_3D_SIZE\n");
        cleanupLocked();
        return false;
      }

      int expected = lutSize * lutSize * lutSize * 4;
      if ((int)values.size() != expected) {
        NSLog(@"[DawnPipeline] .cube: expected %d floats, got %zu\n", expected, values.size());
        cleanupLocked();
        return false;
      }

      // Copy float data into spec
      spec.width = lutSize;
      spec.height = lutSize;
      spec.depth = lutSize;
      spec.format = ResourceFormat::RGBA32Float;
      spec.data.resize(values.size() * sizeof(float));
      memcpy(spec.data.data(), values.data(), spec.data.size());

      NSLog(@"[DawnPipeline] Parsed .cube LUT: size=%d (%zu bytes)\n", lutSize, spec.data.size());
    }

    if (spec.type == ResourceType::Texture3D || spec.type == ResourceType::Texture2D) {
      wgpu::TextureFormat texFmt;
      int bytesPerPixel;
      // Use RGBA16Float for LUTs — RGBA32Float is not filterable in WebGPU,
      // so textureSampleLevel with a linear sampler would return zeros.
      // Convert float32 → float16 on upload. Precision loss is negligible for LUTs.
      std::vector<uint8_t> f16Data;
      if (spec.format == ResourceFormat::RGBA32Float) {
        texFmt = wgpu::TextureFormat::RGBA16Float;
        bytesPerPixel = 8;  // 4 halfs × 2 bytes
        // Convert float32 data to float16
        size_t floatCount = spec.data.size() / sizeof(float);
        f16Data.resize(floatCount * sizeof(uint16_t));
        const float *src = reinterpret_cast<const float *>(spec.data.data());
        uint16_t *dst = reinterpret_cast<uint16_t *>(f16Data.data());
        for (size_t i = 0; i < floatCount; i++) {
          // IEEE 754 float32 to float16 conversion
          uint32_t f32;
          memcpy(&f32, &src[i], 4);
          uint32_t sign = (f32 >> 16) & 0x8000;
          int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
          uint32_t mantissa = (f32 >> 13) & 0x3FF;
          if (exponent <= 0) {
            dst[i] = (uint16_t)sign; // flush to zero
          } else if (exponent >= 31) {
            dst[i] = (uint16_t)(sign | 0x7C00); // infinity
          } else {
            dst[i] = (uint16_t)(sign | (exponent << 10) | mantissa);
          }
        }
        // Replace spec data with f16 data for upload
        spec.data = std::move(f16Data);
        NSLog(@"[DawnPipeline] Converted LUT from RGBA32Float to RGBA16Float (%zu bytes)\n", spec.data.size());
      } else {
        texFmt = wgpu::TextureFormat::RGBA8Unorm;
        bytesPerPixel = 4;
      }

      wgpu::TextureDescriptor texDesc{};
      texDesc.format = texFmt;
      texDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
      texDesc.mipLevelCount = 1;
      texDesc.sampleCount = 1;

      if (spec.type == ResourceType::Texture3D) {
        texDesc.dimension = wgpu::TextureDimension::e3D;
        texDesc.size = {(uint32_t)spec.width, (uint32_t)spec.height, (uint32_t)spec.depth};
        texDesc.label = "CustomTexture3D";
      } else {
        texDesc.dimension = wgpu::TextureDimension::e2D;
        texDesc.size = {(uint32_t)spec.width, (uint32_t)spec.height, 1};
        texDesc.label = "CustomTexture2D";
      }

      ur.texture = _impl->device.CreateTexture(&texDesc);
      ur.textureView = ur.texture.CreateView();

      // Upload data via WriteTexture
      if (!spec.data.empty()) {
        wgpu::TexelCopyTextureInfo dst{};
        dst.texture = ur.texture;
        dst.mipLevel = 0;
        dst.origin = {0, 0, 0};
        dst.aspect = wgpu::TextureAspect::All;

        uint32_t bytesPerRow = (uint32_t)spec.width * bytesPerPixel;
        uint32_t rowsPerLayer = (uint32_t)spec.height;

        wgpu::TexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = bytesPerRow;
        layout.rowsPerImage = rowsPerLayer;

        wgpu::Extent3D extent = texDesc.size;
        _impl->device.GetQueue().WriteTexture(&dst, spec.data.data(), spec.data.size(), &layout, &extent);
      }

      // Create sampler for this texture resource
      wgpu::SamplerDescriptor sampDesc{};
      sampDesc.magFilter = wgpu::FilterMode::Linear;
      sampDesc.minFilter = wgpu::FilterMode::Linear;
      sampDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
      sampDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
      sampDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
      sampDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
      ur.sampler = _impl->device.CreateSampler(&sampDesc);

    } else {
      // StorageBuffer
      wgpu::BufferDescriptor bufDesc{};
      bufDesc.size = spec.data.size();
      bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      bufDesc.label = "CustomStorageBuffer";
      ur.buffer = _impl->device.CreateBuffer(&bufDesc);

      if (!spec.data.empty()) {
        _impl->device.GetQueue().WriteBuffer(ur.buffer, 0, spec.data.data(), spec.data.size());
      }
    }
  }

  // Store per-pass custom input bindings
  NSLog(@"[DawnPipeline] Storing %zu passInput specs across %zu passes\n",
         passInputs.size(), _impl->passes.size());
  _impl->passInputBindings.resize(_impl->passes.size());
  for (const auto& piSpec : passInputs) {
    NSLog(@"[DawnPipeline] passInput: passIndex=%d, %zu bindings\n",
           piSpec.passIndex, piSpec.bindings.size());
    if (piSpec.passIndex >= 0 && piSpec.passIndex < (int)_impl->passes.size()) {
      _impl->passInputBindings[piSpec.passIndex] = piSpec.bindings;
      // Check if any binding references a dynamic resource (e.g. cameraDepth)
      for (const auto& b : piSpec.bindings) {
        NSLog(@"[DawnPipeline] passInput binding: passIndex=%d, bindingIndex=%d, resourceHandle=%d, depthResourceIndex=%d\n",
              piSpec.passIndex, b.bindingIndex, b.resourceHandle, _impl->depthResourceIndex);
        if (b.resourceHandle >= 0 && b.resourceHandle == _impl->depthResourceIndex) {
          _impl->passes[piSpec.passIndex].hasDynamicInputs = true;
          NSLog(@"[DawnPipeline] Pass %d marked as hasDynamicInputs\n", piSpec.passIndex);
          break;
        }
      }
    } else {
      NSLog(@"[DawnPipeline] WARNING: passInput passIndex %d out of range (0..%zu)\n",
             piSpec.passIndex, _impl->passes.size() - 1);
    }
  }

  // Create texture outputs for designated passes
  _impl->passTextureOutputs.resize(_impl->passes.size());
  for (int passIdx : textureOutputPasses) {
    if (passIdx >= 0 && passIdx < (int)_impl->passes.size()) {
      _impl->passes[passIdx].hasTextureOutput = true;

      wgpu::TextureDescriptor texOutDesc{};
      texOutDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
      texOutDesc.format = wgpu::TextureFormat::RGBA16Float;
      texOutDesc.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding;
      texOutDesc.dimension = wgpu::TextureDimension::e2D;
      texOutDesc.mipLevelCount = 1;
      texOutDesc.sampleCount = 1;
      texOutDesc.label = "PassTextureOutput";
      _impl->passTextureOutputs[passIdx] = _impl->device.CreateTexture(&texOutDesc);
    }
  }

  // Cache bind groups for passes 1+ (static ping-pong textures).
  // Pass 0 reads from the camera input which changes every frame — can't cache.
  // Passes with dynamic inputs (e.g. depth) also can't be cached.
  NSLog(@"[DawnPipeline] Caching bind groups for passes 1..%zu\n", _impl->passes.size() - 1);
  for (size_t i = 1; i < _impl->passes.size(); i++) {
    auto& pass = _impl->passes[i];

    // Skip caching for passes with dynamic inputs — built per-frame in processFrame
    if (pass.hasDynamicInputs) {
      NSLog(@"[DawnPipeline] Pass %zu has dynamic inputs — skipping cache (built per-frame)\n", i);
      continue;
    }

    // Pass i reads from the output of pass i-1, writes to the next ping-pong texture.
    // Odd passes read texA write texB, even passes read texB write texA.
    bool readFromA = (i % 2 != 0);
    wgpu::Texture& readTex = readFromA ? _impl->texA : _impl->texB;
    wgpu::Texture& writeTex = readFromA ? _impl->texB : _impl->texA;

    std::vector<wgpu::BindGroupEntry> entries;
    wgpu::BindGroupEntry entry0{};
    entry0.binding = 0;
    entry0.textureView = readTex.CreateView();
    entries.push_back(entry0);

    wgpu::BindGroupEntry entry1{};
    entry1.binding = 1;
    entry1.textureView = writeTex.CreateView();
    entries.push_back(entry1);

    if (pass.hasOutputBuffer && pass.bufferIndex >= 0) {
      auto& sb = _impl->buffers[pass.bufferIndex];
      wgpu::BindGroupEntry entry2{};
      entry2.binding = 2;
      entry2.buffer = sb.gpuBuffer;
      entry2.size = sb.byteSize;
      entries.push_back(entry2);
    } else if (pass.hasTextureOutput && _impl->passTextureOutputs[i]) {
      wgpu::BindGroupEntry entry2{};
      entry2.binding = 2;
      entry2.textureView = _impl->passTextureOutputs[i].CreateView();
      entries.push_back(entry2);
    }

    // Append custom input bindings for this pass
    size_t beforeCustom = entries.size();
    _impl->appendCustomInputEntries((int)i, entries);
    NSLog(@"[DawnPipeline] Pass %zu cached bind group: %zu base entries + %zu custom = %zu total\n",
           i, beforeCustom, entries.size() - beforeCustom, entries.size());
    for (size_t e = 0; e < entries.size(); e++) {
      NSLog(@"[DawnPipeline]   entry[%zu]: binding=%u\n", e, entries[e].binding);
    }

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = pass.bindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    pass.cachedBindGroup = _impl->device.CreateBindGroup(&bgDesc);
    if (!pass.cachedBindGroup) {
      NSLog(@"[DawnPipeline] FAILED to create cached bind group for pass %zu\n", i);
      cleanupLocked();
      return false;
    }
    NSLog(@"[DawnPipeline] Pass %zu cached bind group OK\n", i);
  }

  // Create a separate texture for Skia canvas draws, isolated from compute output.
  // processFrame writes to finalTex; Skia draws go on canvasTex to avoid races.
  if (useCanvas) {
    wgpu::TextureDescriptor canvasTexDesc{};
    canvasTexDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
    canvasTexDesc.format = wgpu::TextureFormat::RGBA16Float;
    canvasTexDesc.usage = wgpu::TextureUsage::TextureBinding |
                          wgpu::TextureUsage::CopySrc |
                          wgpu::TextureUsage::CopyDst |
                          wgpu::TextureUsage::RenderAttachment;
    canvasTexDesc.label = "CanvasTex";
    _impl->canvasTex = _impl->device.CreateTexture(&canvasTexDesc);

    auto backendTex = skgpu::graphite::BackendTextures::MakeDawn(
      _impl->canvasTex.Get());
    _impl->surface = SkSurfaces::WrapBackendTexture(
      ctx.getRecorder(), backendTex,
      kRGBA_F16_SkColorType, nullptr, nullptr);
  }

  NSLog(@"[DawnPipeline] Setup complete: %zu passes, %zu buffers, %dx%d%s\n",
         effectiveShaders.size(), bufferSpecs.size(), width, height,
         useCanvas ? " +canvas" : "");
  return true;
}

bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer) {
  double tWallStart = CACurrentMediaTime();

  // Brief lock just to check _impl and grab stable references
  Impl* impl;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_impl || !pixelBuffer) return false;
    impl = _impl;
  }
  double tAfterLock = CACurrentMediaTime();

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto& device = impl->device;

  double tStart = tAfterLock;

  // ── All GPU work below — no mutex needed ──

  IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
  if (!ioSurface) return false;

  // CPU-side debug: read raw 10-bit Y/UV values from the CVPixelBuffer
  if (impl->appleLog) {
    static int debugCount = 0;
    if (debugCount < 5) {
      // Log the color space attached to this pixel buffer
      CFTypeRef csAttachment = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferCGColorSpaceKey, nullptr);
      if (csAttachment) {
        CGColorSpaceRef cs = (CGColorSpaceRef)csAttachment;
        CFStringRef csName = CGColorSpaceCopyName(cs);
        NSLog(@"[DawnPipeline] CVPixelBuffer colorSpace: %@", (__bridge NSString *)csName);
        if (csName) CFRelease(csName);
        CFRelease(csAttachment);
      } else {
        NSLog(@"[DawnPipeline] CVPixelBuffer colorSpace: (none attached)");
      }

      // Also check transfer function and matrix
      CFTypeRef tf = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, nullptr);
      CFTypeRef matrix = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, nullptr);
      CFTypeRef primaries = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, nullptr);
      NSLog(@"[DawnPipeline] CVPixelBuffer transfer=%@, matrix=%@, primaries=%@",
            tf ? (__bridge NSString *)tf : @"(none)",
            matrix ? (__bridge NSString *)matrix : @"(none)",
            primaries ? (__bridge NSString *)primaries : @"(none)");
      if (tf) CFRelease(tf);
      if (matrix) CFRelease(matrix);
      if (primaries) CFRelease(primaries);

      CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

      // Plane 0: Y (16-bit per pixel, only top 10 bits used)
      uint16_t *yPlane = (uint16_t *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
      size_t yBpr = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) / sizeof(uint16_t);
      size_t yW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
      size_t yH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);

      // Plane 1: UV interleaved (16-bit per component, half res)
      uint16_t *uvPlane = (uint16_t *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
      size_t uvBpr = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1) / sizeof(uint16_t);
      size_t uvW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1);
      size_t uvH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1);

      // Sample center pixel
      size_t cx = yW / 2, cy = yH / 2;
      uint16_t yVal = yPlane[cy * yBpr + cx];
      uint16_t cbVal = uvPlane[(cy / 2) * uvBpr + (cx / 2) * 2];
      uint16_t crVal = uvPlane[(cy / 2) * uvBpr + (cx / 2) * 2 + 1];

      // 10-bit values are in upper 10 bits: shift right by 6
      NSLog(@"[DawnPipeline] CPU RAW center: Y16=%u (10bit=%u, norm=%.4f) Cb16=%u (10bit=%u, norm=%.4f) Cr16=%u (10bit=%u, norm=%.4f) | Y plane %zux%zu bpr=%zu, UV plane %zux%zu bpr=%zu",
            yVal, yVal >> 6, yVal / 65535.0,
            cbVal, cbVal >> 6, cbVal / 65535.0,
            crVal, crVal >> 6, crVal / 65535.0,
            yW, yH, yBpr, uvW, uvH, uvBpr);

      // Sample a few more points
      uint16_t tlY = yPlane[100 * yBpr + 100];
      uint16_t brY = yPlane[(yH - 100) * yBpr + (yW - 100)];
      NSLog(@"[DawnPipeline] CPU RAW TL: Y16=%u (10bit=%u, norm=%.4f) BR: Y16=%u (10bit=%u, norm=%.4f)",
            tlY, tlY >> 6, tlY / 65535.0,
            brY, brY >> 6, brY / 65535.0);

      CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
      debugCount++;
    }
  }

  wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
  ioDesc.ioSurface = ioSurface;

  wgpu::SharedTextureMemoryDescriptor sharedDesc{};
  sharedDesc.nextInChain = &ioDesc;

  auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
  if (!sharedMemory) {
    static bool logged = false;
    if (!logged) { NSLog(@"[DawnPipeline] processFrame: ImportSharedTextureMemory FAILED\n"); logged = true; }
    return false;
  }

  wgpu::Texture inputTexture;
  wgpu::TextureView yPlaneView;   // only used in appleLog mode
  wgpu::TextureView uvPlaneView;  // only used in appleLog mode

  if (impl->appleLog) {
    // Detect 4:2:2 vs 4:2:0 from pixel buffer format
    OSType pixFmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    bool is422 = (pixFmt == kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange ||
                  pixFmt == kCVPixelFormatType_422YpCbCr10BiPlanarFullRange);

    static bool loggedFirst = false;
    if (!loggedFirst) {
      NSLog(@"[DawnPipeline] processFrame: Apple Log path, importing %dx%d YUV IOSurface (4:2:%s)\n",
            _inputWidth, _inputHeight, is422 ? "2" : "0");
    }

    // Import as multi-planar 10-bit YUV (4:2:2 or 4:2:0) — landscape dimensions
    wgpu::TextureDescriptor inputTexDesc{};
    inputTexDesc.size = {(uint32_t)_inputWidth, (uint32_t)_inputHeight, 1};
    inputTexDesc.format = is422
      ? wgpu::TextureFormat::R10X6BG10X6Biplanar422Unorm
      : wgpu::TextureFormat::R10X6BG10X6Biplanar420Unorm;
    inputTexDesc.usage = wgpu::TextureUsage::TextureBinding;
    inputTexDesc.dimension = wgpu::TextureDimension::e2D;
    inputTexDesc.mipLevelCount = 1;
    inputTexDesc.sampleCount = 1;
    inputTexDesc.label = is422 ? "CameraInputYUV422" : "CameraInputYUV420";

    inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
    if (!inputTexture) {
      if (!loggedFirst) { NSLog(@"[DawnPipeline] processFrame: CreateTexture (YUV) FAILED\n"); loggedFirst = true; }
      return false;
    }
    if (!loggedFirst) NSLog(@"[DawnPipeline] processFrame: YUV texture created OK\n");

    // Create per-plane views
    wgpu::TextureViewDescriptor yViewDesc{};
    yViewDesc.aspect = wgpu::TextureAspect::Plane0Only;
    yViewDesc.format = wgpu::TextureFormat::R16Unorm;
    yPlaneView = inputTexture.CreateView(&yViewDesc);
    if (!yPlaneView) {
      if (!loggedFirst) { NSLog(@"[DawnPipeline] processFrame: Y plane view FAILED\n"); loggedFirst = true; }
      return false;
    }

    wgpu::TextureViewDescriptor uvViewDesc{};
    uvViewDesc.aspect = wgpu::TextureAspect::Plane1Only;
    uvViewDesc.format = wgpu::TextureFormat::RG16Unorm;
    uvPlaneView = inputTexture.CreateView(&uvViewDesc);
    if (!uvPlaneView) {
      if (!loggedFirst) { NSLog(@"[DawnPipeline] processFrame: UV plane view FAILED\n"); loggedFirst = true; }
      return false;
    }
    if (!loggedFirst) {
      NSLog(@"[DawnPipeline] processFrame: Y + UV plane views created OK\n");
      loggedFirst = true;
    }
  } else {
    // Existing BGRA path — landscape dimensions
    wgpu::TextureDescriptor inputTexDesc{};
    inputTexDesc.size = {(uint32_t)_inputWidth, (uint32_t)_inputHeight, 1};
    inputTexDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    inputTexDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
    inputTexDesc.dimension = wgpu::TextureDimension::e2D;
    inputTexDesc.mipLevelCount = 1;
    inputTexDesc.sampleCount = 1;
    inputTexDesc.label = "CameraInput";

    inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
    if (!inputTexture) return false;
  }

  wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
  beginDesc.initialized = true;
  bool accessOk = sharedMemory.BeginAccess(inputTexture, &beginDesc);
  if (!accessOk) {
    static bool loggedAccessFail = false;
    if (!loggedAccessFail) {
      NSLog(@"[DawnPipeline] processFrame: BeginAccess FAILED for %s texture",
            impl->appleLog ? "YUV" : "BGRA");
      loggedAccessFail = true;
    }
    return false;
  }

  double tAfterImport = CACurrentMediaTime();

  // ── Import depth IOSurface when available ──
  wgpu::SharedTextureMemory depthSharedMemory;
  if (impl->useDepth && impl->currentDepthBuffer) {
    IOSurfaceRef depthSurface = CVPixelBufferGetIOSurface(impl->currentDepthBuffer);
    if (depthSurface) {
      size_t depthW = CVPixelBufferGetWidth(impl->currentDepthBuffer);
      size_t depthH = CVPixelBufferGetHeight(impl->currentDepthBuffer);

      wgpu::SharedTextureMemoryIOSurfaceDescriptor depthIoDesc{};
      depthIoDesc.ioSurface = depthSurface;
      wgpu::SharedTextureMemoryDescriptor depthSharedDesc{};
      depthSharedDesc.nextInChain = &depthIoDesc;

      depthSharedMemory = device.ImportSharedTextureMemory(&depthSharedDesc);
      if (depthSharedMemory) {
        wgpu::TextureDescriptor depthTexDesc{};
        depthTexDesc.size = {(uint32_t)depthW, (uint32_t)depthH, 1};
        depthTexDesc.format = wgpu::TextureFormat::R16Float;
        depthTexDesc.usage = wgpu::TextureUsage::TextureBinding;
        depthTexDesc.label = "CameraDepth";

        impl->depthTexture = depthSharedMemory.CreateTexture(&depthTexDesc);
        if (impl->depthTexture) {
          wgpu::SharedTextureMemoryBeginAccessDescriptor depthBeginDesc{};
          depthBeginDesc.initialized = true;
          depthSharedMemory.BeginAccess(impl->depthTexture, &depthBeginDesc);
          impl->depthView = impl->depthTexture.CreateView();

          static bool loggedDepth = false;
          if (!loggedDepth) {
            NSLog(@"[DawnPipeline] Depth texture imported: %zux%zu R16Float", depthW, depthH);
            loggedDepth = true;
          }
        }
      }
    }
  }

  // ── Raw camera path: no compute, just wrap the BGRA texture as SkImage ──
  if (impl->rawCamera) {
    auto outputImage = ctx.MakeImageFromTexture(
      inputTexture, _inputWidth, _inputHeight, wgpu::TextureFormat::BGRA8Unorm);

    wgpu::SharedTextureMemoryEndAccessState endState{};
    sharedMemory.EndAccess(inputTexture, &endState);

    // Brief lock to publish results
    std::lock_guard<std::mutex> lock(_mutex);
    impl->outputImage = std::move(outputImage);
    impl->frameCount++;
    impl->generation++;
    auto now = CACurrentMediaTime();
    if (impl->lastFpsTime == 0) {
      impl->lastFpsTime = now;
    } else if (now - impl->lastFpsTime >= 1.0) {
      impl->currentFps = (int)(impl->frameCount / (now - impl->lastFpsTime));
      impl->frameCount = 0;
      impl->lastFpsTime = now;
    }
    return true;
  }

  // ── Compute path ──

  // Process pending async callbacks from previous frame (e.g. MapAsync).
  // AllowProcessEvents callbacks fire during ProcessEvents(), not Tick().
  if (!impl->syncMode && !impl->buffers.empty()) {
    ctx.getWGPUInstance().ProcessEvents();
  }

  double tAfterBindGroup = tAfterImport; // updated after loop

  auto encoder = device.CreateCommandEncoder();

  for (size_t i = 0; i < impl->passes.size(); i++) {
    auto& pass = impl->passes[i];

    wgpu::BindGroup bindGroup;
    if (i == 0) {
      // Pass 0: camera input changes every frame — build bind group per frame
      bool writeToA = true;
      wgpu::Texture& writeTex = writeToA ? impl->texA : impl->texB;

      std::vector<wgpu::BindGroupEntry> entries;

      if (impl->appleLog) {
        // YUV→RGB pass: Y plane (binding 0) + UV plane (binding 1) + output (binding 2)
        wgpu::BindGroupEntry entry0{};
        entry0.binding = 0;
        entry0.textureView = yPlaneView;
        entries.push_back(entry0);

        wgpu::BindGroupEntry entry1{};
        entry1.binding = 1;
        entry1.textureView = uvPlaneView;
        entries.push_back(entry1);

        wgpu::BindGroupEntry entry2{};
        entry2.binding = 2;
        entry2.textureView = writeTex.CreateView();
        entries.push_back(entry2);
      } else {
        // Standard SDR pass 0: camera input (binding 0) + output (binding 1)
        wgpu::BindGroupEntry entry0{};
        entry0.binding = 0;
        entry0.textureView = inputTexture.CreateView();
        entries.push_back(entry0);

        wgpu::BindGroupEntry entry1{};
        entry1.binding = 1;
        entry1.textureView = writeTex.CreateView();
        entries.push_back(entry1);

        if (pass.hasOutputBuffer && pass.bufferIndex >= 0) {
          auto& sb = impl->buffers[pass.bufferIndex];
          wgpu::BindGroupEntry entry2{};
          entry2.binding = 2;
          entry2.buffer = sb.gpuBuffer;
          entry2.size = sb.byteSize;
          entries.push_back(entry2);
        } else if (pass.hasTextureOutput && impl->passTextureOutputs[0]) {
          wgpu::BindGroupEntry entry2{};
          entry2.binding = 2;
          entry2.textureView = impl->passTextureOutputs[0].CreateView();
          entries.push_back(entry2);
        }
      }

      // Append custom input bindings for pass 0
      size_t beforeCustom = entries.size();
      impl->appendCustomInputEntries(0, entries);

      static bool loggedPass0 = false;
      if (!loggedPass0) {
        NSLog(@"[DawnPipeline] processFrame pass 0: %zu base + %zu custom = %zu entries, appleLog=%d\n",
               beforeCustom, entries.size() - beforeCustom, entries.size(), impl->appleLog);
        for (size_t e = 0; e < entries.size(); e++) {
          NSLog(@"[DawnPipeline]   entry[%zu]: binding=%u\n", e, entries[e].binding);
        }
        loggedPass0 = true;
      }

      wgpu::BindGroupDescriptor bgDesc{};
      bgDesc.layout = pass.bindGroupLayout;
      bgDesc.entryCount = entries.size();
      bgDesc.entries = entries.data();
      bindGroup = device.CreateBindGroup(&bgDesc);
      if (!bindGroup) {
        NSLog(@"[DawnPipeline] FAILED to create bind group for pass 0\n");
        return false;
      }
    } else if (pass.hasDynamicInputs) {
      // Passes with dynamic inputs (e.g. depth): rebuild bind group every frame
      bool readFromA = (i % 2 != 0);
      wgpu::Texture& readTex = readFromA ? impl->texA : impl->texB;
      wgpu::Texture& writeTex = readFromA ? impl->texB : impl->texA;

      std::vector<wgpu::BindGroupEntry> entries;
      wgpu::BindGroupEntry entry0{};
      entry0.binding = 0;
      entry0.textureView = readTex.CreateView();
      entries.push_back(entry0);

      wgpu::BindGroupEntry entry1{};
      entry1.binding = 1;
      entry1.textureView = writeTex.CreateView();
      entries.push_back(entry1);

      if (pass.hasOutputBuffer && pass.bufferIndex >= 0) {
        auto& sb = impl->buffers[pass.bufferIndex];
        wgpu::BindGroupEntry entry2{};
        entry2.binding = 2;
        entry2.buffer = sb.gpuBuffer;
        entry2.size = sb.byteSize;
        entries.push_back(entry2);
      }

      impl->appendCustomInputEntries((int)i, entries);

      wgpu::BindGroupDescriptor bgDesc{};
      bgDesc.layout = pass.bindGroupLayout;
      bgDesc.entryCount = entries.size();
      bgDesc.entries = entries.data();
      bindGroup = device.CreateBindGroup(&bgDesc);

      static int dynLogCount = 0;
      if (dynLogCount < 3) {
        NSLog(@"[DawnPipeline] Dynamic pass %zu: %zu entries, bindGroup=%s, depthView=%s",
              i, entries.size(),
              bindGroup ? "OK" : "FAILED",
              impl->depthView ? "OK" : "nil");
        dynLogCount++;
      }

      if (!bindGroup) {
        // Dynamic input not available yet (e.g. depth not arrived) — skip this pass
        NSLog(@"[DawnPipeline] Skipping dynamic pass %zu — bind group creation failed", i);
        continue;
      }
    } else {
      // Passes 1+: cached bind group (static ping-pong textures)
      bindGroup = pass.cachedBindGroup;
    }

    auto computePass = encoder.BeginComputePass();
    computePass.SetPipeline(pass.pipeline);
    computePass.SetBindGroup(0, bindGroup);
    computePass.DispatchWorkgroups((_width + 15) / 16, (_height + 15) / 16);
    computePass.End();
  }

  tAfterBindGroup = CACurrentMediaTime();

  // Submit compute work
  auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  double tAfterCompute = CACurrentMediaTime();

  // Copy output buffers to staging in a separate submission
  if (!impl->buffers.empty()) {
    auto copyEncoder = device.CreateCommandEncoder();

    for (auto& sb : impl->buffers) {
      int stagingIdx = sb.frameIndex % 2;

      if (sb.mapped[stagingIdx]) {
        sb.staging[stagingIdx].Unmap();
        sb.mapped[stagingIdx] = false;
        sb.mappedData[stagingIdx] = nullptr;
      }

      copyEncoder.CopyBufferToBuffer(sb.gpuBuffer, 0, sb.staging[stagingIdx], 0, sb.byteSize);
    }

    auto copyCommands = copyEncoder.Finish();
    device.GetQueue().Submit(1, &copyCommands);

    // Map staging buffers for readback
    for (auto& sb : impl->buffers) {
      int stagingIdx = sb.frameIndex % 2;

      if (impl->syncMode) {
        auto instance = ctx.getWGPUInstance();
        auto future = sb.staging[stagingIdx].MapAsync(
          wgpu::MapMode::Read, 0, sb.byteSize,
          wgpu::CallbackMode::WaitAnyOnly,
          [&sb, stagingIdx](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
              sb.mappedData[stagingIdx] = sb.staging[stagingIdx].GetConstMappedRange(0, sb.byteSize);
              sb.mapped[stagingIdx] = true;
            }
          }
        );
        instance.WaitAny(future, UINT64_MAX);
      } else {
        // Async readback: AllowProcessEvents callback fires on next ProcessEvents() call.
        // Data arrives one frame behind — imperceptible for overlays.
        sb.staging[stagingIdx].MapAsync(
          wgpu::MapMode::Read, 0, sb.byteSize,
          wgpu::CallbackMode::AllowProcessEvents,
          [&sb, stagingIdx](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
              sb.mappedData[stagingIdx] = sb.staging[stagingIdx].GetConstMappedRange(0, sb.byteSize);
              sb.mapped[stagingIdx] = true;
            }
          }
        );
      }

      sb.frameIndex++;
    }
  }

  double tAfterBuffers = CACurrentMediaTime();

  // Determine final output texture and create SkImage
  bool finalIsA = (impl->passes.size() % 2 != 0);
  auto outputImage = ctx.MakeImageFromTexture(
    *(finalIsA ? &impl->texA : &impl->texB), _width, _height, wgpu::TextureFormat::RGBA16Float);

  double tAfterMakeImage = CACurrentMediaTime();

  // Cleanup IOSurface access
  wgpu::SharedTextureMemoryEndAccessState endState{};
  sharedMemory.EndAccess(inputTexture, &endState);

  // Cleanup depth IOSurface access
  if (depthSharedMemory && impl->depthTexture) {
    wgpu::SharedTextureMemoryEndAccessState depthEndState{};
    depthSharedMemory.EndAccess(impl->depthTexture, &depthEndState);
    impl->depthTexture = nullptr;
    impl->depthView = nullptr;
  }
  impl->currentDepthBuffer = nullptr;

  // ── Brief lock to publish results ──
  {
    std::lock_guard<std::mutex> lock(_mutex);
    impl->finalTex = finalIsA ? &impl->texA : &impl->texB;
    impl->outputImage = std::move(outputImage);

    // Per-step timing (ms)
    impl->t_lockWait = (tAfterLock - tWallStart) * 1000.0;
    impl->t_import = (tAfterImport - tStart) * 1000.0;
    impl->t_bindGroup = (tAfterBindGroup - tAfterImport) * 1000.0;
    impl->t_compute = (tAfterCompute - tAfterBindGroup) * 1000.0;
    impl->t_buffers = (tAfterBuffers - tAfterCompute) * 1000.0;
    impl->t_makeImage = (tAfterMakeImage - tAfterBuffers) * 1000.0;
    impl->t_total = (tAfterMakeImage - tStart) * 1000.0;
    impl->t_wall = (tAfterMakeImage - tWallStart) * 1000.0;

    // FPS tracking + generation counter
    impl->frameCount++;
    impl->generation++;
    auto now = CACurrentMediaTime();
    if (impl->lastFpsTime == 0) {
      impl->lastFpsTime = now;
    } else if (now - impl->lastFpsTime >= 1.0) {
      impl->currentFps = (int)(impl->frameCount / (now - impl->lastFpsTime));
      impl->frameCount = 0;
      impl->lastFpsTime = now;
    }
  }

  return true;
}

bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer, CVPixelBufferRef depthBuffer) {
  if (_impl) {
    _impl->currentDepthBuffer = depthBuffer;
  }
  return processFrame(pixelBuffer);
}

const void* DawnComputePipeline::readBuffer(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return nullptr;

  auto& sb = _impl->buffers[bufferIndex];
  // Double-buffer read: frameIndex points past the last write.
  // The Tick at the start of frame N+1 completes the map from frame N.
  // After Tick + unmap/write/map cycle, the *other* staging buffer holds completed data.
  // Try both — return whichever is mapped.
  for (int i = 0; i < 2; i++) {
    if (sb.mapped[i] && sb.mappedData[i]) return sb.mappedData[i];
  }
  return nullptr;
}

int DawnComputePipeline::getBufferByteSize(int bufferIndex) const {
  if (!_impl || bufferIndex < 0 || bufferIndex >= (int)_impl->buffers.size())
    return 0;
  return _impl->buffers[bufferIndex].byteSize;
}

int DawnComputePipeline::pipelineFps() const {
  if (!_impl) return 0;
  return _impl->currentFps;
}

int DawnComputePipeline::generation() const {
  if (!_impl) return 0;
  return _impl->generation;
}

double DawnComputePipeline::metricLockWait() const { return _impl ? _impl->t_lockWait : 0; }
double DawnComputePipeline::metricImport() const { return _impl ? _impl->t_import : 0; }
double DawnComputePipeline::metricBindGroup() const { return _impl ? _impl->t_bindGroup : 0; }
double DawnComputePipeline::metricCompute() const { return _impl ? _impl->t_compute : 0; }
double DawnComputePipeline::metricBuffers() const { return _impl ? _impl->t_buffers : 0; }
double DawnComputePipeline::metricMakeImage() const { return _impl ? _impl->t_makeImage : 0; }
double DawnComputePipeline::metricTotal() const { return _impl ? _impl->t_total : 0; }
double DawnComputePipeline::metricWall() const { return _impl ? _impl->t_wall : 0; }

void* DawnComputePipeline::getSkSurface() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;
  return &_impl->surface;
}

void DawnComputePipeline::flushCanvas() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl || !_impl->surface) return;

  auto& ctx = RNSkia::DawnContext::getInstance();
  auto recording = _impl->surface->recorder()->snap();
  if (recording) {
    ctx.submitRecording(recording.get(), skgpu::graphite::SyncToCpu::kNo);
  }

  if (_impl->finalTex) {
    // Store as compositedImage so the next nextImage() call returns this
    // instead of outputImage (which processFrame may overwrite on the camera thread).
    _impl->compositedImage = ctx.MakeImageFromTexture(
      *_impl->finalTex, _width, _height, wgpu::TextureFormat::RGBA16Float
    );
  }
}

DawnComputePipeline::FrameData DawnComputePipeline::beginFrame() {
  FrameData fd;
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return fd;

  // Process pending async buffer maps
  if (!_impl->syncMode && !_impl->buffers.empty()) {
    auto& ctx = RNSkia::DawnContext::getInstance();
    ctx.getWGPUInstance().ProcessEvents();
  }

  // Image
  if (_impl->compositedImage) {
    _impl->outputImage = std::move(_impl->compositedImage);
    _impl->compositedImage = nullptr;
  }
  if (_impl->outputImage) {
    fd.image = &_impl->outputImage;
  }

  // Buffers
  fd.bufferCount = std::min((int)_impl->buffers.size(), 8);
  for (int bi = 0; bi < fd.bufferCount; bi++) {
    auto& sb = _impl->buffers[bi];
    fd.bufferByteSizes[bi] = sb.byteSize;
    fd.bufferData[bi] = nullptr;
    for (int si = 0; si < 2; si++) {
      if (sb.mapped[si] && sb.mappedData[si]) {
        fd.bufferData[bi] = sb.mappedData[si];
        break;
      }
    }
  }

  // Canvas surface
  if (_impl->surface) {
    fd.surface = &_impl->surface;
  }

  // FPS + generation + metrics — all in the same lock
  fd.pipelineFps = _impl->currentFps;
  fd.generation = _impl->generation;
  fd.metricLockWait = _impl->t_lockWait;
  fd.metricImport = _impl->t_import;
  fd.metricBindGroup = _impl->t_bindGroup;
  fd.metricCompute = _impl->t_compute;
  fd.metricBuffers = _impl->t_buffers;
  fd.metricMakeImage = _impl->t_makeImage;
  fd.metricTotal = _impl->t_total;
  fd.metricWall = _impl->t_wall;

  return fd;
}

void* DawnComputePipeline::flushCanvasAndGetImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;

  if (_impl->surface && _impl->finalTex && _impl->canvasTex) {
    auto& ctx = RNSkia::DawnContext::getInstance();

    // Copy compute output → canvasTex so Skia draws composite on a stable snapshot.
    // This prevents the race where processFrame overwrites finalTex mid-draw.
    auto encoder = _impl->device.CreateCommandEncoder();
    wgpu::TexelCopyTextureInfo src{};
    src.texture = *_impl->finalTex;
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = _impl->canvasTex;
    wgpu::Extent3D extent = {(uint32_t)_width, (uint32_t)_height, 1};
    encoder.CopyTextureToTexture(&src, &dst, &extent);
    auto commands = encoder.Finish();
    _impl->device.GetQueue().Submit(1, &commands);

    // Flush Skia draws (recorded by onFrame) onto canvasTex
    auto recording = _impl->surface->recorder()->snap();
    if (recording) {
      ctx.submitRecording(recording.get(), skgpu::graphite::SyncToCpu::kNo);
    }

    // Wrap canvasTex (compute + Skia draws) as the output image
    _impl->outputImage = ctx.MakeImageFromTexture(
      _impl->canvasTex, _width, _height, wgpu::TextureFormat::RGBA16Float
    );
  }

  if (!_impl->outputImage) return nullptr;
  return &_impl->outputImage;
}

void* DawnComputePipeline::getOutputSkImage() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_impl) return nullptr;

  // Prefer composited image (from flushCanvas) — consume it so it's one-shot
  if (_impl->compositedImage) {
    _impl->outputImage = std::move(_impl->compositedImage);
    _impl->compositedImage = nullptr;
  }

  if (!_impl->outputImage) return nullptr;
  return &_impl->outputImage;
}

void DawnComputePipeline::cleanup() {
  std::lock_guard<std::mutex> lock(_mutex);
  cleanupLocked();
}

void DawnComputePipeline::cleanupLocked() {
  if (!_impl) return;

  for (auto& sb : _impl->buffers) {
    for (int j = 0; j < 2; j++) {
      if (sb.mapped[j]) {
        sb.staging[j].Unmap();
        sb.mapped[j] = false;
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
  _impl->uploadedResources.clear();
  _impl->passTextureOutputs.clear();
  _impl->passInputBindings.clear();
  _impl->defaultSampler = nullptr;
  _impl->depthSampler = nullptr;
  _impl->depthTexture = nullptr;
  _impl->depthView = nullptr;
  _impl->currentDepthBuffer = nullptr;

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
    bool useCanvas, bool sync, bool appleLog, bool useDepth,
    const void* resourcesPtr, int resourceCount,
    const void* passInputsPtr, int passInputCount,
    const int* textureOutputPassesPtr, int textureOutputPassCount) {
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

  // Cast void* back to typed C++ vectors
  const dawn_pipeline::ResourceSpec* resources =
    static_cast<const dawn_pipeline::ResourceSpec*>(resourcesPtr);
  const dawn_pipeline::PassInputSpec* passInputs =
    static_cast<const dawn_pipeline::PassInputSpec*>(passInputsPtr);

  std::vector<dawn_pipeline::ResourceSpec> resourceVec;
  if (resourceCount > 0 && resources) {
    resourceVec.assign(resources, resources + resourceCount);
  }
  std::vector<dawn_pipeline::PassInputSpec> passInputVec;
  if (passInputCount > 0 && passInputs) {
    passInputVec.assign(passInputs, passInputs + passInputCount);
  }
  std::vector<int> texOutVec;
  if (textureOutputPassCount > 0 && textureOutputPassesPtr) {
    texOutVec.assign(textureOutputPassesPtr, textureOutputPassesPtr + textureOutputPassCount);
  }

  return pipeline->setup(wgslShaders, width, height, specs, useCanvas, sync,
                         resourceVec, passInputVec, texOutVec, appleLog, useDepth);
}

bool dawn_pipeline_process_frame(DawnComputePipelineRef ref, CVPixelBufferRef pixelBuffer) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->processFrame(pixelBuffer);
}

bool dawn_pipeline_process_frame_with_depth(DawnComputePipelineRef ref,
                                             CVPixelBufferRef pixelBuffer,
                                             CVPixelBufferRef depthBuffer) {
  if (!ref) return false;
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->processFrame(pixelBuffer, depthBuffer);
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

    if (propName == "flushCanvasAndGetImage") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto* imgPtr = static_cast<sk_sp<SkImage>*>(
            _pipeline->flushCanvasAndGetImage());
          if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

          auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
            _platformContext, *imgPtr);
          return facebook::jsi::Object::createFromHostObject(rt, hostObj);
        });
    }

    if (propName == "beginFrame") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();

          auto fd = _pipeline->beginFrame();

          auto result = facebook::jsi::Object(rt);

          // image
          if (fd.image) {
            auto* imgPtr = static_cast<sk_sp<SkImage>*>(fd.image);
            if (imgPtr && *imgPtr) {
              auto hostObj = std::make_shared<RNSkia::JsiSkImage>(
                _platformContext, *imgPtr);
              result.setProperty(rt, "image",
                facebook::jsi::Object::createFromHostObject(rt, hostObj));
            }
          }

          // canvas
          if (fd.surface) {
            auto* surfPtr = static_cast<sk_sp<SkSurface>*>(fd.surface);
            if (surfPtr && *surfPtr) {
              SkCanvas* canvas = (*surfPtr)->getCanvas();
              if (canvas) {
                auto hostObj = std::make_shared<RNSkia::JsiSkCanvas>(
                  _platformContext, canvas);
                result.setProperty(rt, "canvas",
                  facebook::jsi::Object::createFromHostObject(rt, hostObj));
              }
            }
          }

          // buffers as array
          auto bufsArr = facebook::jsi::Array(rt, fd.bufferCount);
          for (int i = 0; i < fd.bufferCount; i++) {
            if (fd.bufferData[i] && fd.bufferByteSizes[i] > 0) {
              auto arrayBuffer = rt.global()
                .getPropertyAsFunction(rt, "ArrayBuffer")
                .callAsConstructor(rt, fd.bufferByteSizes[i])
                .asObject(rt)
                .getArrayBuffer(rt);
              memcpy(arrayBuffer.data(rt), fd.bufferData[i], fd.bufferByteSizes[i]);
              bufsArr.setValueAtIndex(rt, i, std::move(arrayBuffer));
            } else {
              bufsArr.setValueAtIndex(rt, i, facebook::jsi::Value::null());
            }
          }
          result.setProperty(rt, "buffers", std::move(bufsArr));

          // FPS + generation + metrics
          result.setProperty(rt, "pipelineFps", fd.pipelineFps);
          result.setProperty(rt, "generation", fd.generation);

          auto m = facebook::jsi::Object(rt);
          m.setProperty(rt, "lockWait", fd.metricLockWait);
          m.setProperty(rt, "import", fd.metricImport);
          m.setProperty(rt, "bindGroup", fd.metricBindGroup);
          m.setProperty(rt, "compute", fd.metricCompute);
          m.setProperty(rt, "buffers", fd.metricBuffers);
          m.setProperty(rt, "makeImage", fd.metricMakeImage);
          m.setProperty(rt, "total", fd.metricTotal);
          m.setProperty(rt, "wall", fd.metricWall);
          result.setProperty(rt, "metrics", std::move(m));

          return result;
        });
    }

    if (propName == "pipelineFps") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime&,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value(0);
          return facebook::jsi::Value(_pipeline->pipelineFps());
        });
    }

    if (propName == "generation") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime&,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value(0);
          return facebook::jsi::Value(_pipeline->generation());
        });
    }

    if (propName == "metrics") {
      return facebook::jsi::Function::createFromHostFunction(
        runtime, name, 0,
        [this](facebook::jsi::Runtime& rt,
               const facebook::jsi::Value&,
               const facebook::jsi::Value*,
               size_t) -> facebook::jsi::Value {
          if (!_alive->load()) return facebook::jsi::Value::null();
          auto obj = facebook::jsi::Object(rt);
          obj.setProperty(rt, "lockWait", _pipeline->metricLockWait());
          obj.setProperty(rt, "import", _pipeline->metricImport());
          obj.setProperty(rt, "bindGroup", _pipeline->metricBindGroup());
          obj.setProperty(rt, "compute", _pipeline->metricCompute());
          obj.setProperty(rt, "buffers", _pipeline->metricBuffers());
          obj.setProperty(rt, "makeImage", _pipeline->metricMakeImage());
          obj.setProperty(rt, "total", _pipeline->metricTotal());
          obj.setProperty(rt, "wall", _pipeline->metricWall());
          return obj;
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

void dawn_pipeline_install_jsi(DawnComputePipelineRef ref, void* jsiRuntime) {
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  auto& runtime = *static_cast<facebook::jsi::Runtime*>(jsiRuntime);

  std::shared_ptr<RNSkia::RNSkManager> skManager = [SkiaManager latestActiveSkManager];
  if (!skManager) {
    NSLog(@"[DawnPipeline] WARNING: SkiaManager not available\n");
    return;
  }
  auto platformContext = skManager->getPlatformContext();
  auto alive = pipeline->alive();

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

      auto* imgPtr = static_cast<sk_sp<SkImage>*>(pipeline->getOutputSkImage());
      if (!imgPtr || !*imgPtr) return facebook::jsi::Value::null();

      auto hostObj = std::make_shared<RNSkia::JsiSkImage>(platformContext, *imgPtr);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_nextImage", std::move(nextImageFn));

  auto createStreamFn = facebook::jsi::Function::createFromHostFunction(
    runtime,
    facebook::jsi::PropNameID::forAscii(runtime, "__webgpuCamera_createStream"),
    0,
    [pipeline, alive, platformContext](
      facebook::jsi::Runtime& rt,
      const facebook::jsi::Value&,
      const facebook::jsi::Value*,
      size_t) -> facebook::jsi::Value {
      auto hostObj = std::make_shared<CameraStreamHostObject>(pipeline, alive, platformContext);
      return facebook::jsi::Object::createFromHostObject(rt, hostObj);
    });
  runtime.global().setProperty(runtime, "__webgpuCamera_createStream", std::move(createStreamFn));

  NSLog(@"[DawnPipeline] JSI bindings installed (nextImage + createStream)\n");
}

} // extern "C"
