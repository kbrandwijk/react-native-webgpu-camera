# Apple Log + LUT Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support native Apple Log camera input (10-bit YUV) with .cube LUT file support for display preview, preserving full dynamic range through the shader pipeline.

**Architecture:** Two pipeline modes (SDR/Apple Log) selected by color space. Apple Log imports the IOSurface as a multi-planar texture (`R10X6BG10X6Biplanar420Unorm`), runs an automatic YUV→RGB conversion shader, and processes through `RGBA16Float` ping-pong textures. LUTs are loaded via the existing `inputs` API as 3D textures with `RGBA32Float` format.

**Tech Stack:** Dawn/WebGPU (multi-planar format features), AVFoundation (Apple Log capture), WGSL compute shaders, TypeScript/.cube parser

**Spec:** `docs/superpowers/specs/2026-03-15-apple-log-lut-design.md`

---

## Task 1: Enable Dawn Multi-Planar Format Features

The Dawn device (created by Skia in `RNDawnUtils.h`) must request multi-planar format features so we can import YUV IOSurfaces. Without these features, `ImportSharedTextureMemory` will reject the bi-planar texture.

**Files:**
- Modify: `packages/react-native-skia/packages/skia/cpp/rnskia/RNDawnUtils.h:176-185`

- [ ] **Step 1: Add multi-planar feature requests to Dawn device creation**

In `RNDawnUtils.h`, after the `SharedTextureMemoryIOSurface` feature check (line 178), add multi-planar format features inside the `#ifdef __APPLE__` block:

```cpp
#ifdef __APPLE__
  if (adapter.HasFeature(wgpu::FeatureName::SharedTextureMemoryIOSurface)) {
    features.push_back(wgpu::FeatureName::SharedTextureMemoryIOSurface);
  }
  if (adapter.HasFeature(wgpu::FeatureName::DawnMultiPlanarFormats)) {
    features.push_back(wgpu::FeatureName::DawnMultiPlanarFormats);
  }
  if (adapter.HasFeature(wgpu::FeatureName::MultiPlanarFormatExtendedUsages)) {
    features.push_back(wgpu::FeatureName::MultiPlanarFormatExtendedUsages);
  }
#else
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-skia/packages/skia/cpp/rnskia/RNDawnUtils.h
git commit -m "feat: enable Dawn multi-planar format features for Apple Log YUV import"
```

---

## Task 2: Add `appleLog` Flag Through the Config Pipeline (TypeScript → Swift → C++)

Thread an `appleLog` boolean from the JS pipeline config all the way to `DawnComputePipeline::setup()`. This flag controls: (a) which pixel format `AVCaptureVideoDataOutput` requests, (b) whether ping-pong textures are `RGBA16Float`, and (c) whether the auto YUV→RGB pass is inserted.

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/types.ts:15-22`
- Modify: `packages/react-native-webgpu-camera/src/useCamera.ts`
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts:240-316,404-456`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts:12-37`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift:133-182`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h:50-56`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm:156-163`

- [ ] **Step 1: Add `colorSpace` to `CameraHandle` in types.ts**

In `packages/react-native-webgpu-camera/src/types.ts`, add `colorSpace` to the `CameraHandle` interface:

```typescript
export interface CameraHandle {
  isReady: boolean;
  width: number;
  height: number;
  fps: number;
  colorSpace: ColorSpace;
}
```

- [ ] **Step 2: Expose `colorSpace` from `useCamera`**

In `packages/react-native-webgpu-camera/src/useCamera.ts`, add `colorSpace` to the return value:

```typescript
export function useCamera(config: CameraConfig): CameraHandle {
  // ... existing state ...
  const colorSpace = config.colorSpace ?? 'sRGB';

  // ... existing useEffect ...

  return {
    isReady,
    width: resolvedWidth,
    height: resolvedHeight,
    fps: resolvedFps,
    colorSpace,
  };
}
```

- [ ] **Step 3: Add `appleLog` to the native module config interface**

In `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`, add `appleLog` to the `setupMultiPassPipeline` config:

```typescript
setupMultiPassPipeline(config: {
  shaders: string[];
  width: number;
  height: number;
  buffers: [number, number, number][];
  useCanvas: boolean;
  sync: boolean;
  appleLog: boolean;  // <-- add this
  resources: { /* ... existing ... */ }[];
  passInputs: { /* ... existing ... */ }[];
  textureOutputPasses: number[];
}): boolean;
```

- [ ] **Step 4: Thread `appleLog` through `buildNativeConfig` and the setup effect**

In `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`:

Add `appleLog` parameter to `buildNativeConfig`:

```typescript
function buildNativeConfig(
  passes: CapturedPass[],
  width: number,
  height: number,
  useCanvas: boolean,
  sync: boolean,
  capturedResources: CapturedResource[],
  appleLog: boolean,  // <-- add
) {
  // ... existing body ...
  return { shaders, width, height, buffers, useCanvas, sync, appleLog, resources, passInputs, textureOutputPasses };
}
```

In the `useEffect` (around line 431), pass `appleLog`:

```typescript
const appleLog = camera.colorSpace === 'appleLog';

const nativeConfig = buildNativeConfig(
  passes,
  camera.width,
  camera.height,
  useCanvas,
  sync,
  capturedResources,
  appleLog,
);
```

Add `camera.colorSpace` to the dependency array (line 456):

```typescript
}, [camera.isReady, camera.width, camera.height, camera.fps, camera.colorSpace]);
```

- [ ] **Step 5: Read `appleLog` in Swift and forward to the bridge**

In `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`, in the `setupMultiPassPipeline` function (around line 133):

After existing config extraction, add:

```swift
let appleLog = config["appleLog"] as? Bool ?? false
```

Pass to the bridge call (add parameter):

```swift
let ok = bridge.setupMultiPass(
  withShaders: shaders,
  width: Int32(width),
  height: Int32(height),
  bufferSpecs: bufferSpecs,
  useCanvas: useCanvas,
  sync: sync,
  appleLog: appleLog,
  resources: resourcesRaw,
  passInputs: passInputsRaw,
  textureOutputPasses: textureOutputPasses
)
```

- [ ] **Step 6: Update DawnPipelineBridge to accept and forward `appleLog`**

In `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`, add `appleLog` parameter:

```objc
- (BOOL)setupMultiPassWithShaders:(nonnull NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(nonnull NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                         appleLog:(BOOL)appleLog
                        resources:(nonnull NSArray<NSDictionary *> *)resources
                       passInputs:(nonnull NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(nonnull NSArray<NSNumber *> *)textureOutputPasses;
```

In `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`, update the method signature to match, and pass `appleLog` to the C function call:

```cpp
return dawn_pipeline_setup_multipass(
  _pipeline,
  cShaders.data(), shaderCount,
  width, height,
  flatSpecs.data(), bufferCount,
  useCanvas, sync, (bool)appleLog,
  resourceSpecs.data(), (int)resourceSpecs.size(),
  passInputSpecs.data(), (int)passInputSpecs.size(),
  texOutPasses.data(), (int)texOutPasses.size()
);
```

- [ ] **Step 7: Add `appleLog` parameter to `DawnComputePipeline::setup()`**

In `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h`:

Update the C++ setup signature (line 50):

```cpp
bool setup(const std::vector<std::string>& wgslShaders,
           int width, int height,
           const std::vector<BufferSpec>& bufferSpecs,
           bool useCanvas, bool sync,
           const std::vector<ResourceSpec>& resources = {},
           const std::vector<PassInputSpec>& passInputs = {},
           const std::vector<int>& textureOutputPasses = {},
           bool appleLog = false);
```

Also update the C extern function signature (line 132):

```c
bool dawn_pipeline_setup_multipass(
  DawnComputePipelineRef ref,
  const char** shaders, int shaderCount,
  int width, int height,
  const int* bufferSpecs, int bufferCount,
  bool useCanvas, bool sync, bool appleLog,
  const void* resources, int resourceCount,
  const void* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount);
```

In `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`:

Update the C++ `setup()` definition (line 156) to match. Update the C function `dawn_pipeline_setup_multipass` implementation (line 934) to accept `bool appleLog` and forward it to `pipeline->setup(...)`.

Store in `Impl`:

```cpp
// In Impl struct, add:
bool appleLog = false;

// In setup(), after existing init:
_impl->appleLog = appleLog;
```

- [ ] **Step 8: Commit**

```bash
git add packages/react-native-webgpu-camera/src/types.ts \
  packages/react-native-webgpu-camera/src/useCamera.ts \
  packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts \
  packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: thread appleLog flag from JS config through Swift bridge to Dawn pipeline"
```

---

## Task 3: Switch AVCaptureVideoDataOutput Pixel Format for Apple Log

When `appleLog` is true, request 10-bit YUV instead of BGRA from the camera. Also set the correct color space.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift:196-293`

- [ ] **Step 1: Store `appleLog` state on the module**

Add a property to `WebGPUCameraModule`:

```swift
var isAppleLog = false
```

In the `setupMultiPassPipeline` function, set it:

```swift
self.isAppleLog = appleLog
```

- [ ] **Step 2: Conditional pixel format in `startCapture`**

In `startCapture()`, replace the hardcoded BGRA video settings (around line 271):

```swift
let output = AVCaptureVideoDataOutput()
if self.isAppleLog {
  output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
  ]
} else {
  output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
  ]
}
output.alwaysDiscardsLateVideoFrames = true
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: request YUV 10-bit pixel format for Apple Log capture"
```

---

## Task 4: RGBA16Float Ping-Pong Textures and Multi-Planar IOSurface Import

When `appleLog` is true: (a) create ping-pong textures as `RGBA16Float`, (b) import the camera IOSurface as `R10X6BG10X6Biplanar420Unorm` and create per-plane texture views, (c) update the `MakeImageFromTexture` call to use the correct format.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm:180-195,489-516,675-679`

- [ ] **Step 1: Conditional ping-pong texture format in `setup()`**

In `setup()` (around line 182), make the format conditional:

```cpp
texDesc.format = _impl->appleLog
  ? wgpu::TextureFormat::RGBA16Float
  : wgpu::TextureFormat::RGBA8Unorm;
```

Also update the passthrough shader to use the correct storage format. The existing `kPassthroughWGSL` (line 204) hardcodes `rgba8unorm`. Add a second passthrough for Apple Log:

```cpp
static const std::string kPassthroughWGSL_16F = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  textureStore(outputTex, vec2i(id.xy), textureLoad(inputTex, vec2i(id.xy), 0));
}
)";
```

Use the appropriate passthrough when auto-inserting:

```cpp
const auto& passthrough = _impl->appleLog ? kPassthroughWGSL_16F : kPassthroughWGSL;
```

- [ ] **Step 2: Multi-planar IOSurface import in `processFrame()`**

In `processFrame()`, replace the single-plane import block (around line 489-516) with a conditional:

```cpp
IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);
if (!ioSurface) return false;

wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
ioDesc.ioSurface = ioSurface;

wgpu::SharedTextureMemoryDescriptor sharedDesc{};
sharedDesc.nextInChain = &ioDesc;

auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
if (!sharedMemory) return false;

wgpu::Texture inputTexture;
wgpu::TextureView yPlaneView;   // only used in appleLog mode
wgpu::TextureView uvPlaneView;  // only used in appleLog mode

if (impl->appleLog) {
  // Import as multi-planar 10-bit YUV
  wgpu::TextureDescriptor inputTexDesc{};
  inputTexDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
  inputTexDesc.format = wgpu::TextureFormat::R10X6BG10X6Biplanar420Unorm;
  inputTexDesc.usage = wgpu::TextureUsage::TextureBinding;
  inputTexDesc.dimension = wgpu::TextureDimension::e2D;
  inputTexDesc.mipLevelCount = 1;
  inputTexDesc.sampleCount = 1;
  inputTexDesc.label = "CameraInputYUV";

  inputTexture = sharedMemory.CreateTexture(&inputTexDesc);
  if (!inputTexture) return false;

  // Create per-plane views
  wgpu::TextureViewDescriptor yViewDesc{};
  yViewDesc.aspect = wgpu::TextureAspect::Plane0Only;
  yViewDesc.format = wgpu::TextureFormat::R16Unorm;
  yPlaneView = inputTexture.CreateView(&yViewDesc);

  wgpu::TextureViewDescriptor uvViewDesc{};
  uvViewDesc.aspect = wgpu::TextureAspect::Plane1Only;
  uvViewDesc.format = wgpu::TextureFormat::RG16Unorm;
  uvPlaneView = inputTexture.CreateView(&uvViewDesc);
} else {
  // Existing BGRA path
  wgpu::TextureDescriptor inputTexDesc{};
  inputTexDesc.size = {(uint32_t)_width, (uint32_t)_height, 1};
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
sharedMemory.BeginAccess(inputTexture, &beginDesc);
```

- [ ] **Step 3: Update bind group creation for pass 0 in Apple Log mode**

In `processFrame()`, the bind group for pass 0 currently binds the camera input texture view at binding 0 and the output ping-pong texture at binding 1 (the standard 2-entry block). In Apple Log mode, pass 0 is the auto-inserted YUV→RGB shader which has a **different layout** (3 bindings). The Apple Log block must **replace** the standard block (use `if/else`), not be added alongside it:

```cpp
if (impl->appleLog && passIdx == 0) {
  // YUV→RGB pass REPLACES the standard 2-entry bind group for pass 0.
  // Layout: Y plane + UV plane + output texture (3 bindings, not 2)
  std::vector<wgpu::BindGroupEntry> entries(3);
  entries[0].binding = 0;
  entries[0].textureView = yPlaneView;
  entries[1].binding = 1;
  entries[1].textureView = uvPlaneView;
  entries[2].binding = 2;
  entries[2].textureView = outTex.CreateView();

  wgpu::BindGroupDescriptor bgDesc{};
  bgDesc.layout = pass.bindGroupLayout;
  bgDesc.entryCount = entries.size();
  bgDesc.entries = entries.data();
  pass0BindGroup = device.CreateBindGroup(&bgDesc);
} else {
  // Standard SDR pass 0: camera input + output (existing 2-entry code)
  // ... existing bind group creation for pass 0 ...
}
```

- [ ] **Step 4: Update MakeImageFromTexture format**

In `processFrame()`, update the `MakeImageFromTexture` call (around line 675) to use the correct format:

```cpp
auto texFormat = impl->appleLog
  ? wgpu::TextureFormat::RGBA16Float
  : wgpu::TextureFormat::RGBA8Unorm;
auto outputImage = ctx.MakeImageFromTexture(
  *(finalIsA ? &impl->texA : &impl->texB), _width, _height, texFormat);
```

- [ ] **Step 5: Also update texture output format for Apple Log**

In `setup()`, where pass texture outputs are created (around line 379), make format conditional:

```cpp
texOutDesc.format = _impl->appleLog
  ? wgpu::TextureFormat::RGBA16Float
  : wgpu::TextureFormat::RGBA8Unorm;
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: RGBA16Float ping-pong textures and multi-planar YUV IOSurface import"
```

---

## Task 5: Built-in YUV→RGB Conversion Shader

Write the WGSL compute shader that converts Apple Log YUV (video range, BT.2020 primaries) to Apple Log RGB. This shader is auto-inserted as pass 0 when `appleLog` is true.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

- [ ] **Step 1: Add the YUV→RGB shader string constant**

Add near the top of the file (after the existing `kPassthroughWGSL`):

```cpp
static const std::string kYUVtoRGBWGSL = R"(
@group(0) @binding(0) var yPlaneTex: texture_2d<f32>;
@group(0) @binding(1) var uvPlaneTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

// BT.2020 video range YCbCr → RGB
// Input: R16Unorm Y plane (full res), RG16Unorm UV plane (half res)
// Video range 10-bit: Y [64..940]/1023, CbCr [64..960]/1023

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(yPlaneTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let coord = vec2i(id.xy);
  let uvCoord = vec2i(id.xy / 2u);

  // Load normalized [0,1] values from R16Unorm/RG16Unorm planes
  let yRaw = textureLoad(yPlaneTex, coord, 0).r;
  let uvRaw = textureLoad(uvPlaneTex, uvCoord, 0).rg;

  // Video range expansion (10-bit: 64/1023 = 0.06256, 940/1023 = 0.91887, 960/1023 = 0.93842)
  let y = (yRaw - 0.06256) / (0.91887 - 0.06256);
  let cb = (uvRaw.r - 0.06256) / (0.93842 - 0.06256) - 0.5;
  let cr = (uvRaw.g - 0.06256) / (0.93842 - 0.06256) - 0.5;

  // BT.2020 non-constant-luminance YCbCr → RGB
  let r = y + 1.4746 * cr;
  let g = y - 0.16455 * cb - 0.57135 * cr;
  let b = y + 1.8814 * cb;

  // Do NOT clamp — Apple Log values outside [0,1] represent valid HDR data
  textureStore(outputTex, coord, vec4f(r, g, b, 1.0));
}
)";
```

- [ ] **Step 2: Auto-insert YUV→RGB as pass 0 when appleLog is true**

In `setup()`, before the user shader compilation loop, prepend the YUV→RGB shader when `appleLog` is true:

```cpp
std::vector<std::string> effectiveShaders;
if (appleLog) {
  effectiveShaders.push_back(kYUVtoRGBWGSL);
}
// ... existing code that builds effectiveShaders from wgslShaders or kPassthroughWGSL ...
```

The existing code (around line 216) builds `effectiveShaders`. The YUV→RGB shader must come first. The YUV→RGB pass has a different bind group layout (3 entries vs 2), so when creating the bind group layout for pass 0 in Apple Log mode, use `GetBindGroupLayout(0)` from the compiled pipeline — Dawn auto-derives it from the shader.

- [ ] **Step 3: Update parity logic**

The existing `finalTex` parity logic (line 285) uses `effectiveShaders.size()`. Since the YUV→RGB pass is now included in `effectiveShaders`, parity is automatically correct — no additional changes needed. Verify this is the case.

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: built-in YUV→RGB compute shader for Apple Log (BT.2020 video range)"
```

---

## Task 6: RGBA32Float Resource Upload for .cube LUT Textures

The existing resource upload path in `DawnComputePipeline.mm` hardcodes `RGBA8Unorm` for all textures. LUT data is `Float32Array` (4 floats × 4 bytes = 16 bytes per texel). Add format support so `RGBA32Float` textures can be uploaded.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h:16-22`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm:300-368`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`
- Modify: `packages/react-native-webgpu-camera/src/GPUResource.ts`
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`

- [ ] **Step 1: Add `format` field to `ResourceSpec`**

In `DawnComputePipeline.h`, add a format enum value and field to `ResourceSpec`:

```cpp
enum class ResourceFormat { RGBA8Unorm, RGBA32Float };

struct ResourceSpec {
  ResourceType type;
  ResourceFormat format = ResourceFormat::RGBA8Unorm;
  std::vector<uint8_t> data;
  int width = 0;
  int height = 0;
  int depth = 0;
};
```

- [ ] **Step 2: Use `format` field in texture resource upload**

In `DawnComputePipeline.mm`, in the resource upload loop (around line 313), make the texture format and bytes-per-row conditional:

```cpp
wgpu::TextureFormat texFmt;
int bytesPerPixel;
if (spec.format == ResourceFormat::RGBA32Float) {
  texFmt = wgpu::TextureFormat::RGBA32Float;
  bytesPerPixel = 16;  // 4 floats × 4 bytes
} else {
  texFmt = wgpu::TextureFormat::RGBA8Unorm;
  bytesPerPixel = 4;
}

texResDesc.format = texFmt;
// ... existing creation ...

wgpu::TexelCopyBufferLayout dataLayout{};
dataLayout.bytesPerRow = spec.width * bytesPerPixel;
dataLayout.rowsPerImage = spec.height;
```

- [ ] **Step 3: Add optional `format` to the TypeScript resource API**

In `packages/react-native-webgpu-camera/src/GPUResource.ts`, update the `texture3D` dims parameter:

```typescript
function texture3D(
  data: ArrayBuffer,
  dims: { width: number; height: number; depth: number; format?: 'rgba8unorm' | 'rgba32float' },
): ResourceHandle<'texture3d'> {
  return {
    __resourceType: 'texture3d',
    __handle: -1,
    __data: data,
    __dims: dims,
  };
}
```

Update `ResourceHandle` interface to include format in dims:

```typescript
export interface ResourceHandle<T extends string> {
  readonly __resourceType: T;
  readonly __handle: number;
  readonly __data?: ArrayBuffer;
  readonly __dims?: { width: number; height: number; depth?: number; format?: string };
}
```

- [ ] **Step 4: Forward `format` through the capture proxy and native config**

In `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`, update the `CapturedResource` interface:

```typescript
interface CapturedResource {
  type: 'texture3d' | 'texture2d' | 'storageBuffer';
  data: ArrayBuffer;
  width?: number;
  height?: number;
  depth?: number;
  format?: string;
}
```

In `capturePipeline()`, when building `capturedResources`, include format:

```typescript
capturedResources.push({
  type: rh.__resourceType as CapturedResource['type'],
  data: rh.__data!,
  width: rh.__dims?.width,
  height: rh.__dims?.height,
  depth: rh.__dims?.depth,
  format: rh.__dims?.format,
});
```

In `buildNativeConfig()`, include format in the resources array:

```typescript
const resources = capturedResources.map((r) => ({
  type: r.type,
  data: r.data,
  width: r.width ?? 0,
  height: r.height ?? 0,
  depth: r.depth ?? 0,
  format: r.format ?? 'rgba8unorm',
}));
```

In `WebGPUCameraModule.ts`, add `format` to the resources interface:

```typescript
resources: {
  type: string;
  data: ArrayBuffer;
  width: number;
  height: number;
  depth: number;
  format: string;
}[];
```

- [ ] **Step 5: Read `format` in the ObjC bridge**

In `DawnPipelineBridge.mm`, in the resource conversion loop, read the format string:

```objc
NSString *format = res[@"format"];
if ([format isEqualToString:@"rgba32float"]) {
  rs.format = dawn_pipeline::ResourceFormat::RGBA32Float;
} else {
  rs.format = dawn_pipeline::ResourceFormat::RGBA8Unorm;
}
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm \
  packages/react-native-webgpu-camera/src/GPUResource.ts \
  packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts \
  packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts
git commit -m "feat: RGBA32Float resource upload for .cube LUT 3D textures"
```

---

## Task 7: .cube File Parser

Parse standard .cube 3D LUT files into Float32Array suitable for upload as a 3D texture.

**Files:**
- Create: `packages/react-native-webgpu-camera/src/parseCubeFile.ts`
- Modify: `packages/react-native-webgpu-camera/src/index.ts`

- [ ] **Step 1: Create the parser**

Create `packages/react-native-webgpu-camera/src/parseCubeFile.ts`:

```typescript
/**
 * Parse a .cube 3D LUT file into a Float32Array suitable for GPU upload.
 *
 * .cube format:
 *   LUT_3D_SIZE N
 *   R G B    (N³ lines, each with 3 floats in [0,1])
 *
 * Returns RGBA (A=1.0) data for use with GPUResource.texture3D({ format: 'rgba32float' }).
 */
export function parseCubeFile(text: string): { data: Float32Array; size: number } {
  const lines = text.split('\n');
  let size = 0;
  const values: number[] = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;

    if (line.startsWith('LUT_3D_SIZE')) {
      size = parseInt(line.split(/\s+/)[1], 10);
      continue;
    }

    // Skip other metadata lines (TITLE, DOMAIN_MIN, DOMAIN_MAX, LUT_1D_SIZE, etc.)
    if (line.match(/^[A-Z_]/)) continue;

    const parts = line.split(/\s+/);
    if (parts.length >= 3) {
      values.push(
        parseFloat(parts[0]),
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        1.0,
      );
    }
  }

  if (size === 0) {
    throw new Error('parseCubeFile: missing LUT_3D_SIZE header');
  }

  const expected = size * size * size * 4;
  if (values.length !== expected) {
    throw new Error(
      `parseCubeFile: expected ${expected / 4} entries for size ${size}, got ${values.length / 4}`,
    );
  }

  return { data: new Float32Array(values), size };
}
```

- [ ] **Step 2: Export from index.ts**

In `packages/react-native-webgpu-camera/src/index.ts`, add:

```typescript
export { parseCubeFile } from './parseCubeFile';
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/src/parseCubeFile.ts \
  packages/react-native-webgpu-camera/src/index.ts
git commit -m "feat: .cube 3D LUT file parser"
```

---

## Task 8: Example LUT Shader

Write a WGSL compute shader that applies a 3D LUT to Apple Log RGB input. The shader samples the LUT texture using the input RGB values as coordinates.

**Files:**
- Create: `apps/example/src/shaders/lut.wgsl.ts`

- [ ] **Step 1: Create the LUT shader**

Create `apps/example/src/shaders/lut.wgsl.ts`:

```typescript
export const LUT_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
// Custom inputs: LUT 3D texture at binding 3, sampler at binding 4
@group(0) @binding(3) var lutTex: texture_3d<f32>;
@group(0) @binding(4) var lutSampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let color = textureLoad(inputTex, vec2i(id.xy), 0);

  // Use Apple Log RGB values as 3D texture coordinates
  // Clamp to [0,1] to stay within LUT bounds
  let lutCoord = clamp(color.rgb, vec3f(0.0), vec3f(1.0));
  // textureSampleLevel (not textureSample) — textureSample is unavailable in compute shaders
  let lutColor = textureSampleLevel(lutTex, lutSampler, lutCoord, 0.0);

  textureStore(outputTex, vec2i(id.xy), vec4f(lutColor.rgb, 1.0));
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add apps/example/src/shaders/lut.wgsl.ts
git commit -m "feat: example LUT application shader (WGSL)"
```

---

## Task 9: Example App — Apple Log Format Selection + .cube File Picker

Add UI to the example app: (a) show Apple Log formats in the format picker, (b) allow picking a .cube file from the device, (c) wire up the LUT pipeline.

**Files:**
- Modify: `apps/example/src/app/index.tsx`

- [ ] **Step 1: Add .cube file picker and Apple Log camera preview**

Add state for LUT and a file picker button. When an Apple Log format is selected and a LUT is loaded, use the LUT shader pipeline. Key additions:

```typescript
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import { GPUResource, parseCubeFile } from 'react-native-webgpu-camera';
import { LUT_WGSL } from '@/shaders/lut.wgsl';
```

Add state in `CameraSpikeScreen`:

```typescript
const [lutResource, setLutResource] = useState<ReturnType<typeof GPUResource.texture3D> | null>(null);
const [lutName, setLutName] = useState<string | null>(null);
```

Add a LUT picker function:

```typescript
const pickLut = async () => {
  const result = await DocumentPicker.getDocumentAsync({ type: '*/*' });
  if (result.canceled) return;
  const file = result.assets[0];
  const text = await FileSystem.readAsStringAsync(file.uri);
  const parsed = parseCubeFile(text);
  const resource = GPUResource.texture3D(parsed.data.buffer, {
    width: parsed.size, height: parsed.size, depth: parsed.size,
    format: 'rgba32float',
  });
  setLutResource(resource);
  setLutName(file.name);
};
```

Add a LUT picker button in the controls (visible when not running):

```tsx
{!isRunning && (
  <Pressable style={styles.button} onPress={pickLut}>
    <Text style={styles.buttonText}>{lutName ?? 'Load LUT'}</Text>
  </Pressable>
)}
```

- [ ] **Step 2: Create an Apple Log camera preview component**

Create a new `AppleLogPreview` component that uses `colorSpace: 'appleLog'` and the LUT shader:

```typescript
function AppleLogPreview({ format, lutResource }: { format?: CameraFormat; lutResource: ReturnType<typeof GPUResource.texture3D> | null }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace: 'appleLog',
  });

  const resources = lutResource ? { lut: lutResource } : undefined;

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    resources,
    pipeline: (frame, res) => {
      'worklet';
      if (res?.lut) {
        frame.runShader(LUT_WGSL, { inputs: { lut: res.lut } });
      }
    },
  });

  // ... same Canvas rendering as CameraPreview ...
}
```

- [ ] **Step 3: Add Apple Log mode to shader list and wire it up**

Add a new shader mode type:

```typescript
type ShaderMode =
  | { name: string; wgsl: readonly string[]; type: 'simple' }
  | { name: string; type: 'histogram' }
  | { name: string; type: 'histogram-onframe' }
  | { name: string; type: 'applelog' };
```

Add to `SHADERS` array:

```typescript
{ name: 'Apple Log', type: 'applelog' },
```

Render the Apple Log preview:

```tsx
{isRunning && shader.type === 'applelog' && (
  <AppleLogPreview key={shader.name} format={selectedFormat} lutResource={lutResource} />
)}
```

- [ ] **Step 4: Install expo-document-picker and expo-file-system**

Run:

```bash
cd apps/example && bunx expo install expo-document-picker expo-file-system
```

Note: These require a native rebuild (EAS Build). The file picker won't work in Expo Go.

- [ ] **Step 5: Commit**

```bash
git add apps/example/src/app/index.tsx apps/example/package.json
git commit -m "feat: Apple Log format selection + .cube LUT file picker in example app"
```

---

## Task 10: User-Facing Shader Storage Format Awareness

User shaders currently hardcode `texture_storage_2d<rgba8unorm, write>` for binding 1 (output texture). In Apple Log mode, the output texture is `rgba16float`. User shaders need to use the correct format.

This is a developer experience concern — for the spike, document it and ensure the example shaders work.

**Files:**
- Modify: `apps/example/src/shaders/passthrough.wgsl.ts` (add Apple Log variant or note)
- Modify: `apps/example/src/shaders/sobel.wgsl.ts` (add Apple Log variant or note)

- [ ] **Step 1: Always use `rgba16float` ping-pong textures (both SDR and Apple Log)**

WebGPU requires the shader's `texture_storage_2d<format>` to exactly match the actual texture format — no implicit conversion. Rather than maintaining two shader variants, for the spike **always use `rgba16float`** for ping-pong textures in both modes. This is slightly more memory than `rgba8unorm` in SDR mode but avoids format mismatch complexity entirely.

Go back to Task 4 Step 1 and remove the conditional — always use `RGBA16Float`:

```cpp
texDesc.format = wgpu::TextureFormat::RGBA16Float;
```

Also remove the `kPassthroughWGSL` vs `kPassthroughWGSL_16F` conditional from Task 4 Step 1 — always use the `rgba16float` passthrough.

- [ ] **Step 2: Update all example shaders to use `rgba16float` output**

Update every shader in `apps/example/src/shaders/` that declares `texture_storage_2d<rgba8unorm, write>` to use `texture_storage_2d<rgba16float, write>` instead. This includes:
- `passthrough.wgsl.ts`
- `sobel.wgsl.ts`
- `histogram.wgsl.ts` (if it has a write pass)
- Any other shaders with `rgba8unorm` storage textures

Also update the built-in `kPassthroughWGSL` in `DawnComputePipeline.mm` to use `rgba16float`.

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/shaders/ \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: always use rgba16float ping-pong textures for uniform shader format"
```
