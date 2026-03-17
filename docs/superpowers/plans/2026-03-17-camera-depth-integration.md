# Camera Depth Data Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time LiDAR/camera depth as an optional per-frame dynamic resource in the GPU frame processor, imported zero-copy via IOSurface and bound into compute shaders.

**Architecture:** `GPUResource.cameraDepth()` declares a dynamic resource. The capture proxy detects it and sets `useDepth: true` in the native config. The Swift module adds `AVCaptureDepthDataOutput` + `AVCaptureDataOutputSynchronizer` to deliver paired video+depth frames. The C++ pipeline imports the depth `CVPixelBuffer` IOSurface as an `R16Float` texture and binds it at the auto-assigned index alongside a linear sampler.

**Tech Stack:** Swift (AVFoundation depth APIs), ObjC++ (Dawn pipeline bridge), C++ (Dawn WebGPU texture import), TypeScript (resource API + capture proxy), WGSL (depth colormap shader)

**Spec:** `docs/superpowers/specs/2026-03-17-camera-depth-integration-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/react-native-webgpu-camera/src/GPUResource.ts` | Modify | Add `cameraDepth()` constructor |
| `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` | Modify | Handle `cameraDepth` in capture proxy, set `useDepth` in native config |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Modify | Add depth output, synchronizer, `useDepth` config, pass depth to bridge |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h` | Modify | Add `processFrame:depthBuffer:` method |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm` | Modify | Forward depth buffer, log depth format |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` | Modify | Update `processFrame` signature, add `_useDepth` flag |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` | Modify | Import depth IOSurface, bind to shaders as `R16Float` + sampler |
| `apps/example/src/shaders/depth-colormap.wgsl.ts` | Create | Depth colormap visualization shader |
| `apps/example/src/app/index.tsx` | Modify | Add Depth Colormap mode to shader picker |

---

## Chunk 1: TypeScript — GPUResource.cameraDepth() + capture proxy

### Task 1: Add `cameraDepth()` to GPUResource

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/GPUResource.ts`

- [ ] **Step 1: Add `cameraDepth` constructor**

Add after the `storageBuffer` function (before the `GPUResource` export):

```typescript
function cameraDepth(): ResourceHandle<'cameraDepth'> {
  return {
    __resourceType: 'cameraDepth',
    __handle: -1,
  };
}
```

- [ ] **Step 2: Export in GPUResource object**

Add `cameraDepth` to the `GPUResource` export object:

```typescript
export const GPUResource = {
  texture3D,
  texture2D: Object.assign(texture2DResource, texture2DToken) as { ... },
  storageBuffer,
  cameraDepth,
};
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/src/GPUResource.ts
git commit -m "feat: add GPUResource.cameraDepth() constructor for dynamic depth data"
```

---

### Task 2: Handle `cameraDepth` in capture proxy and native config

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`

The `cameraDepth` resource is different from static resources — it has no data to upload and its texture is provided per-frame by the native side. The capture proxy needs to:
1. Recognize `cameraDepth` handles and skip data upload
2. Still assign binding indices (texture + sampler = 2 slots) when used in `inputs`
3. Set `useDepth: true` in the native config
4. Send a `cameraDepth` entry in the resources array so the native side knows the resource index

- [ ] **Step 1: Update `capturePipeline` to handle `cameraDepth` resources**

In `capturePipeline`, in the resource handle loop (around line 82), `cameraDepth` handles should be added to `capturedResources` with type `'cameraDepth'` and no data. They also need to be added to `handleToIndex` so they can be referenced in `inputs`.

The existing code at line 82-98 already does this correctly — it pushes `{ type: rh.__resourceType, ... }` and the type will be `'cameraDepth'`. The `data`, `fileUri`, `width`, `height`, `depth` will all be undefined/0 which is fine. No changes needed in `capturePipeline`.

- [ ] **Step 2: Update `buildNativeConfig` to set `useDepth` flag**

In `buildNativeConfig` (line 244), add `useDepth` detection and include it in the returned config.

After the existing `return` statement (line 325), change to:

```typescript
  // Detect if any resource is a dynamic cameraDepth
  const useDepth = capturedResources.some((r) => r.type === 'cameraDepth');

  return { shaders, width, height, buffers, useCanvas, sync, appleLog, useDepth, resources, passInputs, textureOutputPasses };
```

- [ ] **Step 3: Update the `CapturedResource` type**

The `CapturedResource` interface (around line 40) needs `'cameraDepth'` in its type union:

```typescript
interface CapturedResource {
  type: 'texture3d' | 'texture2d' | 'storageBuffer' | 'cameraDepth';
  data?: ArrayBuffer;
  fileUri?: string;
  width?: number;
  height?: number;
  depth?: number;
  format?: string;
}
```

- [ ] **Step 4: Handle `cameraDepth` in input binding resolution**

In the `runShader` capture (around line 122), when a `cameraDepth` handle appears in `options.inputs`, it should be treated as a `texture2d` + `sampler` pair (same as static `texture2d` resources). The existing code at lines 127-134 already handles `texture2d` resource types correctly:

```typescript
if (rh.__resourceType === 'texture3d' || rh.__resourceType === 'texture2d') {
```

Add `'cameraDepth'` to this condition:

```typescript
if (rh.__resourceType === 'texture3d' || rh.__resourceType === 'texture2d' || rh.__resourceType === 'cameraDepth') {
```

And similarly for the `CapturedInput.type` — when pushing the input, use `'texture2d'` as the binding type for `cameraDepth` (since that's how the native side will bind it):

After the existing texture type push, add a mapping for `cameraDepth`:

```typescript
if (rh.__resourceType === 'texture3d' || rh.__resourceType === 'texture2d' || rh.__resourceType === 'cameraDepth') {
  pass.inputs.push({
    name,
    bindingIndex: nextBinding,
    type: rh.__resourceType === 'texture3d' ? 'texture3d' : 'texture2d',
    resourceHandle: resIndex,
  });
```

This maps `cameraDepth` → `texture2d` binding type since both use `texture_2d<f32>` + sampler in the shader.

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: handle cameraDepth in capture proxy, add useDepth to native config"
```

---

## Chunk 2: Native — Swift session changes

### Task 3: Add depth output and synchronizer to WebGPUCameraModule

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

This is the most complex native change. When `useDepth` is true:
1. `setupMultiPassPipeline` stores the `useDepth` flag
2. `startCapture` adds `AVCaptureDepthDataOutput` to the session
3. Replace `FrameDelegate` with a `SynchronizedFrameDelegate` using `AVCaptureDataOutputSynchronizer`
4. The synchronized delegate extracts both video and depth `CVPixelBuffer`s and passes both to `processFrame`

When `useDepth` is false, the existing `FrameDelegate` pattern continues unchanged.

- [ ] **Step 1: Add `useDepth` property and config parsing**

Add property to the module class (after `isYUV422`):

```swift
var useDepth = false
```

In `setupMultiPassPipeline` (around line 148), parse the new config field:

```swift
let useDepth = config["useDepth"] as? Bool ?? false
self.useDepth = useDepth
```

- [ ] **Step 2: Add `AVCaptureDepthDataOutput` and `AVCaptureDataOutputSynchronizer` properties**

Add to the module class properties (after `frameDelegate`):

```swift
private var depthOutput: AVCaptureDepthDataOutput?
private var synchronizer: AVCaptureDataOutputSynchronizer?
private var syncDelegate: SynchronizedFrameDelegate?
```

- [ ] **Step 3: Add depth output to capture session in `startCapture`**

After the existing video output setup (after `session.addOutput(output)`, around line 314), add depth output when `useDepth` is true:

```swift
// Depth output (optional — only when useDepth is true)
if self.useDepth {
  let depthOut = AVCaptureDepthDataOutput()
  depthOut.isFilteringEnabled = true
  depthOut.alwaysDiscardsLateDepthData = true

  if session.canAddOutput(depthOut) {
    session.addOutput(depthOut)
    self.depthOutput = depthOut

    // Use synchronizer for paired video + depth delivery
    let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [output, depthOut])
    let syncDel = SynchronizedFrameDelegate(
      videoOutput: output,
      depthOutput: depthOut,
      module: self
    )
    sync.setDelegate(syncDel, queue: self.frameQueue)
    self.synchronizer = sync
    self.syncDelegate = syncDel

    // Don't set the individual video delegate — synchronizer handles it
    NSLog("[WebGPUCamera] startCapture: depth output added, using synchronizer")
  } else {
    NSLog("[WebGPUCamera] startCapture: WARNING — could not add depth output")
    self.useDepth = false
  }
} else {
  // No depth — use existing individual frame delegate
  // (delegate already set above)
}
```

Important: When using the synchronizer, do NOT set `output.setSampleBufferDelegate` — the synchronizer takes over. Move the existing delegate setup into an `if !self.useDepth` block:

The existing code (around line 307-309):
```swift
let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
self.frameDelegate = delegate
output.setSampleBufferDelegate(delegate, queue: self.frameQueue)
```

Wrap in `if !self.useDepth`:
```swift
if !self.useDepth {
  let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
  self.frameDelegate = delegate
  output.setSampleBufferDelegate(delegate, queue: self.frameQueue)
}
```

- [ ] **Step 4: Create `SynchronizedFrameDelegate` class**

Add after the existing `FrameDelegate` class at the bottom of the file:

```swift
private class SynchronizedFrameDelegate: NSObject, AVCaptureDataOutputSynchronizerDelegate {
  let videoOutput: AVCaptureVideoDataOutput
  let depthOutput: AVCaptureDepthDataOutput
  weak var module: WebGPUCameraModule?
  private var frameCount: Int = 0

  init(videoOutput: AVCaptureVideoDataOutput,
       depthOutput: AVCaptureDepthDataOutput,
       module: WebGPUCameraModule) {
    self.videoOutput = videoOutput
    self.depthOutput = depthOutput
    self.module = module
  }

  func dataOutputSynchronizer(
    _ synchronizer: AVCaptureDataOutputSynchronizer,
    didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection
  ) {
    // Extract synchronized video frame
    guard let syncedVideoData = synchronizedDataCollection.synchronizedData(
      for: videoOutput
    ) as? AVCaptureSynchronizedSampleBufferData,
    !syncedVideoData.sampleBufferWasDropped,
    let pixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer) else {
      return
    }

    // Extract synchronized depth (may be nil if dropped)
    var depthBuffer: CVPixelBuffer? = nil
    if let syncedDepthData = synchronizedDataCollection.synchronizedData(
      for: depthOutput
    ) as? AVCaptureSynchronizedDepthData,
    !syncedDepthData.depthDataWasDropped {
      depthBuffer = syncedDepthData.depthData.depthDataMap
    }

    frameCount += 1
    if frameCount <= 3 {
      let depthFmt = depthBuffer.map { CVPixelBufferGetPixelFormatType($0) } ?? 0
      let depthW = depthBuffer.map { CVPixelBufferGetWidth($0) } ?? 0
      let depthH = depthBuffer.map { CVPixelBufferGetHeight($0) } ?? 0
      NSLog("[SyncDelegate] frame #%d, depth=%@, depthFmt=0x%08x, depthSize=%dx%d",
            frameCount,
            depthBuffer != nil ? "YES" : "NO",
            depthFmt, depthW, depthH)
    }

    // Run Dawn compute pipeline with both buffers
    module?.dawnBridge?.processFrame(pixelBuffer, depthBuffer: depthBuffer)
  }
}
```

- [ ] **Step 5: Clean up depth resources in `stopCapture`**

In `stopCapture` (around line 428), add cleanup:

```swift
self.synchronizer = nil
self.syncDelegate = nil
self.depthOutput = nil
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: add AVCaptureDepthDataOutput with synchronizer for paired video+depth frames"
```

---

## Chunk 3: Native — Bridge and C++ pipeline

### Task 4: Update DawnPipelineBridge for depth buffer

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`

- [ ] **Step 1: Add `processFrame:depthBuffer:` to bridge header**

Add new method alongside existing `processFrame:`:

```objc
- (BOOL)processFrame:(nonnull CVPixelBufferRef)pixelBuffer
         depthBuffer:(nullable CVPixelBufferRef)depthBuffer;
```

- [ ] **Step 2: Implement in bridge .mm**

Add new method that forwards to C++:

```objc
- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer
         depthBuffer:(CVPixelBufferRef)depthBuffer {
  if (!_pipeline) return NO;
  _frameCount++;
  if (_frameCount <= 3 || _frameCount % 300 == 0) {
    OSType fmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    size_t w = CVPixelBufferGetWidth(pixelBuffer);
    size_t h = CVPixelBufferGetHeight(pixelBuffer);
    size_t planes = CVPixelBufferGetPlaneCount(pixelBuffer);
    NSLog(@"[DawnBridge] frame #%d, pixel format: 0x%08x (%c%c%c%c), %zux%zu, %zu planes, depth=%s",
          _frameCount,
          (unsigned)fmt,
          (char)((fmt >> 24) & 0xFF), (char)((fmt >> 16) & 0xFF),
          (char)((fmt >> 8) & 0xFF), (char)(fmt & 0xFF),
          w, h, planes,
          depthBuffer ? "YES" : "NO");
  }
  return dawn_pipeline_process_frame_with_depth(_pipeline, pixelBuffer, depthBuffer);
}
```

Note: the `_frameCount` static was previously in `processFrame:` — move it to an instance variable. Also, keep the existing `processFrame:` method calling with `depthBuffer:nil`:

```objc
- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer {
  return [self processFrame:pixelBuffer depthBuffer:nil];
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git commit -m "feat: bridge processFrame with optional depth buffer"
```

---

### Task 5: Update DawnComputePipeline for depth texture import

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

This is the core GPU integration. The depth buffer is imported as an IOSurface → `R16Float` texture every frame, and stored on the `Impl` struct for bind group creation.

- [ ] **Step 1: Update header — `processFrame` signature and `setup` parameter**

In `DawnComputePipeline.h`:

Add `useDepth` parameter to `setup`:
```cpp
bool setup(const std::vector<std::string>& wgslShaders,
           int width, int height,
           const std::vector<BufferSpec>& bufferSpecs,
           bool useCanvas, bool sync,
           const std::vector<ResourceSpec>& resources = {},
           const std::vector<PassInputSpec>& passInputs = {},
           const std::vector<int>& textureOutputPasses = {},
           bool appleLog = false,
           bool useDepth = false);
```

Add new `processFrame` overload:
```cpp
bool processFrame(CVPixelBufferRef pixelBuffer,
                  CVPixelBufferRef depthBuffer);
```

Add C bridge function:
```c
bool dawn_pipeline_process_frame_with_depth(DawnComputePipelineRef ref,
                                             CVPixelBufferRef pixelBuffer,
                                             CVPixelBufferRef depthBuffer);
```

- [ ] **Step 2: Store `useDepth` flag and depth texture state in Impl**

In `DawnComputePipeline.mm`, add to the `Impl` struct:

```cpp
bool useDepth = false;
int depthResourceIndex = -1;  // index into uploadedResources for the depth slot
wgpu::Texture depthTexture;   // per-frame, re-imported from IOSurface
wgpu::TextureView depthView;
wgpu::Sampler depthSampler;
```

- [ ] **Step 3: Initialize depth state in `setup`**

In `setup()`, after storing `_impl->appleLog = appleLog`:

```cpp
_impl->useDepth = useDepth;

// Find the cameraDepth resource index (if any) and create a reusable sampler
if (useDepth) {
  for (size_t ri = 0; ri < resources.size(); ri++) {
    if (resources[ri].type == ResourceType::CameraDepth) {
      _impl->depthResourceIndex = (int)ri;
      break;
    }
  }

  // Create a linear sampler for bilinear depth upsampling
  wgpu::SamplerDescriptor sampDesc{};
  sampDesc.magFilter = wgpu::FilterMode::Linear;
  sampDesc.minFilter = wgpu::FilterMode::Linear;
  sampDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
  sampDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
  _impl->depthSampler = _impl->device.CreateSampler(&sampDesc);

  NSLog(@"[DawnPipeline] Depth enabled, resourceIndex=%d\n", _impl->depthResourceIndex);
}
```

- [ ] **Step 4: Add `CameraDepth` to ResourceType enum**

In `DawnComputePipeline.h`:

```cpp
enum class ResourceType { Texture3D, Texture2D, StorageBuffer, CameraDepth };
```

- [ ] **Step 5: Handle `CameraDepth` resource type in bridge config parsing**

In `DawnPipelineBridge.mm`, in the resource type parsing loop (around line 60):

```objc
if ([type isEqualToString:@"texture3d"]) {
  rs.type = dawn_pipeline::ResourceType::Texture3D;
} else if ([type isEqualToString:@"texture2d"]) {
  rs.type = dawn_pipeline::ResourceType::Texture2D;
} else if ([type isEqualToString:@"cameraDepth"]) {
  rs.type = dawn_pipeline::ResourceType::CameraDepth;
} else {
  rs.type = dawn_pipeline::ResourceType::StorageBuffer;
}
```

- [ ] **Step 6: Skip GPU upload for CameraDepth resources in setup**

In `DawnComputePipeline.mm`, in the resource upload loop (around line 442 `if (spec.type == ResourceType::Texture3D || spec.type == ResourceType::Texture2D)`), add a guard to skip `CameraDepth`:

```cpp
// Skip CameraDepth — texture is provided per-frame, not uploaded at setup
if (spec.type == ResourceType::CameraDepth) {
  continue;
}
```

- [ ] **Step 7: Import depth IOSurface in `processFrame`**

Add the new `processFrame` overload. After the existing video IOSurface import and before the compute passes:

```cpp
bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer,
                                        CVPixelBufferRef depthBuffer) {
  // Import depth buffer if present and depth is enabled
  if (depthBuffer && _impl && _impl->useDepth) {
    IOSurfaceRef depthSurface = CVPixelBufferGetIOSurface(depthBuffer);
    if (depthSurface) {
      size_t depthW = CVPixelBufferGetWidth(depthBuffer);
      size_t depthH = CVPixelBufferGetHeight(depthBuffer);

      wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
      ioDesc.ioSurface = depthSurface;
      wgpu::SharedTextureMemoryDescriptor sharedDesc{};
      sharedDesc.nextInChain = &ioDesc;

      auto sharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
      if (sharedMemory) {
        wgpu::TextureDescriptor texDesc{};
        texDesc.size = {(uint32_t)depthW, (uint32_t)depthH, 1};
        texDesc.format = wgpu::TextureFormat::R16Float;
        texDesc.usage = wgpu::TextureUsage::TextureBinding;
        texDesc.label = "CameraDepth";

        _impl->depthTexture = sharedMemory.CreateTexture(&texDesc);
        if (_impl->depthTexture) {
          wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
          beginDesc.initialized = true;
          sharedMemory.BeginAccess(_impl->depthTexture, &beginDesc);
          _impl->depthView = _impl->depthTexture.CreateView();
        }

        static bool loggedDepth = false;
        if (!loggedDepth) {
          NSLog(@"[DawnPipeline] Depth texture imported: %zux%zu R16Float\n", depthW, depthH);
          loggedDepth = true;
        }
      }
    }
  }

  // Call the existing processFrame for video
  return processFrame(pixelBuffer);
}
```

Wait — this won't work because the existing `processFrame` doesn't know about the depth texture for bind groups. Instead, integrate depth import into the existing `processFrame` flow. The cleanest approach:

Store the depth buffer on the Impl, import it early in the existing `processFrame`, and use it during bind group creation.

Actually, the simplest approach: add `depthBuffer` as a member of `Impl` that's set before calling the existing `processFrame`:

```cpp
bool DawnComputePipeline::processFrame(CVPixelBufferRef pixelBuffer,
                                        CVPixelBufferRef depthBuffer) {
  if (_impl) {
    _impl->currentDepthBuffer = depthBuffer;  // store for use in processFrame
  }
  return processFrame(pixelBuffer);
}
```

Then in the existing `processFrame`, after IOSurface import and before compute passes, import the depth:

```cpp
// Import depth IOSurface if available
wgpu::SharedTextureMemory depthSharedMemory;
if (impl->useDepth && impl->currentDepthBuffer) {
  IOSurfaceRef depthSurface = CVPixelBufferGetIOSurface(impl->currentDepthBuffer);
  if (depthSurface) {
    size_t depthW = CVPixelBufferGetWidth(impl->currentDepthBuffer);
    size_t depthH = CVPixelBufferGetHeight(impl->currentDepthBuffer);

    wgpu::SharedTextureMemoryIOSurfaceDescriptor ioDesc{};
    ioDesc.ioSurface = depthSurface;
    wgpu::SharedTextureMemoryDescriptor sharedDesc{};
    sharedDesc.nextInChain = &ioDesc;

    depthSharedMemory = device.ImportSharedTextureMemory(&sharedDesc);
    if (depthSharedMemory) {
      wgpu::TextureDescriptor texDesc{};
      texDesc.size = {(uint32_t)depthW, (uint32_t)depthH, 1};
      texDesc.format = wgpu::TextureFormat::R16Float;
      texDesc.usage = wgpu::TextureUsage::TextureBinding;
      texDesc.label = "CameraDepth";

      impl->depthTexture = depthSharedMemory.CreateTexture(&texDesc);
      if (impl->depthTexture) {
        wgpu::SharedTextureMemoryBeginAccessDescriptor beginDesc{};
        beginDesc.initialized = true;
        depthSharedMemory.BeginAccess(impl->depthTexture, &beginDesc);
        impl->depthView = impl->depthTexture.CreateView();

        static bool loggedDepth = false;
        if (!loggedDepth) {
          NSLog(@"[DawnPipeline] Depth texture imported: %zux%zu R16Float\n", depthW, depthH);
          loggedDepth = true;
        }
      }
    }
  }
}
```

And after EndAccess for the video texture, also end access for depth:

```cpp
if (depthSharedMemory && impl->depthTexture) {
  wgpu::SharedTextureMemoryEndAccessState depthEndState{};
  depthSharedMemory.EndAccess(impl->depthTexture, &depthEndState);
  impl->depthTexture = nullptr;
  impl->depthView = nullptr;
}
```

- [ ] **Step 8: Bind depth texture in `appendCustomInputEntries`**

The existing `appendCustomInputEntries` method appends custom bindings to the bind group. For `CameraDepth` resources, it should bind the per-frame depth texture view and sampler instead of a pre-uploaded texture.

Find the `appendCustomInputEntries` method and add handling for the `depthResourceIndex`:

When the binding references a resource whose index matches `depthResourceIndex`, bind `impl->depthView` and `impl->depthSampler` instead of the uploaded resource's texture/sampler.

In the existing loop that creates entries for custom inputs, add a check:

```cpp
bool isDynamic = (ib.resourceHandle == depthResourceIndex && depthResourceIndex >= 0);
if (isDynamic) {
  if (ib.type == InputBindingType::Texture2D) {
    entry.textureView = depthView;
  } else if (ib.type == InputBindingType::Sampler) {
    entry.sampler = depthSampler;
  }
} else {
  // existing static resource binding logic
}
```

- [ ] **Step 9: Add C bridge function**

In `DawnComputePipeline.mm` (or a separate bridge file), add the C function:

```cpp
bool dawn_pipeline_process_frame_with_depth(DawnComputePipelineRef ref,
                                             CVPixelBufferRef pixelBuffer,
                                             CVPixelBufferRef depthBuffer) {
  if (!ref) return false;
  auto* pipeline = static_cast<dawn_pipeline::DawnComputePipeline*>(ref);
  return pipeline->processFrame(pixelBuffer, depthBuffer);
}
```

- [ ] **Step 10: Pass `useDepth` through bridge `setupMultiPass`**

Update `DawnPipelineBridge.mm` to forward the new parameter. In `setupMultiPassWithShaders:`, parse `useDepth` from config (if passed as a separate parameter or from the existing config dict).

The simplest approach: add `useDepth` as a parameter to `setupMultiPassWithShaders:`:

Header:
```objc
- (BOOL)setupMultiPassWithShaders:(nonnull NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(nonnull NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                         appleLog:(BOOL)appleLog
                         useDepth:(BOOL)useDepth
                        resources:(nonnull NSArray<NSDictionary *> *)resources
                       passInputs:(nonnull NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(nonnull NSArray<NSNumber *> *)textureOutputPasses;
```

Forward to C++:
```cpp
return dawn_pipeline_setup_multipass(
    _pipeline,
    cShaders.data(), shaderCount,
    width, height,
    flatSpecs.data(), bufferCount,
    useCanvas, sync, (bool)appleLog, (bool)useDepth,
    resourceSpecs.data(), (int)resourceSpecs.size(),
    passInputSpecs.data(), (int)passInputSpecs.size(),
    texOutPasses.data(), (int)texOutPasses.size()
);
```

Update the Swift caller in `WebGPUCameraModule.swift` to pass `useDepth`:
```swift
let ok = bridge.setupMultiPass(
  withShaders: shaders,
  width: Int32(width),
  height: Int32(height),
  bufferSpecs: bufferSpecs,
  useCanvas: useCanvas,
  sync: sync,
  appleLog: appleLog,
  useDepth: self.useDepth,
  resources: resourcesRaw,
  passInputs: passInputsRaw,
  textureOutputPasses: textureOutputPasses
)
```

- [ ] **Step 11: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git commit -m "feat: import depth IOSurface as R16Float texture, bind to compute shaders"
```

---

## Chunk 4: Example shader and app integration

### Task 6: Create depth colormap shader

**Files:**
- Create: `apps/example/src/shaders/depth-colormap.wgsl.ts`

- [ ] **Step 1: Write shader**

```typescript
// Depth colormap visualization shader.
// Blends camera image with a colormap of the depth data.
// Near (0m) = blue, mid (2.5m) = green, far (5m) = yellow.
// Depth is sampled with bilinear interpolation from 320×240 → video resolution.

export const DEPTH_COLORMAP_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) { return; }

  let color = textureLoad(inputTex, vec2i(id.xy), 0);

  // Sample depth with bilinear upsampling (320x240 → video resolution)
  let uv = (vec2f(id.xy) + 0.5) / vec2f(outDims);
  let depth = textureSampleLevel(depthTex, depthSampler, uv, 0.0).r;

  // Colormap: 0m (near) = blue → 2.5m = green → 5m+ (far) = yellow
  let t = clamp(depth / 5.0, 0.0, 1.0);
  let r = smoothstep(0.25, 0.75, t);
  let g = 1.0 - abs(t - 0.5) * 2.0;
  let b = 1.0 - smoothstep(0.0, 0.5, t);
  let depthColor = vec3f(r, g, b);

  // Blend: 60% camera + 40% depth colormap
  let blended = mix(color.rgb, depthColor, 0.4);
  textureStore(outputTex, vec2i(id.xy), vec4f(blended, 1.0));
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add apps/example/src/shaders/depth-colormap.wgsl.ts
git commit -m "feat: depth colormap visualization shader"
```

---

### Task 7: Add Depth Colormap mode to example app

**Files:**
- Modify: `apps/example/src/app/index.tsx`

- [ ] **Step 1: Import shader and GPUResource.cameraDepth**

Add to imports at the top:

```typescript
import { DEPTH_COLORMAP_WGSL } from '@/shaders/depth-colormap.wgsl';
```

`GPUResource` is already imported.

- [ ] **Step 2: Add 'Depth' to SHADERS array**

Add a new shader mode type and entry:

In the `ShaderMode` type, add `| { name: string; type: 'depth' }`.

In the `SHADERS` array, add:

```typescript
{ name: 'Depth', type: 'depth' },
```

- [ ] **Step 3: Create DepthPreview component**

```typescript
function DepthPreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, {
    resources: {
      depth: GPUResource.cameraDepth(),
    },
    pipeline: (frame, { depth }) => {
      'worklet';
      frame.runShader(DEPTH_COLORMAP_WGSL, { inputs: { depth } });
    },
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Depth ${camera.width}x${camera.height}` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}
```

- [ ] **Step 4: Wire into render switch**

In `CameraSpikeScreen`, add the depth preview alongside the other modes:

```typescript
{isRunning && shader.type === 'depth' && <DepthPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
```

- [ ] **Step 5: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: add Depth Colormap mode to example app"
```

---

## Chunk 5: Verify

### Task 8: Build and test on device

- [ ] **Step 1: Verify TypeScript compiles**

```bash
bunx tsc
```

Expected: No errors.

- [ ] **Step 2: Native rebuild**

This requires a native rebuild since we modified Swift, ObjC++, and C++ files.

```bash
eas build --platform ios --profile development --local
```

Or via EAS cloud build.

- [ ] **Step 3: Test on device**

1. Launch app, select "Depth" shader mode
2. Check Console.app for:
   - `[SyncDelegate] frame #1, depth=YES, depthFmt=...` — confirms depth frames arriving
   - `[DawnPipeline] Depth texture imported: 320x240 R16Float` — confirms IOSurface import
   - `[DawnPipeline] Depth enabled, resourceIndex=0` — confirms setup
3. Verify colormap visualization on screen — objects closer should be blue, farther should be yellow
4. Switch to non-depth shaders (Sobel, Passthrough) — verify they still work without depth overhead
5. Check fps — depth shouldn't significantly impact performance (320x240 texture is tiny)

- [ ] **Step 4: Final commit if adjustments needed**

```bash
git add -A
git commit -m "fix: adjustments from depth integration device testing"
```
