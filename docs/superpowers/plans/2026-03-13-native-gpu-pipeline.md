# Native GPU Pipeline — Implementation Plan

## Architecture

```
camera CVPixelBuffer → IOSurface → Dawn SharedTextureMemory → input GPUTexture
    → compute shader (WGSL string, passed from JS at setup) → output GPUTexture
    → MakeImageFromTexture → SkImage (handle, not pixels)
    → Skia Canvas: drawImage(output) + user overlays (boxes, text, paths)
    → Canvas backing texture → encoder / display
```

Everything GPU-resident. No pixel data crosses JS bridge.

## Key discoveries

### Skia Graphite + Dawn shared context
- Skia Graphite bundles Dawn (WebGPU impl) and provides `navigator.gpu` via JSI
- `DawnContext::getInstance()` is a singleton — one Dawn device shared with Skia's Graphite renderer
- `getWGPUDevice()` returns the same `wgpu::Device` that Skia uses for rendering
- This means GPU textures created via Dawn are directly usable by Skia — no copies needed

### Canvas backing texture is a swapchain texture
- `DawnWindowContext::getSurface()` calls `_surface.GetCurrentTexture()` each frame
- Returns a different `wgpu::Texture` each frame (swapchain rotation)
- Skia wraps it via `SkSurfaces::WrapBackendTexture()` for drawing
- **Cannot write compute output directly to Canvas backing texture** — it rotates, and Skia redraws fully each frame
- Instead: compute writes to a persistent output texture, Skia `drawImage`s it (free — same GPU, texture bind only)

### Dawn JS bindings limitations (rnwgpu layer)
- `writeTexture()`: Implemented in C++ but returns zeros from JS (unclear why — possibly ArrayBuffer bridging issue)
- `copyBufferToTexture()`: **Stubbed out** in `GPUQueue.cpp` (commented/unimplemented)
- `copyBufferToBuffer()`: Works correctly
- `ImportSharedTextureMemory()`: **Not exposed to JS** — C++ only
- `MakeImageFromTexture()`: Works from JS — returns valid texture-backed SkImage

### IOSurface zero-copy import (reference impl in RNDawnContext.h:105-162)
```cpp
// Already exists in Skia's codebase:
wgpu::SharedTextureMemoryIOSurfaceDescriptor platformDesc;
platformDesc.ioSurface = CVPixelBufferGetIOSurface(pixelBuffer);

wgpu::SharedTextureMemoryDescriptor desc = {};
desc.nextInChain = &platformDesc;
wgpu::SharedTextureMemory memory = device.ImportSharedTextureMemory(&desc);

wgpu::Texture texture = memory.CreateTexture(&textureDesc);
memory.BeginAccess(texture, &beginAccessDesc);
// texture is now usable — zero copy from camera
```

### Camera frame format
- iOS camera: BGRA8Unorm (32bpp)
- Dawn preferred format: also BGRA8Unorm on Apple (`DawnUtils::PreferredTextureFormat`)
- No format conversion needed

## Implementation steps

### 1. C++ helper: DawnComputePipeline
New file: `modules/webgpu-camera/ios/DawnComputePipeline.h/cpp`

- Accesses `DawnContext::getInstance()` for shared Dawn device
- `setup(wgslCode, width, height)` — compile shader, create pipeline, create output texture
- `importCameraFrame(CVPixelBufferRef)` — IOSurface → SharedTextureMemory → input texture
- `dispatch()` — run compute, write to output texture
- `getOutputImage()` → `sk_sp<SkImage>` via `MakeImageFromTexture`

### 2. Wire into WebGPUCameraModule
- `setupComputePipeline(wgslCode, width, height)` — JS-callable, passes string to C++
- `getComputeOutputImage()` → returns SkImage to JS (via Skia's JSI host object)
- FrameDelegate calls C++ `importCameraFrame` + `dispatch` per frame (native thread)

### 3. JS side (simplified)
```tsx
// Setup
WebGPUCameraModule.setupComputePipeline(SOBEL_WGSL, 1920, 1080);
WebGPUCameraModule.startCameraPreview('back', 1920, 1080);

// Per frame — just get the output image handle
const outputImage = WebGPUCameraModule.getComputeOutputImage();

// Canvas
<Canvas>
  <Image image={outputImage} fit="cover" />
  <Rect x={box.x} y={box.y} ... color="red" />
</Canvas>
```

## Files to create/modify
- **NEW**: `modules/webgpu-camera/ios/DawnComputePipeline.h` — C++ Dawn interop
- **MOD**: `modules/webgpu-camera/ios/WebGPUCameraModule.swift` — new JS-callable functions, call C++ from frame delegate
- **MOD**: `apps/example/src/app/index.tsx` — simplified, no pixel bridge
- **DEL**: `apps/example/src/hooks/useGPUPipeline.ts` — no longer needed (pipeline is native)
