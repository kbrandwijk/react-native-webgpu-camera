# Camera Depth Data Integration — Design Spec

## Summary

Add real-time LiDAR/dual-camera depth data as an optional dynamic resource in the GPU frame processor pipeline. Depth arrives as a per-frame `CVPixelBuffer` (IOSurface-backed), imported zero-copy into WebGPU as a `texture_2d<f32>`, and bound into compute shaders alongside video frames. Enabled only when the user declares `GPUResource.cameraDepth()` in their pipeline resources — zero overhead otherwise.

## API

### Resource declaration

```ts
useGPUFrameProcessor(camera, {
  resources: {
    depth: GPUResource.cameraDepth(),
  },
  pipeline: (frame, { depth }) => {
    'worklet';
    frame.runShader(DEPTH_COLORMAP_WGSL, { inputs: { depth } });
  },
});
```

`GPUResource.cameraDepth()` returns a `ResourceHandle<'cameraDepth'>` — a new dynamic resource type. Unlike static resources (`texture3D`, `texture2D`, `storageBuffer`) which are uploaded once at setup, `cameraDepth` is re-imported from a new IOSurface every frame.

### Opt-in behavior

- No `GPUResource.cameraDepth()` in resources → no `AVCaptureDepthDataOutput` added, no synchronizer, zero overhead
- With `cameraDepth` → native side adds depth output to session, switches to synchronized delivery

### Shader binding

Depth follows the existing auto-assigned binding system. A `cameraDepth` resource occupies two binding slots (texture + sampler), same as `texture2D`:

```wgsl
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;
```

The sampler enables bilinear interpolation, allowing shaders to sample the low-resolution depth (320×240) at video resolution without a separate upscale pass.

## Native Implementation

### Session changes (WebGPUCameraModule.swift)

When `setupMultiPassPipeline` config includes a `cameraDepth` resource:

1. Create `AVCaptureDepthDataOutput` and add to session
2. Enable `isFilteringEnabled = true` on the depth output (fills holes, reduces noise)
3. Replace individual `AVCaptureVideoDataOutputSampleBufferDelegate` with `AVCaptureDataOutputSynchronizer` for paired video + depth delivery
4. Synchronizer delegate receives both `AVCaptureSynchronizedSampleBufferData` (video) and `AVCaptureSynchronizedDepthData` (depth) in one callback
5. Pass both `CVPixelBuffer`s to `processFrame`

When no depth is requested, the current unsynchronized `FrameDelegate` pattern continues unchanged.

### Depth format

iOS delivers depth as `kCVPixelFormatType_DepthFloat16` — 16-bit float, values in meters. This maps to `wgpu::TextureFormat::R16Float` in Dawn.

Typical resolution: 320×240 from LiDAR via AVFoundation (iPhone 12 Pro and later).

### processFrame changes (DawnComputePipeline)

`processFrame` gains an optional depth parameter:

```cpp
bool processFrame(CVPixelBufferRef pixelBuffer,
                  CVPixelBufferRef depthBuffer = nullptr);
```

When `depthBuffer` is non-null:

1. Extract IOSurface: `CVPixelBufferGetIOSurface(depthBuffer)`
2. Import via `SharedTextureMemory` (same path as video)
3. Create texture with format `R16Float`, dimensions from the depth buffer (320×240)
4. `BeginAccess` / `EndAccess` guards (same as video)
5. Create a texture view for binding
6. Auto-pair with a linear sampler (bilinear filtering for upsampling)
7. Store on the `Impl` struct for bind group creation

The depth texture is re-imported every frame — each frame's IOSurface is different. Same lifecycle as the video input texture.

### Bind group integration

The depth texture + sampler are bound at auto-assigned indices (3+) through the existing `passInputs` system. The capture proxy on the JS side handles index assignment — no changes to the binding logic needed.

For pass 0 (which builds bind groups per-frame), the depth entries are appended after the standard entries. For cached passes 1+, depth is included in the cached bind group.

### Bridge changes (DawnPipelineBridge.mm)

Update `processFrame:` to accept an optional depth buffer:

```objc
- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer
         depthBuffer:(CVPixelBufferRef _Nullable)depthBuffer;
```

### Config flow (JS → native)

The `setupMultiPassPipeline` config gets a new field:

```ts
{
  shaders: [...],
  width: 1920, height: 1080,
  // ... existing fields ...
  useDepth: true,  // signals native to set up depth capture
}
```

The `useDepth` flag is derived from the presence of `GPUResource.cameraDepth()` in the resources block during capture proxy execution.

## TypeScript Changes

### GPUResource.cameraDepth()

New constructor in `GPUResource.ts`:

```ts
const GPUResource = {
  // ... existing constructors ...
  cameraDepth(): ResourceHandle<'cameraDepth'> {
    return {
      __resourceType: 'cameraDepth',
      __handle: -1,
      __isDynamic: true,
    };
  },
};
```

### Capture proxy changes (useGPUFrameProcessor.ts)

The capture proxy recognizes `cameraDepth` resources:

- During `capturePipeline`, when a `cameraDepth` handle appears in `inputs`, it's treated as a `texture2D` + `sampler` pair (2 binding slots)
- The `buildNativeConfig` function sets `useDepth: true` when any resource has type `'cameraDepth'`
- No data upload needed (unlike static resources) — the native side provides the texture per-frame

### Resource type handling

In the native config's `resources` array, a `cameraDepth` entry is sent as:

```ts
{ type: 'cameraDepth', width: 0, height: 0, depth: 0 }
```

Width/height are 0 because the actual dimensions come from the depth camera at runtime. The native side ignores these for dynamic resources.

## Example Shader: Depth Colormap

```wgsl
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
  let uv = vec2f(id.xy) / vec2f(outDims);
  let depth = textureSampleLevel(depthTex, depthSampler, uv, 0.0).r;

  // Colormap: 0m (near) = blue → 2.5m = green → 5m (far) = yellow
  let t = clamp(depth / 5.0, 0.0, 1.0);
  let r = t;
  let g = 1.0 - abs(t - 0.5) * 2.0;
  let b = 1.0 - t;
  let depthColor = vec3f(r, g, b);

  // Blend: 60% camera + 40% depth colormap
  let blended = mix(color.rgb, depthColor, 0.4);
  textureStore(outputTex, vec2i(id.xy), vec4f(blended, 1.0));
}
```

Example app integration:

```ts
const DEPTH_COLORMAP_WGSL = `...`;  // shader above

function DepthPreview({ format, colorSpace }) {
  const camera = useCamera({ device: 'back', format, colorSpace });

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
    <Canvas style={StyleSheet.absoluteFill}>
      <Fill color="black" />
      <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
    </Canvas>
  );
}
```

## Scope

### In scope
- `GPUResource.cameraDepth()` resource constructor
- `AVCaptureDepthDataOutput` + `AVCaptureDataOutputSynchronizer` on native side
- Zero-copy IOSurface → Dawn texture import for depth (R16Float)
- Auto-paired linear sampler for bilinear upsampling
- Depth colormap example shader
- Opt-in: zero overhead when not used

### Out of scope
- Normalized depth mode (`{ normalized: true }`) — easy to add later as a parameter
- ARKit scene depth (higher resolution but requires AR session)
- Depth recording
- Multiple simultaneous dynamic resources (only one `cameraDepth` supported)

### Notes
- Front camera TrueDepth works through the same `AVCaptureDepthDataOutput` API — no special handling needed. If the selected device supports depth, it works regardless of front/back.

## Files to create/modify

| File | Action | Change |
|------|--------|--------|
| `packages/react-native-webgpu-camera/src/GPUResource.ts` | Modify | Add `cameraDepth()` constructor |
| `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` | Modify | Handle `cameraDepth` in capture proxy, set `useDepth` flag |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Modify | Add depth output, synchronizer, pass depth to bridge |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h` | Modify | Update processFrame signature |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm` | Modify | Forward depth buffer to C++ |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` | Modify | Update processFrame signature, add depth state |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` | Modify | Import depth IOSurface, bind to shaders |
| `apps/example/src/shaders/depth-colormap.wgsl.ts` | Create | Depth colormap shader |
| `apps/example/src/app/index.tsx` | Modify | Add DepthPreview mode |
