# useGPUFrameProcessor Design

## Goal

Expose the validated zero-copy GPU compute pipeline as a composable React hook API. Users pass WGSL shader strings and Skia drawing commands to process live camera frames on the GPU, with results rendered via Skia Canvas — all without pixels ever leaving the GPU or touching the JS thread.

## Architecture Context

The Phase 0 spike validated the full pipeline on iPhone 16 Pro: 4K@120fps, 2.88ms GPU, zero frame drops, nominal thermal. The architecture:

```
Camera (AVCaptureSession)
  → CVPixelBuffer (IOSurface — already on GPU)
  → Dawn SharedTextureMemory (zero-copy import)
  → Compute Shader (WGSL)
  → SkImage (texture-backed via Skia Graphite)
  → useFrameCallback worklet (UI thread)
  → SharedValue<SkImage> → Skia Canvas
```

No pixel copies anywhere. JS thread not involved in the render loop. The hooks are a thin React layer on top of this proven native pipeline.

---

## Public API

### useCamera

Owns camera lifecycle. Returns an opaque handle consumed by `useGPUFrameProcessor`.

```typescript
interface CameraConfig {
  device: 'back' | 'front';
  width: number;
  height: number;
  fps: number;
}

interface CameraHandle {
  /** True once camera is producing frames */
  isReady: boolean;
  /** Increments each time a new frame arrives from the native camera */
  frameCounter: number;
}

function useCamera(config: CameraConfig): CameraHandle;
```

**Lifecycle:**
- Calls `WebGPUCameraModule.startCameraPreview()` on mount
- Calls `WebGPUCameraModule.stopCameraPreview()` on unmount
- Restarts if config changes (device/width/height/fps)

**Not in scope:** permissions, device enumeration, camera controls (zoom, focus, torch).

### useGPUFrameProcessor

Processes camera frames with GPU compute shaders and Skia drawing. Returns a shared value with the latest processed frame.

```typescript
type FrameProcessor = (frame: ProcessorFrame) => void;

interface ProcessorFrame {
  /** Run a WGSL compute shader on the current frame */
  runShader(wgslCode: string): void;

  /** Skia canvas targeting the output texture — draws are recorded */
  canvas: SkCanvas;

  /** Current frame dimensions */
  width: number;
  height: number;
}

interface GPUFrameProcessorResult {
  /** Latest processed frame as SkImage — drive a Skia Canvas with this */
  currentFrame: SharedValue<SkImage | null>;
}

function useGPUFrameProcessor(
  camera: CameraHandle,
  processor: FrameProcessor,
): GPUFrameProcessorResult;
```

### Usage

```typescript
const camera = useCamera({
  device: 'back',
  width: 3840,
  height: 2160,
  fps: 120,
});

const { currentFrame } = useGPUFrameProcessor(camera, (frame) => {
  'worklet';
  // GPU compute
  frame.runShader(SOBEL_WGSL);

  // Skia drawing into the recorded output
  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  frame.canvas.drawRect(Skia.XYWHRect(bbox.x, bbox.y, bbox.w, bbox.h), paint);
});

// User owns the Canvas — overlays are just JSX (NOT recorded)
<Canvas style={StyleSheet.absoluteFill}>
  <Image image={currentFrame} fit="cover" />
  <Text x={20} y={40} text={`${fps} FPS`} color="white" />
</Canvas>
```

### Two-layer rendering model

1. **Effect layer** (inside processor callback): GPU compute + Skia canvas draws. This is the canonical frame — both screen and recorder consume it.
2. **Overlay layer** (user's JSX Canvas): screen-only drawing on top. Not recorded.

```
Camera → [Processor: compute + Skia draws] → Canonical Frame
                                                  ├→ Recorder (future)
                                                  └→ [User Canvas: overlays] → Screen
```

---

## Threading Model

- **Camera thread (native):** captures frame → runs compute shaders via `processFrame()`
- **UI thread (worklet):** `useFrameCallback` grabs latest result via `stream.nextImage()` → runs user's Skia draws → updates `currentFrame` shared value
- **JS thread:** not involved in the hot path

### Shader compilation

`runShader()` in the worklet does not compile the shader every frame. Shaders are compiled once at setup time via `setupComputePipeline()`. The worklet call dispatches a pre-compiled pipeline. If the user passes a different WGSL string, it triggers recompilation (with a dev mode warning).

---

## Hook internals

### useCamera

1. `useEffect` calls `startCameraPreview(device, width, height, fps)` on mount
2. Returns cleanup that calls `stopCameraPreview()`
3. Tracks `isReady` via `frameCounter > 0`
4. Restarts capture session when config changes

### useGPUFrameProcessor

1. When `camera.isReady`: calls `setupComputePipeline(wgsl, width, height)` to compile shader and create GPU resources
2. Creates stream host object via `__webgpuCamera_createStream()` — a JSI host object shared across Reanimated runtimes
3. Stores stream in a `useSharedValue<CameraStream | null>`
4. `useFrameCallback` worklet runs every display frame:
   - Calls `stream.nextImage()` to get the latest compute output as `SkImage`
   - Calls user's processor callback with `ProcessorFrame` (the SkImage + Skia canvas)
   - Disposes previous frame, writes new frame to `currentFrame` shared value
5. On unmount: nulls the stream, calls `cleanupComputePipeline()`

### Native foundation (existing, proven)

- `DawnComputePipeline` (C++): shader compilation, IOSurface import, compute dispatch, SkImage output
- `CameraStreamHostObject` (C++ JSI): host object with `nextImage()` that Reanimated shares across runtimes
- `DawnPipelineBridge` (ObjC): bridges C++ to Swift
- `WebGPUCameraModule` (Swift Expo module): camera lifecycle, compute setup, JSI installation

---

## Future: Frame History

Persistent textures across frames for temporal effects (motion blur, noise reduction, optical flow).

```typescript
const { currentFrame } = useGPUFrameProcessor(camera, (frame) => {
  'worklet';
  const prevFrame = frame.useTexture('prev');

  frame.runShader(MOTION_BLUR_WGSL, {
    previous: prevFrame,
  });

  frame.copyToTexture(frame.output, prevFrame);
});
```

- `frame.useTexture(name)`: named persistent `wgpu::Texture`, same dimensions as frame. Created on first call, reused across frames. Destroyed on unmount.
- `frame.runShader(wgsl, bindings)`: bindings object maps names to textures. `@group(0) @binding(0)` is always the current input. Additional bindings mapped by declaration order.
- `frame.copyToTexture(src, dst)`: GPU-side copy, no CPU involvement.

Not implemented in v1. The `runShader` signature already accepts an optional bindings parameter so the API is forward-compatible.

---

## Future: Native Pipeline Path

For performance-critical use cases where shader chains are static. Compute runs on the native camera thread with zero JS involvement. The worklet handles only dynamic per-frame logic.

```typescript
const { currentFrame } = useGPUFrameProcessor(camera, {
  // Native — defined once, runs every frame on camera thread
  pipeline: (ctx) => {
    const prevFrame = ctx.useTexture('prev');
    ctx.runShader(DENOISE_WGSL, { previous: prevFrame });
    ctx.runShader(COLOR_CORRECT_WGSL);
    ctx.copyToTexture(ctx.output, prevFrame);
  },

  // Worklet — runs on UI thread after native pipeline completes
  onFrame: (frame) => {
    'worklet';
    for (const box of boundingBoxes.value) {
      frame.canvas.drawRect(box, paint);
    }
  },
});
```

**Execution order per frame:**
1. Camera thread: capture → native pipeline (all shaders) → intermediate texture
2. UI thread worklet: receives intermediate texture → Skia draws → `currentFrame` shared value

**Shorthand forms:**
- Worklet only (v1): `useGPUFrameProcessor(camera, callback)`
- Native + worklet: `useGPUFrameProcessor(camera, { pipeline, onFrame })`
- Native only: `useGPUFrameProcessor(camera, { pipeline })`

Not implemented in v1. The worklet-only path ships first. v1 internals are structured to support this addition without breaking changes.

---

## v1 Scope

### In scope
- `useCamera` hook — camera lifecycle management
- `useGPUFrameProcessor` with worklet callback — `frame.runShader(wgsl)` + `frame.canvas`
- `currentFrame` as `SharedValue<SkImage | null>`
- Single shader per `runShader()` call (chaining via multiple calls)
- iOS only

### Documented, not implemented in v1
- Frame history (`frame.useTexture`, `frame.copyToTexture`)
- Native pipeline path (`{ pipeline, onFrame }`)
- `runShader` bindings parameter
- Android support

### Not in scope
- Recording API (separate future hook)
- Camera controls (zoom, focus, exposure, torch)
- Device enumeration
- Permissions handling

## File Locations

- `packages/react-native-webgpu-camera/src/useCamera.ts`
- `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`
- `packages/react-native-webgpu-camera/src/types.ts` — shared types
- `packages/react-native-webgpu-camera/src/index.ts` — re-exports hooks and types
