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
  /** True once camera is producing frames. Triggers React re-render when it changes. */
  isReady: boolean;
  /** Camera frame dimensions and config (for useGPUFrameProcessor to read) */
  width: number;
  height: number;
  fps: number;
}

function useCamera(config: CameraConfig): CameraHandle;
```

**Lifecycle:**
- Calls `WebGPUCameraModule.startCameraPreview()` on mount
- Calls `WebGPUCameraModule.stopCameraPreview()` on unmount
- Restarts if config changes (device/width/height/fps)
- `isReady` transitions to `true` after a short delay post-start (camera needs time to produce first frame). Implemented via `setTimeout` after `startCameraPreview()` — matches existing spike behavior. A native `onCameraReady` event would be more precise but is not needed for v1.

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

  // Skia drawing into the recorded output (e.g. bounding boxes from ML)
  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  paint.setStyle(PaintStyle.Stroke);
  paint.setStrokeWidth(3);
  // detections would come from a SharedValue populated by a separate ML pipeline
  for (const box of detections.value) {
    frame.canvas.drawRect(Skia.XYWHRect(box.x, box.y, box.w, box.h), paint);
  }
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

- **Camera thread (native):** captures frame → IOSurface import (zero-copy) → frame available for compute
- **UI thread (worklet):** `useFrameCallback` runs every display frame:
  1. Calls `stream.nextImage()` to get the latest camera frame as SkImage
  2. Calls `frame.runShader()` which dispatches pre-compiled compute via JSI into native
  3. Runs user's Skia canvas draws on the output
  4. Updates `currentFrame` shared value
- **JS thread:** not involved in the hot path

### How `runShader()` works

`runShader()` is called from the worklet but dispatches compute on the GPU via a JSI call into the native `DawnComputePipeline`. The shader is compiled once at setup time via `setupComputePipeline()`. Each `runShader()` call in the worklet submits a pre-compiled compute pass — it does not recompile. The JSI call overhead is microseconds; the GPU work is asynchronous.

If the user passes a different WGSL string than what was compiled at setup, the hook triggers recompilation (with a dev mode warning about the performance cost). Chaining multiple shaders via multiple `runShader()` calls requires multiple compiled pipelines — the hook tracks which shaders have been seen and compiles each once.

### How `frame.canvas` works

The native pipeline produces an SkImage (read-only compute output). To enable Skia drawing on top, the hook maintains a persistent SkSurface backed by a GPU texture at the frame dimensions. Each frame:

1. `stream.nextImage()` returns the compute output as SkImage
2. The hook draws this SkImage onto the persistent SkSurface
3. The user's processor callback receives the SkSurface's canvas — draws go directly onto the GPU texture
4. After the callback, the hook extracts an SkImage snapshot from the SkSurface
5. This composited SkImage (compute + user draws) becomes `currentFrame.value`

When the user's callback does no Skia drawing (compute-only), the hook skips the SkSurface compositing step and passes the compute output directly to `currentFrame` — no overhead.

### Error handling

`GPUFrameProcessorResult` includes an error state:

```typescript
interface GPUFrameProcessorResult {
  currentFrame: SharedValue<SkImage | null>;
  /** Non-null if shader compilation or pipeline setup failed */
  error: string | null;
}
```

Shader compilation errors surface via this field rather than throwing in the worklet.

---

## Hook internals

### useCamera

1. `useEffect` calls `startCameraPreview(device, width, height, fps)` on mount
2. Returns cleanup that calls `stopCameraPreview()`
3. Sets `isReady = true` after a short delay post-start (matching spike's `setTimeout` pattern)
4. Restarts capture session when config changes (compares previous config via ref)

### useGPUFrameProcessor

1. When `camera.isReady`: calls `setupComputePipeline(wgsl, width, height)` to compile shader and create GPU resources
2. Creates stream host object via `__webgpuCamera_createStream()` — a JSI host object shared across Reanimated runtimes
3. Stores stream in a `useSharedValue<CameraStream | null>`
4. `useFrameCallback` worklet runs every display frame:
   - Calls `stream.nextImage()` to get the latest compute output as `SkImage`
   - Calls user's processor callback with `ProcessorFrame` (the SkImage + Skia canvas)
   - Disposes previous frame, writes new frame to `currentFrame` shared value
5. On unmount: nulls the stream, calls `cleanupComputePipeline()`

### SkImage disposal

The hook owns SkImage lifecycle. Each frame, the hook disposes the previous `currentFrame.value` before replacing it with the new one. Consumers should NOT call `dispose()` on the shared value — the hook handles this. When the hook unmounts, it disposes the final frame.

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

**Important distinction:** The `pipeline` callback is a declarative builder, not a runtime function. It runs once at setup time to describe which shaders and textures the native pipeline should use. The native C++ side then executes that description every frame on the camera thread. This is fundamentally different from the worklet `onFrame` callback, which runs JS logic per frame on the UI thread. They share the `runShader` name for API consistency, but `pipeline.runShader` registers a shader in the native graph while `onFrame.runShader` dispatches a pre-compiled compute pass from JS.

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
