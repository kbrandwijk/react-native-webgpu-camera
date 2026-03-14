# Multi-Pass Compute + frame.canvas Design

## Goal

Extend `useGPUFrameProcessor` with multi-pass shader chaining, real Skia canvas drawing on processed frames, and GPU buffer readback — enabling use cases like running edge detection followed by object detection, then drawing bounding boxes on the result.

## Architecture Context

v1 validated the zero-copy pipeline at 4K@120fps with a single compute shader. The processor callback runs once at setup to capture a WGSL string; the native camera thread runs compute every frame; a `useFrameCallback` worklet grabs results via `stream.nextImage()`.

This design extends that foundation:
- Multiple `runShader()` calls → multi-pass compute with ping-pong textures
- `frame.canvas` → real SkSurface backed by GPU texture (replacing the no-op Proxy stub)
- Buffer readback → GPU storage buffer data returned as typed arrays for use in Skia draws

---

## Public API

### Shorthand — compute only (backwards compatible with v1)

```typescript
useGPUFrameProcessor(camera, (frame) => {
  'worklet';
  frame.runShader(SOBEL_WGSL);
  frame.runShader(SHARPEN_WGSL); // chained — reads from Sobel's output
});
```

### Object form — pipeline + per-frame draws and readback

```typescript
useGPUFrameProcessor(camera, {
  sync: false, // default: onFrame gets latest available buffer data
  pipeline: (frame) => {
    frame.runShader(SOBEL_WGSL);
    const detections = frame.runShader(DETECT_WGSL, {
      output: Float32Array, count: 256,
    });
    return { detections };
  },
  onFrame: (frame, { detections }) => {
    'worklet';
    if (detections) {
      const paint = Skia.Paint();
      paint.setColor(Skia.Color('red'));
      paint.setStyle(PaintStyle.Stroke);
      paint.setStrokeWidth(3);
      for (let i = 0; i < detections.length; i += 4) {
        frame.canvas.drawRect(
          Skia.XYWHRect(detections[i], detections[i+1], detections[i+2], detections[i+3]),
          paint,
        );
      }
    }
  },
});
```

### `runShader` overloads

```typescript
interface ProcessorFrame {
  /** Run a compute shader — output feeds into next pass or becomes final frame */
  runShader(wgsl: string): void;

  /** Run a compute shader with buffer output — returns a handle resolved per-frame */
  runShader<T extends TypedArrayConstructor>(
    wgsl: string,
    options: { output: T; count?: number },
  ): BufferHandle<InstanceType<T>>;

  /** Skia canvas targeting the current pass's output texture */
  canvas: SkCanvas;

  /** Current frame dimensions */
  width: number;
  height: number;
}

type TypedArrayConstructor = new (buffer: ArrayBuffer) => ArrayBufferView;
```

### `BufferHandle`

`BufferHandle<T>` is structurally identical to `T`. At setup time (inside `pipeline`), it's an opaque placeholder. In `onFrame`, the runtime resolves it to live data each frame.

```typescript
type BufferHandle<T> = T;
```

The type transformation from handle to nullable data happens via the `NullableBuffers` mapped type on `onFrame`'s second parameter.

### Full type definitions

```typescript
type NullableBuffers<B> = { [K in keyof B]: B[K] | null };

interface ProcessorConfig<B extends Record<string, any>> {
  /** When true, onFrame blocks until current frame's compute + readback completes.
   *  Default false: onFrame receives most recent available data (may be 1 frame behind). */
  sync?: boolean;

  /** Runs once at setup. Declares shader chain and buffer outputs.
   *  Return value maps buffer names to handles for use in onFrame. */
  pipeline: (frame: ProcessorFrame) => B;

  /** Runs every display frame on UI thread.
   *  Receives resolved buffer data and a canvas for Skia draws. */
  onFrame?: (
    frame: { canvas: SkCanvas; width: number; height: number },
    buffers: NullableBuffers<B>,
  ) => void;
}

// Overloads
function useGPUFrameProcessor(
  camera: CameraHandle,
  processor: (frame: ProcessorFrame) => void,
): GPUFrameProcessorResult;

function useGPUFrameProcessor<B extends Record<string, any>>(
  camera: CameraHandle,
  config: ProcessorConfig<B>,
): GPUFrameProcessorResult;
```

### `sync` flag behavior

- `sync: false` (default): `onFrame` runs every display frame. Buffer data is the most recently resolved readback — typically 1 frame behind, `null` if no readback has completed yet. Zero stall.
- `sync: true`: `onFrame` waits for the current frame's compute and buffer mapping to complete before running. Guarantees fresh data but introduces a GPU→CPU stall. Use when correctness matters more than latency.

---

## Multi-Pass Compute

### Ping-pong textures

The native side allocates two persistent textures (A and B) at frame dimensions, RGBA8Unorm. Passes alternate between them:

```
Camera IOSurface → input texture (zero-copy)
  Pass 0: read input, write A
  Pass 1: read A,     write B
  Pass 2: read B,     write A
  ...
  Final output = whichever texture the last pass wrote to
```

### Shader bindings

Every compute shader follows the same binding layout:
- `@group(0) @binding(0)`: input texture (`texture_2d<f32>`)
- `@group(0) @binding(1)`: output texture (`texture_storage_2d<rgba8unorm, write>`)
- `@group(0) @binding(2)`: output storage buffer (optional, only when `output` is declared)

This means existing v1 shaders (Sobel, etc.) work unchanged in multi-pass chains — they already use bindings 0 and 1.

### Single command encoder

All passes go into one `wgpu::CommandEncoder`, one `queue.Submit()` call per frame. WebGPU guarantees sequential execution within a command buffer — no explicit barriers needed between compute passes.

### Canvas draws between passes

When the pipeline builder detects canvas usage after a shader pass:

1. After the pass's compute dispatch, wrap the output texture in an SkSurface via `DawnContext::MakeSurfaceFromTexture()`
2. User's Skia draws go to that surface's canvas
3. The next pass reads from that same texture — draws are baked in
4. When no canvas draws happen between passes, the SkSurface step is skipped (zero overhead)

---

## Buffer Readback

### Double-buffered staging

For each declared output buffer:
- Two staging buffers (S0, S1) with `MapRead | CopyDst` usage
- Per frame: compute writes to the output buffer, then a copy command transfers output → current staging buffer
- After submit, async map begins on the current staging buffer
- `onFrame` reads the *previously* mapped staging buffer (already resolved)

### Readback flow per frame

```
Frame N:
  1. Compute dispatch writes to output buffer
  2. Copy output buffer → staging[N % 2]
  3. Submit command buffer
  4. Begin async map on staging[N % 2]
  5. onFrame reads staging[(N-1) % 2] → Float32Array (already mapped)
```

With `sync: true`, step 5 instead waits for staging[N % 2] to finish mapping, then reads it.

### Typed array creation

The native `readBuffer` JSI function:
1. Reads the mapped staging buffer pointer
2. Creates a `jsi::ArrayBuffer` backed by a copy of the mapped data
3. Wraps it with the appropriate typed array constructor (determined at setup from the `output` parameter)
4. Returns it to the worklet runtime

The copy is expected to be small (bounding box data, histograms — hundreds of bytes, not megapixels).

---

## `frame.canvas` Implementation

### SkSurface backed by GPU texture

The native side creates a persistent `SkSurface` via Skia Graphite's `DawnContext::MakeSurfaceFromTexture()`, targeting the current output texture. This surface's canvas is exposed as a `JsiSkCanvas` host object accessible from the worklet runtime.

### Canvas in `onFrame`

In the object form, `onFrame` receives a frame object with a `canvas` property. This canvas targets the final compute output texture. Draws go directly onto the GPU texture — no CPU-side copies.

After `onFrame` completes, the texture (with compute + Skia draws composited) is wrapped as an SkImage and written to `currentFrame` shared value.

### Canvas in shorthand form

In the shorthand form, `frame.canvas` is available and targets the output of the last shader pass. This supports simple overlay use cases without needing the object form. However, since the shorthand's callback runs once at setup (not per-frame), canvas draws in the shorthand are static — they're baked in at setup time and repeated every frame.

For dynamic draws (bounding boxes that change each frame), use the object form with `onFrame`.

### When canvas is unused

The native side tracks whether the pipeline configuration uses canvas draws. If not, no SkSurface is allocated and the compute output goes directly to `currentFrame` — identical to v1, zero overhead.

---

## Hook Internals

### Setup flow (runs once when camera is ready)

1. Pipeline callback executes with a capture proxy
2. Proxy collects: ordered list of WGSL strings, buffer declarations (constructor, count), canvas usage flag
3. Native `setupMultiPassPipeline` receives the full pipeline description
4. Native side:
   - Compiles all shaders into `wgpu::ComputePipeline` objects
   - Allocates two ping-pong textures (RGBA8Unorm, frame dimensions)
   - Allocates output buffers + double-buffered staging buffers for each declared output
   - Allocates SkSurface if canvas is used
5. Creates `CameraStreamHostObject` with new methods for buffer access and canvas

### Per-frame flow

```
Camera thread:
  CVPixelBuffer → IOSurface → input texture (zero-copy)
  For each pass:
    Create bind group: input tex, output tex, [output buffer]
    Dispatch compute
    [If canvas draws between passes: SkSurface composite]
  Copy output buffers → staging buffers
  Submit command encoder
  Begin async map on staging buffers
  Wrap final texture as SkImage

UI thread (useFrameCallback):
  stream.nextImage() → SkImage
  stream.readBuffer(index) → ArrayBuffer | null (from mapped staging)
  If onFrame present:
    Resolve buffer handles to typed arrays
    Call onFrame(frame, buffers)
    Canvas draws composited onto SkImage
  Update currentFrame shared value
  Dispose previous frame
```

### `CameraStreamHostObject` changes

New methods (additive):
- `readBuffer(index: number): ArrayBuffer | null` — returns data from the mapped staging buffer at the given index
- `getCanvas(): JsiSkCanvas | null` — returns the SkSurface canvas for Skia drawing
- `flushCanvas()` — finalizes canvas draws and composites them onto the output texture

`nextImage()` unchanged.

### Cleanup on unmount

Same pattern as v1: dispose final frame, null stream, call `cleanupComputePipeline()`. Cleanup now also tears down all compiled pipelines, both ping-pong textures, all staging buffers, and the SkSurface.

### Error handling

Same as v1: `error: string | null` on the result. If any shader in the chain fails to compile, the error identifies which pass failed (e.g. "Shader compilation failed at pass 2").

---

## Native API Changes

### `setupComputePipeline` → `setupMultiPassPipeline`

New native function accepting the full pipeline description:

```typescript
// Called from hook setup
WebGPUCameraModule.setupMultiPassPipeline({
  shaders: string[],           // ordered WGSL strings
  width: number,
  height: number,
  buffers: Array<{             // one per runShader with output
    passIndex: number,         // which pass produces this buffer
    elementSize: number,       // bytes per element (4 for f32/u32, etc.)
    count: number,             // number of elements
  }>,
  useCanvas: boolean,          // whether to allocate SkSurface
  sync: boolean,               // whether to block on buffer readback
});
```

The v1 `setupComputePipeline(wgsl, width, height)` is replaced by this. The shorthand hook form constructs a single-shader config internally.

### `processFrame` changes

`processFrame` now iterates over compiled pipelines, ping-ponging textures, dispatching each pass, and handling buffer copies. The camera thread still calls it per-frame — the interface doesn't change, only the internal implementation.

---

## Scope

### In scope
- Multiple `runShader()` calls with ping-pong textures
- `frame.canvas` backed by real SkSurface (replaces Proxy stub)
- Canvas draws between shader passes
- Buffer readback with typed array return
- `sync` flag for blocking vs best-effort readback
- Object form `{ pipeline, onFrame, sync }` for the hook
- Shorthand form remains backwards compatible
- iOS only

### Not in scope
- Frame history (`useTexture`, `copyToTexture`) — separate future work, API TBD
- Android support
- Recording API
- Camera controls
- `runShader` bindings parameter (for passing additional textures)

## File Locations

### TypeScript (modify)
- `packages/react-native-webgpu-camera/src/types.ts` — updated types
- `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` — multi-pass capture proxy, object form handling, onFrame worklet
- `packages/react-native-webgpu-camera/src/index.ts` — re-exports (if new types added)

### Native (modify)
- `modules/webgpu-camera/ios/DawnComputePipeline.mm` — multi-pipeline, ping-pong textures, staging buffers, SkSurface, updated CameraStreamHostObject
- `modules/webgpu-camera/ios/DawnComputePipeline.h` — updated C interface
- `modules/webgpu-camera/ios/DawnPipelineBridge.mm` — bridge updated methods
- `modules/webgpu-camera/ios/DawnPipelineBridge.h` — bridge updated interface
- `modules/webgpu-camera/ios/WebGPUCameraModule.swift` — call new setup function
