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

The shorthand form does not support `frame.canvas` — canvas draws require the object form with `onFrame` since draws must run per-frame. The `canvas` property on `ProcessorFrame` is only valid inside the `pipeline` callback for between-pass draws (see "Canvas draws between passes" below).

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
    options: { output: T; count: number },
  ): BufferHandle<InstanceType<T>>;

  /** Skia canvas targeting the current pass's output texture.
   *  Only valid inside `pipeline` for between-pass draws.
   *  Draws are recorded once at setup time and replayed every frame (static).
   *  For per-frame dynamic draws, use `onFrame`'s `RenderFrame.canvas`. */
  canvas: SkCanvas;

  /** Current frame dimensions */
  width: number;
  height: number;
}

type TypedArrayConstructor =
  | typeof Float32Array | typeof Float64Array
  | typeof Int8Array | typeof Int16Array | typeof Int32Array
  | typeof Uint8Array | typeof Uint16Array | typeof Uint32Array
  | typeof Uint8ClampedArray;
```

### `BufferHandle`

At setup time (inside `pipeline`), `runShader` returns a `BufferHandle` — an opaque placeholder that is resolved to live data in `onFrame`. The branded type prevents accidentally using a handle as real data inside `pipeline`.

```typescript
type BufferHandle<T> = T & { readonly __brand: unique symbol };
```

The type transformation from handle to nullable data happens via the `NullableBuffers` mapped type on `onFrame`'s second parameter, which strips the brand and adds `| null`.

### `RenderFrame` — the `onFrame` frame type

`onFrame` receives a `RenderFrame`, which is distinct from `ProcessorFrame`. `ProcessorFrame` is the setup-time interface with `runShader()`. `RenderFrame` is the per-frame interface with `canvas` only.

```typescript
interface RenderFrame {
  /** Skia canvas targeting the final compute output texture */
  canvas: SkCanvas;
  /** Current frame dimensions */
  width: number;
  height: number;
}
```

### Full type definitions

```typescript
type NullableBuffers<B> = { [K in keyof B]: B[K] extends BufferHandle<infer U> ? U | null : B[K] | null };

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
    frame: RenderFrame,
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

### Input format asymmetry

The camera IOSurface is imported as BGRA8Unorm (native iOS camera format). Ping-pong textures are RGBA8Unorm. Both formats return `vec4<f32>` from `textureLoad`, but the channel order differs: the camera texture has B in `.r` and R in `.b`.

The native side handles this transparently by using a **texture view format override** on the camera input texture. When creating the input texture view for pass 0, it specifies `format: RGBA8Unorm` on the BGRA8Unorm texture. Dawn supports this swizzle via view format compatibility. This means all shaders see RGBA consistently — no shader changes needed for pass 0 vs. subsequent passes.

### Shader bindings

Every compute shader follows the same binding layout:
- `@group(0) @binding(0)`: input texture (`texture_2d<f32>`)
- `@group(0) @binding(1)`: output texture (`texture_storage_2d<rgba8unorm, write>`)
- `@group(0) @binding(2)`: output storage buffer (optional, only when `output` is declared)

This means existing v1 shaders (Sobel, etc.) work unchanged in multi-pass chains — they already use bindings 0 and 1.

### Command submission strategy

Compute passes and Skia Graphite draws use separate submission paths:

- **Compute passes** use a `wgpu::CommandEncoder` with explicit compute pass dispatch
- **Skia Graphite draws** use Skia's `Recorder` → `snap()` → `Context::insertRecording()` → `Context::submit()`

These cannot be interleaved in a single command buffer. When canvas draws occur between compute passes, the submission order is:

```
Compute passes 0..K → submit command encoder
Skia canvas draws   → Recorder::snap() + Context::submit()
Compute passes K+1..N → new command encoder → submit
```

When no canvas draws occur between passes, all compute passes go into a single command encoder with one submit — optimal performance.

### Canvas draws between passes

When the pipeline builder detects canvas usage after a shader pass:

1. Submit the current command encoder (flushes all compute passes so far)
2. Wrap the current output texture as an SkSurface via `SkSurfaces::WrapBackendTexture(recorder, backendTexture, colorType, colorSpace, surfaceProps)` where `backendTexture` is created via `skgpu::graphite::BackendTextures::MakeDawn(texture.Get())`
3. User's Skia draws go to that surface's canvas
4. Snap the recording and submit via Skia Graphite's context
5. The next compute pass reads from that same texture — draws are baked in
6. When no canvas draws happen between passes, the SkSurface step is skipped (zero overhead)

**Performance note:** Canvas draws between passes force multiple GPU submits within a single `processFrame` call (compute submit → Skia submit → compute submit). This is the slow path — Skia Graphite's `Context::submit()` runs while holding the pipeline mutex, which may stall the camera thread. Profile when using this feature. For most use cases, canvas draws in `onFrame` (which run on the UI thread, after compute is done) are the better choice.

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
  5. onFrame reads staging[(N-1) % 2] → typed array (already mapped)
```

With `sync: true`, step 5 instead waits for staging[N % 2] to finish mapping, then reads it.

### Typed array creation

The native `readBuffer` JSI function:
1. Reads the mapped staging buffer pointer
2. Creates a `jsi::ArrayBuffer` backed by a copy of the mapped data
3. Wraps it with the appropriate typed array constructor (determined at setup from the `output` parameter — element size derived from the constructor's `BYTES_PER_ELEMENT`)
4. Returns it to the worklet runtime

The copy is expected to be small (bounding box data, histograms — hundreds of bytes, not megapixels).

---

## `frame.canvas` Implementation

### SkSurface backed by GPU texture

The native side creates an SkSurface via `SkSurfaces::WrapBackendTexture()`, targeting the final compute output texture. The `wgpu::Texture` is wrapped as a `skgpu::graphite::BackendTexture` via `BackendTextures::MakeDawn(texture.Get())`. The SkSurface's canvas is exposed as a `JsiSkCanvas` host object accessible from the worklet runtime.

The SkSurface is persistent (same dimensions as frame, created once at setup). The v1 output texture already has `RenderAttachment` usage, which `SkSurfaces::WrapBackendTexture` requires.

### Canvas in `onFrame`

In the object form, `onFrame` receives a `RenderFrame` with a `canvas` property. This canvas targets the final compute output texture. Draws go directly onto the GPU texture — no CPU-side copies.

The per-frame flow ensures `currentFrame` is set *after* `onFrame` completes, so canvas draws are included in the output:

1. `stream.nextImage()` returns the compute output as SkImage
2. The hook draws this SkImage onto the persistent SkSurface
3. `onFrame` runs — user draws on the SkSurface's canvas
4. After `onFrame` completes, the hook snapshots the SkSurface as a new SkImage
5. This composited SkImage (compute + user draws) becomes `currentFrame.value`

When `onFrame` does no Skia drawing (buffer readback only), the hook skips the SkSurface step and passes the compute output directly to `currentFrame`.

### When canvas is unused

The native side tracks whether the pipeline configuration uses canvas draws (either between passes or in `onFrame`). If not, no SkSurface is allocated and the compute output goes directly to `currentFrame` — identical to v1, zero overhead.

---

## Threading Model

### Thread boundaries

- **Camera thread**: captures frame, runs all compute passes, copies readback buffers, submits GPU work
- **Reanimated UI runtime (worklet on UI thread)**: `useFrameCallback` runs `onFrame`, reads mapped buffers, draws on canvas, updates `currentFrame`

Canvas draws in `onFrame` happen on the **Reanimated UI runtime** (worklet on the UI thread), not the camera thread. Skia JSI host objects (`JsiSkCanvas`, `JsiSkImage`) are accessible from this runtime. This is safe because:

1. Camera thread completes all compute passes and submits the command buffer *before* the output texture is available via `nextImage()`
2. `nextImage()` returns the latest completed compute output — by the time the UI thread calls it, the GPU work is already submitted
3. The UI thread's canvas draws happen on a persistent SkSurface (separate from the ping-pong textures used by compute)
4. The next camera frame's compute dispatch creates a fresh bind group pointing at the ping-pong textures, not the SkSurface

The existing `std::mutex` in `DawnComputePipeline` protects the shared state (output image pointer, buffer pointers) during the handoff between camera and UI threads.

### Canvas between passes (camera thread)

Canvas draws *between* shader passes in the `pipeline` callback are different — these are static draws captured at setup time. The native side records them as an `SkPicture` at setup and replays the picture each frame on the camera thread between compute submits. This happens within the camera thread's processing, so no cross-thread synchronization is needed.

---

## Hook Internals

### Setup flow (runs once when camera is ready)

1. Pipeline callback executes with a capture proxy
2. Proxy collects: ordered list of WGSL strings, buffer declarations (constructor, count, pass index), canvas usage flag
3. Buffer names are mapped to integer indices in declaration order (e.g., first `runShader` with output → index 0, second → index 1). The hook tracks this mapping to resolve named handles in `onFrame`.
4. Native `setupMultiPassPipeline` receives the full pipeline description
5. Native side:
   - Compiles all shaders into `wgpu::ComputePipeline` objects
   - Allocates two ping-pong textures (RGBA8Unorm, frame dimensions)
   - Allocates output buffers + double-buffered staging buffers for each declared output
   - Allocates SkSurface if canvas is used
6. Creates `CameraStreamHostObject` with new methods for buffer access and canvas

### Dimension changes

When camera dimensions change (e.g., config update), the hook's `useEffect` dependencies (`camera.width`, `camera.height`) trigger cleanup and re-setup — same as v1. The cleanup tears down all pipelines, textures, buffers, and SkSurface. Re-setup allocates everything at the new dimensions.

### Per-frame flow

```
Camera thread:
  CVPixelBuffer → IOSurface → input texture (zero-copy, view format override BGRA→RGBA)
  For each pass:
    Create bind group: input tex, output tex, [output buffer]
    Begin compute pass, dispatch, end compute pass
    [If canvas draws between passes: submit encoder, SkPicture replay via Skia recorder, submit recording, new encoder]
  Copy output buffers → staging buffers
  Submit command encoder
  Begin async map on staging buffers
  Wrap final texture as SkImage → store as latest output

UI thread (useFrameCallback):
  stream.nextImage() → SkImage (latest compute output)
  If onFrame present:
    stream.readBuffer(0..N) → ArrayBuffer | null (from mapped staging)
    Resolve buffer handles to typed arrays via name→index mapping
    Draw compute output onto persistent SkSurface
    Call onFrame(renderFrame, buffers)  [user draws on SkSurface canvas]
    Snapshot SkSurface → composited SkImage
    Update currentFrame shared value with composited SkImage
  Else:
    Update currentFrame shared value with compute output SkImage
  Dispose previous frame
```

### `CameraStreamHostObject` changes

New methods (additive):
- `readBuffer(index: number): ArrayBuffer | null` — returns data from the mapped staging buffer at the given index. Returns `null` if no readback has completed yet.
- `getCanvas(): JsiSkCanvas | null` — returns the SkSurface canvas for Skia drawing
- `flushCanvas(): void` — snaps the Skia recording and submits it. The composited result is read by the next `nextImage()` call.

`nextImage()` unchanged.

### Cleanup on unmount

Same pattern as v1: dispose final frame, null stream, call `cleanupMultiPassPipeline()`. Cleanup tears down:
- All compiled `wgpu::ComputePipeline` objects
- Both ping-pong textures
- All output buffers
- All staging buffers (unmapped first if currently mapped)
- The SkSurface (if allocated)

### Error handling

Same as v1: `error: string | null` on the result. If any shader in the chain fails to compile, the error identifies which pass failed (e.g. "Shader compilation failed at pass 2 of 3").

---

## Native API Changes

### `setupComputePipeline` → `setupMultiPassPipeline`

New native function replacing `setupComputePipeline`:

```typescript
// Called from hook setup
WebGPUCameraModule.setupMultiPassPipeline({
  shaders: string[],           // ordered WGSL strings
  width: number,
  height: number,
  buffers: Array<{             // one per runShader with output
    passIndex: number,         // which pass produces this buffer
    elementSize: number,       // bytes per element (derived from constructor's BYTES_PER_ELEMENT)
    count: number,             // number of elements
  }>,
  useCanvas: boolean,          // whether to allocate SkSurface
  sync: boolean,               // whether to block on buffer readback
});
```

The v1 `setupComputePipeline(wgsl, width, height)` and its C interface `dawn_pipeline_setup` are removed. The shorthand hook form constructs the config internally — the capture proxy collects all `runShader` calls into the shaders array (supporting multi-pass in shorthand), with empty buffers and useCanvas false.

### `processFrame` changes

`processFrame` now iterates over compiled pipelines, ping-ponging textures, dispatching each pass, and handling buffer copies + async mapping. The camera thread still calls it per-frame — the `DawnPipelineBridge.processFrame:` interface doesn't change, only the internal implementation.

---

## Scope

### In scope
- Multiple `runShader()` calls with ping-pong textures
- `frame.canvas` backed by real SkSurface (replaces Proxy stub)
- Canvas draws between shader passes (via SkPicture replay)
- Canvas draws in `onFrame` (per-frame dynamic draws)
- Buffer readback with typed array return
- `sync` flag for blocking vs best-effort readback
- Object form `{ pipeline, onFrame, sync }` for the hook
- Shorthand form backwards compatible (compute only, no canvas)
- iOS only

### Not in scope
- Frame history (`useTexture`, `copyToTexture`) — separate future work, API TBD
- Android support
- Recording API
- Camera controls
- `runShader` bindings parameter (for passing additional textures)

## File Locations

### TypeScript (modify)
- `packages/react-native-webgpu-camera/src/types.ts` — updated types (`ProcessorFrame`, `RenderFrame`, `BufferHandle`, `ProcessorConfig`, overloads)
- `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` — multi-pass capture proxy, object form handling, onFrame worklet, buffer resolution
- `packages/react-native-webgpu-camera/src/index.ts` — re-exports (new types)

### Native (modify)
- `modules/webgpu-camera/ios/DawnComputePipeline.mm` — multi-pipeline, ping-pong textures, staging buffers, SkSurface, updated CameraStreamHostObject
- `modules/webgpu-camera/ios/DawnComputePipeline.h` — updated C interface (remove old, add new)
- `modules/webgpu-camera/ios/DawnPipelineBridge.mm` — bridge updated setup method
- `modules/webgpu-camera/ios/DawnPipelineBridge.h` — bridge updated interface
- `modules/webgpu-camera/ios/WebGPUCameraModule.swift` — call new setup function, remove old
