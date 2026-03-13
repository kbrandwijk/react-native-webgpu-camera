# Spike Implementation Design

## Goal

Implement all four spikes from the Phase 0 spike plan in a single example app, validating the foundational viability questions for react-native-webgpu-camera. One EAS device build tests everything.

This spike produces a single integrated proof-of-concept app. While architecture decisions are documented here to guide implementation, the code is exploratory and will be rewritten in Phase 1.

## Approach: "Skia Owns the World" (Approach A)

Use react-native-skia's Graphite backend as the single GPU context owner. Skia Graphite creates a Dawn Instance + Device and exposes it as `navigator.gpu`. All WebGPU compute, rendering, and recording use this shared device. If Graphite's shared context doesn't support the full pipeline, fall back to Approach B (react-native-wgpu Canvas + Ganesh overlay).

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Skia Graphite (Dawn Instance + Device)             │
│  └─ navigator.gpu ← shared WebGPU device            │
│     ├─ Compute pipeline (Sobel shader)              │
│     ├─ Skia Canvas (overlay drawing)                │
│     └─ Render targets (screen + recorder surface)   │
├─────────────────────────────────────────────────────┤
│  Camera Module (Swift / Kotlin Expo Module)         │
│  └─ AVCaptureSession / Camera2                      │
│     └─ Pixel buffer → Rust frame slot → JS worklet  │
│        └─ device.queue.writeTexture()               │
├─────────────────────────────────────────────────────┤
│  Worklet Thread (Reanimated)                        │
│  └─ Render loop:                                    │
│     1. Poll camera frame from Rust frame slot       │
│     2. writeTexture → input texture                 │
│     3. Dispatch Sobel compute shader                │
│     4. Skia draws overlays onto output texture      │
│     5. Render pass → screen surface                 │
│     6. If recording: render pass → recorder surface │
│     7. metrics.recordFrame()                        │
├─────────────────────────────────────────────────────┤
│  Recorder Module (Swift / Kotlin)                   │
│  └─ AVAssetWriter / MediaRecorder                   │
│     └─ Owns native surface registered with Dawn     │
└─────────────────────────────────────────────────────┘
```

### Key decisions

- **No react-native-wgpu `<Canvas>` component** — we use Skia's Canvas, which is backed by Graphite/Dawn.
- **Camera in Swift/Kotlin** — not Rust. AVCaptureSession and Camera2 are first-class native APIs. Rust keeps the thread-safe frame slot and UniFFI bridge for cross-thread coordination.
- **Recorder in Swift/Kotlin** — same reasoning. AVAssetWriter and MediaRecorder are native APIs.
- **Worklet thread drives the render loop** — camera polling, compute dispatch, Skia overlay, and render pass all happen off the JS main thread.
- **Single texture flows through the pipeline** — camera → compute → Skia overlays → screen/recorder. No intermediate copies within the GPU.

### Fallback: Approach B

If Graphite's shared context doesn't support compute dispatch or bidirectional texture access, the fallback is:

- Use react-native-wgpu's `<Canvas>` for compute + rendering (separate Dawn device).
- Use react-native-skia with Ganesh backend for overlays.
- Bridge via `readPixels()` → `writeTexture()` (~0.6ms overhead per frame).
- Camera module and recorder module code are identical in both approaches.

---

## Spike 1: Camera Frame → GPU Texture

### Goal

Get camera frames onto a WebGPU texture. Primary delivery is via `device.queue.writeTexture()` (copy path). Zero-copy import is investigated but not expected to work without patching react-native-wgpu (see investigation below).

### Where the code lives

- **iOS:** `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`
- **Android:** `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt`
- **Rust frame slot:** `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/lib.rs` (existing stubs)

### Data flow

**iOS:**

```text
AVCaptureSession (configured in Swift)
  → AVCaptureVideoDataOutput delegate callback
  → CMSampleBuffer → CVPixelBuffer
  → CVPixelBufferLockBaseAddress → copy bytes to Rust frame slot
  → JS worklet polls get_current_frame_pixels()
  → device.queue.writeTexture() with the pixel data
```

**Android:**

```text
Camera2 (configured in Kotlin)
  → ImageReader.OnImageAvailableListener
  → Image → planes[0].buffer
  → Copy bytes to Rust frame slot
  → Same JS polling + writeTexture path
```

### Rust frame slot (already scaffolded)

- `CURRENT_FRAME_PIXELS: Mutex<Vec<u8>>` — holds latest BGRA pixel buffer
- `FRAME_DIMS: Mutex<FrameDimensions>` — width, height, bytesPerRow
- `CURRENT_FRAME_HANDLE: AtomicU64` — IOSurface/HardwareBuffer handle captured and logged for future zero-copy path
- Swift/Kotlin camera callback writes in, JS worklet reads out

### Zero-copy investigation (non-blocking)

Prior research confirmed react-native-wgpu v0.5.8 does not expose `importExternalTexture()` or any mechanism for importing platform-native GPU resources. Dawn internally supports IOSurface (iOS) and AHardwareBuffer (Android) via `SharedTextureMemory`, but react-native-wgpu's JSI bridge doesn't surface these APIs. `ExternalTextureDescriptor` is `// TODO: implement` in the C++ source.

The spike should still:

1. **Capture and log the native handle** — IOSurface handle (iOS) and HardwareBuffer handle (Android) to prove they exist and are accessible
2. **Check react-native-wgpu source/issues** for any new work on external texture import since v0.5.8
3. **Attempt Dawn C++ API calls from Rust** — if the handle is available, try calling Dawn's `SharedTextureMemory` API directly to see if it's reachable without JSI changes

If any of these succeed unexpectedly, zero-copy works. If not (expected), the copy path is the validated result and zero-copy is deferred to the follow-up task (patching react-native-wgpu, ~300-500 lines C++).

### Frame slot performance note

The current `get_current_frame_pixels()` clones the entire `Vec<u8>` through a `Mutex` on every call (~8MB at 1080p BGRA, ~240MB/s at 30fps). This will likely need optimization for sustained frame rates — double-buffering with atomic swap, or a shared memory pointer. Flag during spike if it becomes a bottleneck.

### Validation criteria

- Camera frames appear as WebGPU textures on a physical device
- Console logs report: copy path used, time from camera callback to texture available
- Frame rate sustained at target (30fps minimum)

---

## Spike 2: WebGPU Compute Dispatch from Worklet Thread

### Goal

Dispatch the Sobel edge detection compute shader from a Reanimated worklet at 30-60fps.

### Where the code lives

- `apps/example/src/hooks/useGPUPipeline.ts` (new) — device setup, pipeline creation, buffer management
- `apps/example/src/shaders/sobel.wgsl.ts` (exists) — the compute shader

### Setup (JS main thread)

In Approach A, `navigator.gpu` is provided by Skia Graphite's Dawn backend (see `RNSkManager.cpp` in react-native-skia). The device obtained here should be the same Dawn device that Skia uses for rendering — Spike 3 validates this. If Approach B, the device comes from react-native-wgpu's standalone Canvas instead.

```typescript
// In Approach A, this returns Skia Graphite's shared Dawn device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const module = device.createShaderModule({ code: SOBEL_WGSL });
const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: { module, entryPoint: 'main' }
});

const inputTexture = device.createTexture({ /* 1920x1080, rgba8unorm */ });
const outputTexture = device.createTexture({ /* same */ });
```

### Render loop (worklet thread)

```typescript
const renderFrame = () => {
  'worklet';

  // 1. Get camera pixels from Rust frame slot
  const pixels = WebGPUCameraModule.getCurrentFramePixels();
  const dims = WebGPUCameraModule.getFrameDimensions();

  // 2. Upload to GPU
  device.queue.writeTexture(
    { texture: inputTexture },
    pixels,
    { bytesPerRow: dims.bytesPerRow },
    { width: dims.width, height: dims.height }
  );

  // 3. Dispatch Sobel compute
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(dims.width / 16),
    Math.ceil(dims.height / 16)
  );
  pass.end();
  device.queue.submit([encoder.finish()]);
};
```

### Loop driver

No `requestAnimationFrame` in worklets. The render loop mechanism is an open question the spike must answer. Options to try in order:

1. **`context.present()` blocking** — react-native-wgpu's Canvas context may block until vsync, providing natural frame pacing
2. **`setTimeout(renderFrame, 16)`** — simple timer-based loop at ~60fps
3. **Camera-driven** — trigger render on each new camera frame arrival (natural 30fps pacing from camera)

The spike should determine which mechanism gives the most stable frame pacing.

### Validation criteria

- `createShaderModule()` works on worklet thread
- `createComputePipeline()` works (no async compilation issues)
- `beginComputePass` / `dispatchWorkgroups` / `submit` work synchronously
- Stable for 60 seconds: no crashes, no memory leaks
- Metrics: frames dispatched, frames dropped, min/avg/max frame time, thermal state

---

## Spike 3: Skia Graphite Shared Context

### Goal

Validate that Skia Graphite's `navigator.gpu` device supports the full pipeline: compute dispatch, texture round-tripping (WebGPU texture ↔ Skia), and overlay drawing. This is the gate for Approach A.

### Where the code lives

- `apps/example/src/components/SpikeOverlay.tsx` (new) — Skia overlay component

### What we're testing

1. **Graphite activates** — `navigator.gpu` exists at runtime, returns a working device
2. **Compute works on Graphite's device** — `createComputePipeline` + `dispatchWorkgroups` succeeds (confirms the device from `navigator.gpu` is the same Dawn device Skia uses)
3. **Skia can draw onto our compute output texture** — Skia targets the compute output as a canvas surface, draws overlays (text, shapes, debug info) directly onto it
4. **The composited texture is usable downstream** — the same texture (compute output + Skia overlays) feeds both screen rendering and recorder

### Specific API verification items

These are the concrete things to try, in order:

- Does `SK_GRAPHITE=1` build successfully with current react-native-skia? (May be the default — check build config)
- Does Graphite auto-detect Dawn? (react-native-skia docs say it does)
- Can you obtain a Skia `SkSurface` backed by a Dawn texture? Try: `SkSurfaces::WrapBackendTexture()` with a `GrBackendTexture` created from the compute output's Dawn texture handle
- Can a Skia canvas `drawImage()` with a WebGPU texture as the image source? Try: wrap the compute output GPUTexture as a `SkImage` via `SkImages::BorrowTextureFrom()` or Graphite's equivalent
- Are there separate `GPUDevice` instances, or is it a shared context? Try: compare device IDs or create a buffer on one device and use it on the other
- Can Skia's canvas output be read back as a `GPUTexture`? Try: get the backing texture from Skia's rendered surface for the recorder to consume

### Fallback cascade

1. **Best:** Skia draws directly onto the compute output texture — zero overhead, single texture through entire pipeline
2. **Acceptable:** Skia draws to its own texture on the shared device, we composite both in a final render pass (GPU→GPU alpha blend, trivial shader, negligible cost) — still Approach A
3. **Last resort:** Approach B — separate contexts, react-native-wgpu Canvas for compute/rendering, Ganesh readback for Skia overlays

There is no hybrid "Graphite for rendering but Ganesh for overlays." It's Approach A (all shared context) or Approach B (separate contexts).

### Validation criteria

- Graphite mode builds and runs on device
- Compute shader dispatches successfully on Graphite's device
- Skia overlay content (text, shapes) visible on the rendered output
- Composited frame (compute + overlays) available as a single texture for recording
- Measure: overlay draw time, compositing overhead (if fallback 2)

---

## Spike 4: Rendering to Platform Recorder Surface

### Goal

Test whether Dawn can render to a platform recorder surface, enabling zero-copy video recording.

### Where the code lives

- **iOS:** AVAssetWriter setup in `WebGPUCameraModule.swift`
- **Android:** MediaRecorder/MediaCodec setup in `WebGPUCameraModule.kt`
- Rust `start_test_recorder` / `stop_test_recorder` stubs move to Swift/Kotlin (same reasoning as camera: native APIs in native languages)

### iOS approach

Dawn's `makeSurface()` requires a `CAMetalLayer*`. This is normally an on-screen layer, but we need it as a bridge to AVAssetWriter. This is a non-standard usage and a high-uncertainty step — the spike should test whether a CAMetalLayer created programmatically (not attached to a UIView) can feed frames to AVAssetWriter via an IOSurface intermediary. If this doesn't work, the alternative is rendering to a Dawn texture and using `readPixels()` to feed CVPixelBuffers to AVAssetWriter directly.

```text
AVAssetWriter + AVAssetWriterInputPixelBufferAdaptor
  → Create offscreen CAMetalLayer (not attached to a view)
  → Dawn makeSurface(instance, metalLayer, w, h) → wgpu::Surface
  → Render pass writes composited frame to this surface
  → Extract IOSurface from CAMetalLayer's nextDrawable
  → Feed IOSurface to AVAssetWriter
```

### Android approach

More straightforward — `MediaRecorder.getSurface()` returns a standard `Surface` backed by `ANativeWindow`, which is exactly what Dawn's `makeSurface()` accepts.

```text
MediaRecorder.getSurface() → Surface
  → ANativeWindow_fromSurface() → ANativeWindow*
  → Dawn makeSurface(instance, window, w, h) → wgpu::Surface
  → Render pass writes composited frame to this surface
  → MediaRecorder encodes directly
```

### What we're testing

1. Can we create a Dawn `wgpu::Surface` from a recorder-owned native surface?
2. Can we render to it alongside the screen surface in the same frame?
3. Does the recorded output contain correct frames at the expected framerate?
4. Any frame pacing / synchronization issues?

### Dual-target rendering

The composited texture (compute output + Skia overlays) gets rendered to two targets each frame:

- Screen surface (for preview)
- Recorder surface (for capture, when recording is active)

Each target gets a simple render pass that samples the composited texture and draws a fullscreen quad.

### Fallback cascade

1. **Best:** Direct surface rendering works — zero-copy recording
2. **Acceptable:** GPU texture copy to recorder surface (GPU→GPU via Dawn's `CopyTextureToTexture` or `SurfaceInfo` offscreen→onscreen mechanism)
3. **Last resort:** GPU readback → CPU → platform encoder (~2ms overhead during recording only)

### Validation criteria (per spec)

- A 5-second video file recorded on physical device
- Contains frames rendered by Dawn (test pattern or processed camera feed)
- Video plays correctly with correct frame rate
- If audio enabled, A/V sync correct

---

## Spike 5: End-to-End Integration

Per the spike plan, Spike 5 integrates results from Spikes 1-4. It runs the full pipeline for 60 seconds on a physical device and reports sustained FPS, frame drops, CPU usage, memory, and thermal state.

Since we're building all spikes in a single app, Spike 5 is essentially "run the app and measure." The metrics infrastructure already exists in `apps/example/src/hooks/useSpikeMetrics.ts`.

### Go/no-go per spec

- Sustained 60fps, <5% frame drops → proceed to Phase 1
- Sustained 30fps but not 60fps → investigate bottleneck, 30fps acceptable for MVP
- Cannot sustain 30fps → fundamental architecture issue, re-evaluate

---

## Files to Create or Modify

### New files

| File | Purpose |
|------|---------|
| `apps/example/src/hooks/useGPUPipeline.ts` | Device setup, compute pipeline, render loop |
| `apps/example/src/components/SpikeOverlay.tsx` | Skia overlay component |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Camera capture + recorder (iOS) |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt` | Camera capture + recorder (Android) |

### Modified files

| File | Change |
|------|--------|
| `apps/example/src/app/index.tsx` | Wire up real pipeline (replace stubs) |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/lib.rs` | May simplify if camera/recorder move to Swift/Kotlin |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/` | iOS/Android camera modules (may become thin wrappers or removed) |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/recorder/` | iOS/Android recorder modules (may become thin wrappers or removed) |

### Unchanged

| File | Reason |
|------|--------|
| `apps/example/src/shaders/sobel.wgsl.ts` | Already complete |
| `apps/example/src/hooks/useSpikeMetrics.ts` | Already complete |
| `packages/react-native-webgpu-camera/src/index.ts` | Library API not changing during spike |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graphite's `navigator.gpu` doesn't support compute | Kills Approach A | Fall back to Approach B (rn-wgpu Canvas + Ganesh) |
| Skia can't draw onto a WebGPU-created texture | Need compositing render pass | Fallback 2: Skia draws to own texture, GPU blend |
| Dawn can't create surface from recorder's native surface | No zero-copy recording | GPU readback fallback (~2ms during recording) |
| Worklet thread can't call WebGPU APIs | Different threading model needed | Main thread compute (blocks JS ~5-10ms) or native thread |
| Camera pixel copy too slow at target resolution | Frame drops | Reduce resolution or target 30fps instead of 60fps |
| Graphite mode doesn't build with current react-native-skia | No shared context | Approach B |
| iOS: offscreen CAMetalLayer → AVAssetWriter is non-standard | May not work for recording | Fall back to readPixels → CVPixelBuffer → AVAssetWriter |
| Rust frame slot clone (~8MB/call) too slow at 30fps | Frame drops, GC pressure | Double-buffer with atomic swap or shared memory pointer |

---

## Follow-up Tasks (Not In Scope)

- **Zero-copy camera import:** Patch react-native-wgpu to expose Dawn's SharedTextureMemory for IOSurface/HardwareBuffer import. ~300-500 lines C++. Logged as follow-up.
- **Shared context upstream contribution:** If Approach A works, contribute the integration pattern back to react-native-wgpu / react-native-skia.
