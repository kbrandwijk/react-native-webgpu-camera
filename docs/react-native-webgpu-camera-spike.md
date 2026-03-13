# react-native-webgpu-camera — Phase 0 Spike

## Purpose

This spike answers the foundational viability questions before any real
implementation begins. Every architecture decision in the main plan
(`react-native-webgpu-camera-plan.md`) depends on the outcomes here.

**No production code is written during this spike.** The output is a set of
minimal proof-of-concept apps (one per question), each answering a single
binary question with a working demo on a physical device. Based on the
results, the main plan is updated to commit to specific architecture paths.

**Timeline:** 1-2 weeks. All spikes can run in parallel.

**Hardware required:** At least one physical iOS device (iPhone 12+) and one
physical Android device (API 28+, Pixel 5+ or Samsung S21+ recommended).
Simulators/emulators are insufficient — they don't have real camera buffers,
hardware-backed GPU interop, or accurate GPU performance characteristics.

---

## Spike 1: Zero-Copy Camera Frame → WebGPU Texture

### Question

Can we import a platform camera buffer (IOSurface on iOS, HardwareBuffer on
Android) as a Dawn/WebGPU texture without copying pixel data through CPU?

### Why This Matters

If yes: frame delivery is ~0ms, entire pipeline stays on GPU, 60fps at 4K
is feasible, and the architecture is genuinely zero-copy end-to-end.

If no: we fall back to `device.queue.writeTexture()` with pixel data copied
through CPU. This costs ~0.3ms at 1080p and ~1.2ms at 4K. Still viable for
60fps at 1080p, but changes our performance claims and makes 4K60 tight.

### What to Build

Minimal Expo app with:
1. A native module (Rust or Objective-C/Kotlin — whichever is faster to
   spike with) that opens the camera and extracts the platform buffer handle
2. A React component that receives the handle and attempts to create a
   `GPUTexture` from it via `react-native-wgpu`

**iOS path to test:**
```
AVCaptureSession → AVCaptureVideoDataOutput
  → CMSampleBufferGetImageBuffer() → CVPixelBuffer
  → CVPixelBufferGetIOSurface() → IOSurface handle
  → Pass handle to JS via JSI
  → JS calls Dawn API to create texture from IOSurface
```

The key unknown is the last step. Dawn's Metal backend *can* create an
`MTLTexture` from an `IOSurface` — Chrome does this for video decoding.
But does `react-native-wgpu` expose this capability through the JS API?

**Investigate:**
- Does `react-native-wgpu` support `importExternalTexture()` or any
  mechanism for importing platform-native GPU resources?
- If not exposed in JS, can we call Dawn's C++ API directly from Rust
  to create the texture, then pass the `GPUTexture` handle to JS?
- Does William (wcandillon) have any existing work or plans for external
  texture import? Check react-native-wgpu issues and discussions.

**Android path to test:**
```
Camera2 → ImageReader → Image.getHardwareBuffer()
  → HardwareBuffer handle
  → Dawn Vulkan backend imports via
    VK_ANDROID_external_memory_android_hardware_buffer
  → GPUTexture available in JS
```

Same unknown: is this import path exposed through `react-native-wgpu`?

### Fallback Path (if zero-copy fails)

If external texture import isn't available through react-native-wgpu:

```typescript
// The copy path — proven to work, costs ~0.3ms at 1080p
const pixels = getCameraFrameAsArrayBuffer(); // from native module
device.queue.writeTexture(
  { texture: inputTexture },
  pixels,
  { bytesPerRow: width * 4 },
  { width, height }
);
```

This fallback doesn't block the project. It changes the performance profile
and some marketing claims, but the core architecture (compute → Skia → capture)
works identically regardless of how the frame got onto the GPU.

### Go/No-Go Criteria

| Result | Decision |
|--------|----------|
| ✅ Zero-copy works on both platforms | Use zero-copy as primary path in main plan |
| ⚠️ Zero-copy works on iOS only | Use zero-copy on iOS, copy fallback on Android |
| ⚠️ Zero-copy works but requires Dawn C++ calls from Rust | Acceptable — add to Rust module scope |
| ❌ Zero-copy not possible through react-native-wgpu | Use copy path. Update main plan: remove zero-copy claims, adjust performance targets, simplify Rust module (no GPU bridge needed, just pixel buffer extraction) |

### Deliverable

A demo app running on a physical device that displays the live camera feed
rendered via WebGPU. Console logs report: copy/zero-copy path used, time
from camera callback to texture available (nanoseconds).

---

## Spike 2: WebGPU Compute Dispatch from Worklet Thread

### Question

Can we call WebGPU APIs (createShaderModule, createComputePipeline,
beginComputePass, dispatchWorkgroups) from a Reanimated worklet thread that
receives camera frames?

### Why This Matters

If yes: the entire compute + render pipeline runs on a dedicated worklet
thread, off the JS main thread, with no cross-thread synchronization.

If no: we need a different threading model — either compute on the main JS
thread (bad for UI responsiveness) or a custom native thread with manual
signaling (more complex Rust code).

### What to Build

Minimal app that:
1. Sets up a `react-native-wgpu` Canvas with a GPUDevice
2. Creates a Reanimated worklet that receives the GPUDevice
3. Inside the worklet, creates a compute pipeline and dispatches it
4. Reads back a result buffer to verify the compute shader ran correctly

```typescript
import { Canvas } from 'react-native-wgpu';
import { runOnUI } from 'react-native-reanimated';

// Create device on main thread
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Attempt compute dispatch on worklet thread
const computeOnWorklet = (device: GPUDevice) => {
  'worklet';
  
  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> output: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3u) {
        output[id.x] = f32(id.x) * 2.0;
      }
    `
  });
  
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' }
  });
  
  // Create buffer, bind group, dispatch, submit
  // ...
  
  // If this executes without crashing, the spike passes
};

runOnUI(computeOnWorklet)(device);
```

**Key things to verify:**
- Does `device.createShaderModule()` work on the worklet thread?
- Does `device.createComputePipeline()` work? (May involve async compilation)
- Does command encoder / dispatch / submit work synchronously on the worklet?
- What happens if you dispatch every frame at 60fps for 60 seconds?
  (Stability test — no crashes, no memory leaks, no frame drops)

### Fallback Path (if worklet dispatch fails)

If WebGPU APIs can't be called from Reanimated worklets:

**Option A:** Run compute on the main JS thread. Camera frames arrive via
a callback, compute dispatches happen synchronously. This blocks the JS
thread during processing (~5-10ms per frame) but may be acceptable if the
compute pipeline is fast enough.

**Option B:** Create a dedicated native thread in Rust that owns the
GPUDevice, receives frame handles, dispatches compute, and signals JS
when results are ready. More complex but keeps JS thread clean.

react-native-wgpu explicitly documents passing GPUDevice to worklets and
has demos of rendering on worklet threads, so this spike is expected to
pass. But "rendering" (render pass) and "compute" (compute pass) may
behave differently — we need to verify compute specifically.

### Go/No-Go Criteria

| Result | Decision |
|--------|----------|
| ✅ Compute dispatch works on worklet thread at 60fps | Proceed with worklet-based architecture |
| ⚠️ Works but with occasional frame drops | Profile and optimize — likely acceptable |
| ❌ Crashes or hangs on worklet thread | Use Option A (main thread) or Option B (native thread). Update main plan threading model |

### Deliverable

A demo app that dispatches a compute shader 60 times per second on a worklet
thread for 60 seconds, reads back results, and reports: success/failure,
frames dispatched, frames dropped, any errors.

---

## Spike 3: Skia Graphite + Dawn Shared GPU Context

### Question

When Skia is built with `SK_GRAPHITE=1` and react-native-wgpu provides Dawn,
do they share the same `GPUDevice` / GPU context? Can Skia draw onto a
WebGPU texture, and can WebGPU sample a Skia-rendered texture?

### Why This Matters

If yes: zero-cost compositing of Skia 2D overlays on WebGPU compute output.
Bidirectional texture sharing. This is the premium path.

If no: Skia and WebGPU are separate GPU contexts. Compositing requires
rendering Skia to an offscreen surface, reading pixels to CPU, uploading
to a WebGPU texture. This is the Ganesh fallback path — works but adds
~0.6ms per frame.

### What to Build

Minimal app that:
1. Builds react-native-skia with `SK_GRAPHITE=1`
2. Creates a react-native-wgpu Canvas with a GPUDevice
3. Creates a WebGPU texture (e.g., solid red via a compute shader)
4. Attempts to use that texture as a source in a Skia canvas
5. Attempts to draw Skia content (text, shapes) onto the WebGPU texture
6. Renders the final composite to the screen

**What to verify:**
- Does `SK_GRAPHITE=1` build successfully with current react-native-skia?
- Does Graphite auto-detect Dawn? (react-native-skia docs say it does)
- Can you obtain a Skia `SkSurface` backed by a Dawn texture?
- Can a Skia canvas `drawImage()` with a WebGPU texture as the image source?
- Are there separate `GPUDevice` instances, or is it a shared context?

**Also test the Ganesh fallback:**
- Build WITHOUT `SK_GRAPHITE=1`
- Render Skia to an offscreen surface
- `readPixels()` → CPU buffer → `device.queue.writeTexture()` → WebGPU texture
- Composite in a WebGPU render pass
- Measure the round-trip cost (expected ~0.6ms at 1080p)

### Go/No-Go Criteria

| Result | Decision |
|--------|----------|
| ✅ Shared context, bidirectional texture access | Use Graphite as primary path. Document as key feature. |
| ⚠️ Graphite works but textures aren't directly shareable | May need explicit texture copy within Dawn (GPU→GPU, fast). Investigate Dawn texture sharing APIs. |
| ⚠️ Graphite builds but isn't stable (crashes, rendering glitches) | Use Ganesh fallback for MVP. Revisit Graphite in a later phase. |
| ❌ Graphite build fails or doesn't detect Dawn | Use Ganesh fallback. Update main plan: remove bidirectional compositing from feature list, keep overlay-in-capture via copy path. |

### Deliverable

A demo app showing: (a) a WebGPU-rendered scene with Skia text/shapes
composited on top, rendered to screen. (b) Ganesh fallback version of the
same composite with measured per-frame overhead.

---

## Spike 4: Rendering to Platform Recorder Surface

### Question

Can Dawn render a WebGPU render pass output to a platform video recorder's
input Surface?

**iOS:** Can we render to an `IOSurface` that feeds `AVAssetWriterInput`?
**Android:** Can we render to a `Surface` from `MediaRecorder.getSurface()`?

### Why This Matters

If yes: video recording is "elegantly simple" — we render processed frames
to the recorder's Surface, and the platform handles audio, encoding, muxing,
and A/V sync internally. ~80-100 lines of native code per platform.

If no: we fall back to GPU readback per frame → feed raw pixels to encoder.
On iOS this means `AVAssetWriter` with `CVPixelBuffer` from readback. On
Android this means `MediaCodec` with input buffers (not Surface), plus
manual `AudioRecord` + `MediaMuxer` + timestamp alignment. This is
VisionCamera's approach — ~300-400 lines per platform, error-prone A/V sync.

### What to Build

Two minimal apps (one per platform):

**iOS spike:**
1. Create a Dawn GPUDevice
2. Create an `IOSurface` (or obtain one from `AVAssetWriterInputPixelBufferAdaptor`)
3. Create a Dawn texture backed by that IOSurface
4. Render a simple scene (colored quad, incrementing frame counter) to it
5. Feed the IOSurface to AVAssetWriter
6. Record 5 seconds, save the file, verify the video contains the rendered frames

**Android spike:**
1. Configure a `MediaRecorder` with Surface input
2. Get the Surface from `MediaRecorder.getSurface()`
3. Create a Dawn/Vulkan swapchain or texture targeting that Surface
4. Render a simple scene to it at 30fps
5. Start/stop MediaRecorder, save the file, verify the video is valid

**The key unknown for Android:** Dawn typically creates its own Surface via
`ANativeWindow`. `MediaRecorder.getSurface()` returns a different Surface.
Can Dawn render to an arbitrary Android Surface it didn't create? This may
require creating a `VkSurfaceKHR` from the MediaRecorder's `ANativeWindow`.

**Alternative Android approach if direct Surface fails:**
Use a shared `HardwareBuffer` as an intermediary:
```
Dawn renders to HardwareBuffer-backed texture
  → Same HardwareBuffer imported as Surface for MediaCodec
  (not MediaRecorder — MediaCodec gives more control over surface binding)
```

This is more complex but still zero-copy on the GPU side.

### Go/No-Go Criteria

| Result | Decision |
|--------|----------|
| ✅ Direct rendering to recorder Surface works on both platforms | Use MediaRecorder (Android) + AVAssetWriter (iOS) as documented. ~80-100 lines per platform. |
| ⚠️ Works on iOS, fails on Android | Use AVAssetWriter on iOS. On Android, try HardwareBuffer intermediary. If that also fails, fall back to GPU readback → MediaCodec input buffers. |
| ⚠️ Works but with frame pacing issues (jitter, dropped frames) | Investigate double-buffering or frame pacing synchronization. May be solvable with a presentation timestamp strategy. |
| ❌ Fails on both platforms | Fall back to GPU readback → platform encoder. Per-frame readback at 1080p30 is ~8MB/s × 30 = ~240MB/s, within memory bandwidth budget but not zero-copy. Update main plan: recording architecture, complexity estimates, native code scope all increase. |

### Deliverable

A 5-second video file recorded on each physical device, containing frames
rendered by Dawn (not camera frames — just a test pattern or animated scene).
Video must play correctly, have correct frame rate, and if audio is enabled,
be in sync.

---

## Spike 5: End-to-End Camera → Compute → Render at 60fps

### Question

With the paths proven in Spikes 1-3 (using fallbacks where needed), can we
run the full pipeline — camera frame → GPU texture → compute shader → render
pass → screen — at sustained 60fps on a physical device without frame drops?

### Why This Matters

Individual spikes may pass but the combined pipeline may have latency issues
from threading, synchronization, or resource contention that only appear
under sustained load.

### What to Build

**Prerequisites:** Spikes 1, 2, and 3 must be completed first. Use whichever
paths were proven (zero-copy or fallback copy, Graphite or Ganesh).

Minimal app that:
1. Opens the camera (via native module or VisionCamera as a temporary crutch)
2. Gets each frame as a GPU texture (zero-copy or copy, per Spike 1 result)
3. Runs a non-trivial compute shader on the frame (e.g., edge detection —
   enough work to be measurable, not just a passthrough)
4. Optionally composites a Skia overlay (per Spike 3 result)
5. Renders the result to a WebGPU canvas
6. Runs for 60 seconds on a physical device
7. Reports: sustained FPS, frame drops, CPU usage, memory, thermal state

**This is NOT a VisionCamera comparison.** This is a self-validation: does
our pipeline maintain 60fps under sustained load? The benchmark suite
(Phase 4 in the main plan) handles comparison later.

### Go/No-Go Criteria

| Result | Decision |
|--------|----------|
| ✅ Sustained 60fps, <5% frame drops over 60 seconds | Proceed to Phase 1 implementation |
| ⚠️ Sustained 30fps but not 60fps | Investigate bottleneck. 30fps may be acceptable for MVP. Profile whether bottleneck is copy, compute, render, or threading. |
| ⚠️ 60fps but with thermal throttling after 30+ seconds | Optimize compute shader complexity. May need to offer quality/performance tradeoffs in the API. |
| ❌ Cannot sustain 30fps | Fundamental architecture issue. Re-evaluate threading model, copy costs, shader complexity. |

### Deliverable

Console output or on-screen overlay showing: sustained FPS over 60 seconds,
frame drop count, CPU %, memory MB, thermal state transitions. Screenshot
or screen recording of the running demo.

---

## Spike Execution Plan

### Parallelization

```
Week 1:
  Spike 1 (zero-copy import)     ← can run immediately
  Spike 2 (compute on worklet)   ← can run immediately
  Spike 3 (Skia Graphite)        ← can run immediately
  Spike 4 (recorder Surface)     ← can run immediately

Week 2:
  Spike 5 (end-to-end 60fps)     ← depends on Spikes 1, 2, 3
  Update main plan based on all results
```

Spikes 1-4 are independent and can run in parallel (same developer or
different developers). Spike 5 integrates the results and must wait.

### Who Runs Each Spike

Each spike is a single developer task, not a full agent workstream. The
spikes are small enough (1-2 days each) that one person could run all
five sequentially in a week, or two people could run them in parallel.

**Spike 1** requires native code (camera buffer extraction) — best suited
for someone comfortable with Objective-C or Kotlin and react-native-wgpu.

**Spike 2** is pure TypeScript — any developer familiar with WebGPU and
Reanimated worklets.

**Spike 3** requires building react-native-skia from source with a custom
flag — best suited for someone who's built native React Native dependencies
before.

**Spike 4** requires native code (recorder setup) and understanding of
platform video encoding APIs.

**Spike 5** integrates results from 1-3 and requires a physical device —
can be the same person who ran the earlier spikes.

---

## Post-Spike Decision Matrix

After all spikes complete, update the main plan based on this matrix:

```
                          Spike 1       Spike 2       Spike 3       Spike 4
                          (Zero-Copy)   (Worklet)     (Graphite)    (Recorder)
                          ─────────     ─────────     ──────────    ──────────
Best case (all pass):     Zero-copy     Worklet       Graphite      MediaRecorder
                          import        compute       compositing   + AVAssetWriter
                          ~0ms/frame    off JS thread zero-copy     zero-copy record

Degraded (some fail):     Copy path     Main thread   Ganesh        GPU readback
                          ~0.3ms/frame  ~5-10ms block +0.6ms/frame  → MediaCodec
                                        per frame                   + manual mux

Impact on main plan:      §3.4 updated  §2.2 updated  §2.3 updated  §3.5 rewritten
                          §9 targets    §6.1-6.2      §6.3 tasks    §3.2 API changes
                          adjusted      tasks change   adjust        complexity ↑
```

**Even in the worst case (all spikes fail to the fallback path), the project
is still viable.** The pipeline becomes:

```
Camera → copy to GPU (~0.3ms) → compute on main thread (~5-10ms)
  → Skia overlay via Ganesh copy (~0.6ms) → render pass → preview
  → GPU readback for recording (~2ms, only during recording)
```

Total overhead: ~8-13ms per frame. That's 30fps with headroom, which is
acceptable for an MVP. The zero-copy and worklet paths are performance
optimizations, not architectural requirements.

**The only true project-killer would be:** WebGPU compute shaders don't work
at all on mobile via react-native-wgpu. This is extremely unlikely given the
Shopify team's public demos of compute shaders, TensorFlow.js on WebGPU, and
ComputeToys running in React Native. But Spike 2 explicitly validates this.

---

## Relationship to Main Plan

After the spike phase, the main plan (`react-native-webgpu-camera-plan.md`)
is updated:

1. **Open Questions (§10)** — resolved questions are removed, replaced with
   "validated in spike" references.

2. **Architecture sections** — conditional language ("if zero-copy works")
   is replaced with committed decisions based on spike results.

3. **Agent task breakdown (§6)** — tasks are resequenced into implementation
   phases:
   - **Phase 1 (MVP):** Camera → compute → Skia overlay → photo capture.
     Uses whichever paths the spikes validated. Copy fallbacks are fine.
   - **Phase 2:** Add zero-copy import (if Spike 1 passed). Add Graphite
     (if Spike 3 passed). Performance optimization.
   - **Phase 3:** Add video recording (architecture per Spike 4 results).
     Add audio.
   - **Phase 4:** Examples package + benchmark suite. Only after the core
     pipeline is stable and proven on physical devices.

4. **Agent deliverables** — split into "simulator harness" (code compiles,
   TypeScript tests pass, static image processing works) and "physical device
   milestone" (actual camera frames, actual GPU interop, actual capture output
   verified in Photos app).

5. **Performance targets (§9)** — adjusted based on which paths are zero-copy
   vs copy. Targets reference measured spike results, not theoretical best case.
