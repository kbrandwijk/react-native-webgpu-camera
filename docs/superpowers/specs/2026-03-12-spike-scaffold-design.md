# Spike Scaffold Design

Scaffold the real monorepo, core package, Expo module, Rust setup, and example
app — then implement minimal paths to validate all four independent spikes from
`docs/react-native-webgpu-camera-spike.md` in a single camera screen.

---

## 1. Monorepo Structure

Bun workspaces. No Nx, no Turborepo.

```
react-native-webgpu-camera/
├── package.json                  ← bun workspace root
├── bun.lock
├── tsconfig.json                 ← base tsconfig
├── .gitignore
├── CLAUDE.md
├── docs/
│   ├── react-native-webgpu-camera-plan.md
│   └── react-native-webgpu-camera-spike.md
├── scripts/
│   ├── build-rust.sh             ← cross-compile Rust for all targets
│   ├── generate-bindings.sh      ← UniFFI → TypeScript + native bindings
│   └── build-skia-graphite.sh    ← wrapper: SK_GRAPHITE=1 bun run build-skia
│
├── packages/
│   └── react-native-webgpu-camera/
│       ├── package.json
│       ├── tsconfig.json
│       ├── src/
│       │   └── index.ts
│       └── modules/
│           └── webgpu-camera/    ← local Expo module (create-expo-module --local)
│               ├── expo-module.config.json
│               ├── src/          ← Swift/Kotlin bridge (thin, calls into Rust)
│               ├── ios/
│               ├── android/
│               └── rust/         ← Rust crate
│                   ├── Cargo.toml
│                   ├── ubrn.config.yaml
│                   └── src/
│                       └── lib.rs
│
└── apps/
    └── example/                  ← Expo 55 app
        ├── package.json
        ├── app.json
        ├── eas.json
        └── app/                  ← Expo Router
            └── (tabs)/
                └── index.tsx     ← single camera screen proving all 4 spikes
```

Key decisions:

- Expo module lives inside the core package at
  `packages/react-native-webgpu-camera/modules/webgpu-camera/`, following the
  `create-expo-module --local` pattern.
- Rust crate lives inside the Expo module directory. UniFFI generates bindings
  that the module's Swift/Kotlin code calls.
- Single example app with one camera screen exercising all spike concerns.
- No benchmark app, no `@webgpu-camera/examples` package — not needed for spike.

---

## 2. Dependencies & Versions

### Core package (`packages/react-native-webgpu-camera`)

| Dependency | Version | Role |
|------------|---------|------|
| `react-native-wgpu` | ^0.5.8 | WebGPU API (Dawn) |
| `@shopify/react-native-skia` | latest Expo 55 compat | Skia 2D drawing (built with `SK_GRAPHITE=1`) |
| `react-native-reanimated` | >=4.2.1 | Worklet threading (peer dep of react-native-wgpu) |
| `react-native-worklets` | >=0.7.2 | Worklet core (peer dep of react-native-wgpu). **Verify exact package name from react-native-wgpu's peerDependencies — may be `react-native-worklets-core`.** |

### Expo module (`modules/webgpu-camera`)

| Dependency | Version | Role |
|------------|---------|------|
| `uniffi-bindgen-react-native` | ^0.30.0 | Rust → TS/native bindings |

### Rust crate

| Dependency | Version | Role |
|------------|---------|------|
| `uniffi` | version must match uniffi-bindgen-react-native | Binding annotations. **Pin to whatever version ubrn 0.30.x requires — check its Cargo.toml.** |
| Platform FFI crates | as needed | `objc2` (iOS), `jni` (Android) |

### Example app (`apps/example`)

| Dependency | Version | Role |
|------------|---------|------|
| `expo` | ^55.0.0 | Framework |
| `react-native` | 0.83.x | Comes with Expo 55 |
| `expo-router` | SDK 55 compat | Navigation |
| `expo-camera` | SDK 55 compat | Permissions only (not frame capture) |
| `react-native-webgpu-camera` | `workspace:*` | The core package |

### Platform requirements

| Platform | Minimum | Reason |
|----------|---------|--------|
| iOS | 15.1 | Expo 55 floor. **Note:** main plan says iOS 13.0+ — update main plan post-spike to reflect Expo 55's actual floor. |
| Android | API 28 | HardwareBuffer for zero-copy Vulkan import |
| React Native | 0.83 | Expo 55 |
| Architecture | New Architecture only | Expo 55 dropped Legacy Architecture |

---

## 3. Rust Module — Spike Scope

Minimal API surface — just enough to prove the pipeline. Not the full API from
the main plan.

```rust
// lib.rs — spike-only surface

#[uniffi::export]
pub fn start_camera_preview(device_id: String, width: u32, height: u32);

#[uniffi::export]
pub fn stop_camera_preview();

/// Returns an opaque handle to the current frame's platform buffer.
/// iOS: IOSurface pointer. Android: HardwareBuffer pointer.
/// JS side uses this to attempt zero-copy GPU texture import.
///
/// Frame delivery model: polling. The Rust side holds the latest frame
/// in a thread-safe slot (AtomicPtr or Mutex). The camera callback
/// updates this slot on each new frame. JS calls this function on its
/// render loop tick to pull the latest frame. If the same frame is
/// pulled twice (render loop faster than camera), the handle is the
/// same — no harm. If a frame is skipped (camera faster than render
/// loop), the older frame is dropped — also fine for a preview pipeline.
#[uniffi::export]
pub fn get_current_frame_handle() -> u64;

/// Fallback: copies current frame pixels into a byte array.
/// Used if zero-copy import isn't available through react-native-wgpu.
#[uniffi::export]
pub fn get_current_frame_pixels() -> Vec<u8>;

/// Returns frame dimensions.
#[uniffi::export]
pub fn get_frame_dimensions() -> FrameDimensions;

#[derive(uniffi::Record)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

// --- Spike 4: recorder Surface ---

/// Creates a platform video recorder and returns its Surface handle.
/// iOS: IOSurface from AVAssetWriterInputPixelBufferAdaptor.
/// Android: Surface from MediaRecorder.getSurface().
#[uniffi::export]
pub fn start_test_recorder(output_path: String, width: u32, height: u32) -> u64;

#[uniffi::export]
pub fn stop_test_recorder() -> String;  // returns output file path

// --- Spike 5: metrics ---

/// Returns device thermal state.
/// iOS: ProcessInfo.ThermalState. Android: PowerManager thermal status.
/// Returns: "nominal", "fair", "serious", or "critical".
#[uniffi::export]
pub fn get_thermal_state() -> String;
```

### What this proves

- **Spike 1:** Can we extract platform camera buffers and pass handles to JS?
  The zero-copy question is answered on the JS/WebGPU side.
- **Spike 4:** Can we obtain a recorder Surface handle and render to it from
  Dawn?

### What's deferred

- Camera controls (zoom, focus, exposure, torch)
- EXIF metadata extraction
- Photo capture
- Device enumeration
- Full recording API (audio, bitrate config, etc.)

---

## 4. Example App — Single Camera Screen

One screen exercises all four spikes simultaneously with automatic fallbacks.

### Screen layout

- Full-screen WebGPU canvas showing processed camera feed
- On-screen debug overlay (Skia text) with real-time metrics
- "Record 5s" button for Spike 4 validation

### Per-frame pipeline

1. **Spike 1 (zero-copy import):** Rust delivers frame handle. JS attempts
   `GPUTexture` import. Falls back to `get_current_frame_pixels()` +
   `device.queue.writeTexture()`.

2. **Spike 2 (compute on worklet):** Inside a Reanimated worklet, dispatches a
   Sobel edge detection compute shader. Non-trivial enough to be measurable.

3. **Spike 3 (Skia Graphite):** Skia draws debug overlay text and test shapes
   onto the WebGPU texture (Graphite) or composites via Ganesh fallback.

4. **Spike 4 (recorder Surface):** "Record 5s" starts the test recorder. Final
   render pass outputs to both preview canvas and recorder Surface.

### Architecture sketch

The spike wires together two independent subsystems:

1. **Rust camera module** — delivers frame handles via a polling API. The
   camera callback updates a thread-safe slot; JS pulls the latest handle
   on each render tick.

2. **react-native-wgpu Canvas** — provides a `GPUDevice` and render loop.
   WebGPU objects are passed to a Reanimated worklet via `runOnUI`.

These are NOT unified by a single hook like VisionCamera's
`useFrameProcessor`. The spike validates whether we _can_ build such a
hook (that's the future `useGPUFrameProcessor` from the main plan). For
the spike, wiring is explicit:

```typescript
// Setup: get GPUDevice from react-native-wgpu Canvas
// Pass device + camera module to worklet thread

const renderLoop = (device: GPUDevice) => {
  'worklet';

  // Spike 1: get camera frame as GPU texture
  const handle = WebGPUCameraModule.get_current_frame_handle();
  let inputTexture: GPUTexture;
  try {
    inputTexture = importExternalTexture(device, handle); // zero-copy attempt
    log('SPIKE1: zero-copy');
  } catch {
    const pixels = WebGPUCameraModule.get_current_frame_pixels();
    device.queue.writeTexture(/* ... */);
    log('SPIKE1: copy-fallback');
  }

  // Spike 2: compute shader on worklet thread
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(sobelPipeline);
  computePass.dispatchWorkgroups(width / 16, height / 16);
  computePass.end();

  // Spike 3: Skia overlay (Graphite or Ganesh fallback)
  // Draw debug text + test shapes via Skia canvas

  // Spike 4: render pass outputs to preview canvas
  // + recorder Surface (if recording active)
  const renderPass = encoder.beginRenderPass(/* ... */);
  // composite computed frame + Skia overlay
  renderPass.end();
  device.queue.submit([encoder.finish()]);
};

runOnUI(renderLoop)(device);
```

**Metrics collection (Spike 5):** FPS and frame drop counts are tracked in
the worklet via `performance.now()` timestamps. Memory is read from JS via
`performance.memory` (if available) or a periodic native call. Thermal
state is polled from the Rust module's `get_thermal_state()` function.
After 60 seconds, the summary is logged to console.

---

## 5. Build & Run Workflow

### Local prebuild steps (once, then after Rust/Skia changes)

```bash
./scripts/build-rust.sh          # cross-compile Rust for all targets
./scripts/generate-bindings.sh   # UniFFI -> TypeScript + native bindings
SK_GRAPHITE=1 bun run build-skia # build Skia with Graphite backend
```

Precompiled Rust libraries are committed to the repo under
`packages/react-native-webgpu-camera/prebuilt/` so EAS Build workers don't
need the Rust toolchain.

### Device builds via EAS Build

```bash
cd apps/example
eas build --platform ios --profile development
eas build --platform android --profile development
```

EAS Build produces a development client installed on the physical device.
TypeScript/WGSL changes are then picked up via Metro fast refresh.

### `apps/example/eas.json`

```json
{
  "cli": { "version": ">= 12.0.0" },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": { "buildConfiguration": "Debug" },
      "android": { "buildType": "apk" },
      "env": { "SK_GRAPHITE": "1" }
    }
  }
}
```

### Development iteration

| Change type | Action |
|-------------|--------|
| TypeScript / WGSL | Metro fast refresh (instant) |
| Rust code | Rebuild locally, commit prebuilts, new EAS build |
| Skia build flag | Rebuild Skia locally, new EAS build |

---

## 6. Spike Success Criteria & Fallback Behavior

The app detects failures, falls back automatically, and reports. Console output
is the primary deliverable.

### Per-spike detection and fallback

| Spike | Attempt | Failure detection | Fallback | Console output |
|-------|---------|-------------------|----------|----------------|
| 1 (zero-copy) | Import buffer handle as `GPUTexture` | Exception or null texture | `get_current_frame_pixels()` + `writeTexture()` | `SPIKE1: zero-copy 0.02ms` or `SPIKE1: copy-fallback 0.31ms` |
| 2 (worklet compute) | `createComputePipeline()` + dispatch in worklet | Exception on create/dispatch | Compute on main JS thread | `SPIKE2: worklet-compute 1.2ms` or `SPIKE2: main-thread-compute 1.4ms` |
| 3 (Skia Graphite) | Skia detects Dawn, draws onto WebGPU texture | Graphite unavailable at runtime | `readPixels()` + `writeTexture()` + composite | `SPIKE3: graphite 0.0ms` or `SPIKE3: ganesh-fallback 0.6ms` |
| 4 (recorder Surface) | Render to recorder Surface handle | Invalid video file produced | GPU readback + feed pixels to encoder | `SPIKE4: surface-record` or `SPIKE4: readback-record` |

### 60-second summary (Spike 5 implicit)

After 60 seconds of sustained operation:

```
=== SPIKE RESULTS ===
Spike 1: copy-fallback (0.31ms avg)
Spike 2: worklet-compute (1.2ms avg)
Spike 3: graphite (0.0ms overhead)
Spike 4: surface-record
Sustained FPS: 58.3
Frame drops: 12/3600 (0.3%)
Memory: 142MB peak
Thermal: nominal -> fair at 45s
========================
```

Each line maps to a row in the post-spike decision matrix from the spike doc
and determines which architecture path the main plan commits to.

---

## 7. Out of Scope

These are explicitly not part of the spike scaffold:

- CI/CD setup (per user instruction)
- Benchmark app (`apps/benchmark/`)
- `@webgpu-camera/examples` package (ML inference examples)
- Full camera controls API (zoom, focus, exposure, torch)
- EXIF metadata extraction
- Photo capture
- Device enumeration
- Production error handling
- Tests (spike is validated on physical device, not in CI)
