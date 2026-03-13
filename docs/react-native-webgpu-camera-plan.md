# react-native-webgpu-camera

## Agent Implementation Plan

---

## 1. Project Overview

`react-native-webgpu-camera` is a standalone React Native camera library that replaces
VisionCamera + Skia Frame Processors with a unified WebGPU compute + Skia Graphite
pipeline. It delivers zero-copy camera frame processing, GPU compute shaders written
entirely in TypeScript/WGSL, Skia 2D overlay drawing that works in both preview AND
captured output, and hardware-accelerated photo/video encoding.

**The core proposition:** Everything VisionCamera does, but with WebGPU compute shaders
instead of native frame processor plugins, and with Skia overlays that actually appear
in photos and videos.

### 1.1 Why This Exists

VisionCamera v4 has three fundamental limitations:

1. **Skia overlays are preview-only.** Anything drawn via `useSkiaFrameProcessor` does
   not appear in captured photos or recorded video. Margelo offers custom solutions for
   this as paid consulting work.

2. **No GPU compute from JS.** Heavy per-pixel operations (motion detection, histograms,
   temporal processing) require native frame processor plugins written in Objective-C++
   or Kotlin — per platform, no code sharing.

3. **No temporal/stateful processing.** Skia RuntimeEffects are stateless fragment
   shaders. There's no way to persist data between frames (e.g., previous frame for
   motion detection, running average for background subtraction) without CPU-side buffers
   and GPU→CPU→GPU round-trips.

This library solves all three by owning the entire pipeline from camera to display/capture.

### 1.2 Package Structure

Monorepo with two packages. The core library is what users install. The examples
package demonstrates integration patterns (especially ML inference) that users
copy, adapt, or depend on optionally.

**Why two packages:** The core library provides primitives — camera, compute
pipeline, Skia overlay, capture, and pre-built WGSL shaders. These have zero
heavy external dependencies (the WGSL shaders are just strings). The ML examples
drag in heavy runtimes (TF.js ~2MB, ONNX Runtime ~5MB, model files ~10-50MB)
with their own release cycles and model version dependencies. When MediaPipe
updates their face detection model, that shouldn't be a patch release of the
camera library. ML hooks are integration patterns, not library code — someone
can write `useFaceDetection()` in an afternoon from our docs.

```
react-native-webgpu-camera/          (monorepo root)
├── packages/
│   ├── react-native-webgpu-camera/  (core library — what users npm install)
│   │   ├── /camera                  → Camera device management, frame delivery
│   │   ├── /compute                 → WebGPU compute pipeline orchestration
│   │   ├── /skia                    → Skia Graphite 2D overlay compositing
│   │   ├── /capture                 → Photo capture + video recording
│   │   ├── /processors              → Pre-built WGSL compute shaders
│   │   └── (root)                   → Re-exports primary hooks and components
│   │
│   └── @webgpu-camera/examples/     (reference implementations — optional)
│       ├── /face-detection          → MediaPipe Face Detection (TF.js WebGPU)
│       ├── /pose-estimation         → MediaPipe Pose (TF.js WebGPU)
│       ├── /object-detection        → EfficientDet / SSD MobileNet (TF.js WebGPU)
│       ├── /segmentation            → MediaPipe Selfie Segmentation (TF.js WebGPU)
│       ├── /text-recognition        → PaddleOCR (ONNX Runtime WebGPU)
│       ├── /barcode-scanning        → ZXing WASM or custom compute shader
│       ├── /image-classification    → MobileNet (TF.js WebGPU)
│       └── /custom-onnx-template    → Template for any ONNX model
│
└── apps/
    ├── example/                     (demo app showcasing both packages)
    └── benchmark/                   (performance comparison vs VisionCamera)
```

```typescript
// Core library — always installed
import { useCamera, GPUCamera } from 'react-native-webgpu-camera';
import { useComputePipeline } from 'react-native-webgpu-camera/compute';
import { useSkiaOverlay } from 'react-native-webgpu-camera/skia';
import { useCapture } from 'react-native-webgpu-camera/capture';
import { motionDetect, histogram } from 'react-native-webgpu-camera/processors';

// Examples package — optional, install if you want ready-made ML hooks
// Or copy the source and adapt it to your needs
import { useFaceDetection } from '@webgpu-camera/examples/face-detection';
import { usePoseEstimation } from '@webgpu-camera/examples/pose-estimation';
```

### 1.3 Key Dependencies

| Dependency | Role | Notes |
|------------|------|-------|
| `react-native-wgpu` | WebGPU API (Dawn) | Provides `GPUDevice`, compute, render |
| `@shopify/react-native-skia` | Skia 2D drawing | **Must build with `SK_GRAPHITE=1`** |
| `react-native-reanimated` | Worklet threading | WebGPU calls run on worklet threads |
| `uniffi-bindgen-react-native` | Rust → TS bindings | Camera module + video encoder |

### 1.4 Platform Requirements

| Platform | Minimum | Reason |
|----------|---------|--------|
| iOS | 13.0+ | Metal required by Dawn |
| Android | API 28+ (Android 9) | `HardwareBuffer` for zero-copy Vulkan import |
| React Native | 0.81+ | Required by react-native-wgpu (New Architecture only) |

---

## 2. Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTIRELY ON GPU — NO CPU PIXEL COPIES         │
│                                                                  │
│  Camera Sensor                                                   │
│    → Platform Camera API (AVCaptureSession / Camera2)            │
│    → Native frame buffer (CMSampleBuffer / HardwareBuffer)       │
│    → Zero-copy import to WebGPU texture via Dawn                 │
│       (IOSurface → MTLTexture on iOS)                            │
│       (HardwareBuffer → VkImage on Android)                      │
│                                                                  │
│    → Compute Pass 1..N  (WGSL shaders, TypeScript-orchestrated)  │
│       - Color grading, filters, effects                          │
│       - Motion detection, histograms (persistent storage buffers)│
│       - ML preprocessing / postprocessing                        │
│       - Custom user shaders                                      │
│                                                                  │
│    → Skia Graphite Overlay (shared Dawn GPU context)             │
│       - drawRect, drawText, drawPath, drawImage                  │
│       - Bounding boxes, labels, annotations                      │
│       - Same Skia API developers already know                    │
│                                                                  │
│    → Final Composite Render Pass (outputs to MULTIPLE surfaces)  │
│       ├→ Preview Surface:  WebGPU canvas (screen display)        │
│       ├→ Photo:    GPU readback on capture event only            │
│       └→ Recorder Surface: platform video recorder (zero-copy)   │
│                                                                  │
│  Recording Architecture (zero-copy, framework-managed A/V sync): │
│                                                                  │
│  iOS:     Render → IOSurface → AVAssetWriterInput                │
│           Audio:  AVCaptureAudioDataOutput (same session clock)  │
│           Muxing: AVAssetWriter (framework handles A/V sync)     │
│                                                                  │
│  Android: Render → MediaRecorder.getSurface()                    │
│           Audio:  MediaRecorder internal mic capture              │
│           Muxing: MediaRecorder (framework handles A/V sync)     │
│                                                                  │
│  No manual AudioRecord. No manual MediaCodec. No manual          │
│  MediaMuxer. No manual timestamp alignment. The platform         │
│  recording APIs handle audio capture, encoding, muxing, and      │
│  A/V sync internally — we just render processed frames to their  │
│  Surface, same as we render to the preview canvas.               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Critical: CPU never touches pixel data during normal operation.**
The only CPU involvement is:
- Control plane: start/stop camera, configure parameters, trigger capture
- Small result buffer readback: motion counts, histogram bins (~bytes to KB)
- Photo capture: single-frame GPU readback to encode JPEG/PNG (on-demand only)

### 2.2 Threading Model

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  JS Thread   │    │  Camera Thread    │    │  Worklet Thread  │
│              │    │  (Rust native)    │    │  (Reanimated)    │
│  React UI    │    │                   │    │                  │
│  State mgmt  │    │  Frame delivery   │    │  WebGPU dispatch │
│  User input  │────│  Zero-copy import │────│  Compute shaders │
│              │    │  Encoder feeding  │    │  Skia overlay    │
│              │    │                   │    │  Preview render  │
└──────────────┘    └───────────────────┘    └──────────────────┘
```

We control the threading model end-to-end. No VisionCamera worklet runtime
dependency. All GPU work (compute + Skia + render) runs on a single Reanimated
worklet thread, avoiding cross-thread synchronization.

### 2.3 Skia Graphite Integration

**CRITICAL BUILD REQUIREMENT:** Skia Graphite is NOT the default backend.
Users (and our setup docs) must build Skia with the Graphite flag:

```bash
SK_GRAPHITE=1 yarn build-skia
```

Graphite requires Android API 26+ (our floor is API 28, so this is satisfied).
React Native Skia auto-detects Dawn library files when Graphite is enabled.

**Why Graphite matters:** Graphite renders via Dawn (the same WebGPU implementation
our compute pipeline uses). This means Skia and WebGPU share the same GPU context,
command queue, and texture memory. A Skia canvas can draw directly onto a WebGPU
texture — zero-copy compositing of 2D overlays on computed frames.

**Fallback when Graphite is unavailable:** If the user hasn't built with
`SK_GRAPHITE=1`, detect this at runtime and fall back to:
1. Render Skia overlay to an offscreen Ganesh surface (Metal/OpenGL)
2. `readPixels()` to CPU (~0.3ms for a typical overlay layer)
3. `device.queue.writeTexture()` back to a WebGPU texture (~0.3ms)
4. Composite in the final render pass (identical to the Graphite path)

The library should warn loudly at init if Graphite is not detected and
recommend rebuilding Skia.

**IMPORTANT: The fallback still supports overlay in captured output.**
The compositing render pass is identical regardless of how the Skia overlay
texture arrived — it samples two textures (processed frame + overlay) and
alpha-composites them. The recorder Surface and photo readback both receive
the full composite. The Ganesh fallback does NOT degrade to preview-only
like VisionCamera's Skia integration — it just costs ~0.6ms extra per frame.

**Capability matrix:**

| Feature | With Graphite | Without Graphite (Ganesh fallback) |
|---------|--------------|-----------------------------------|
| Compute shaders in preview | ✅ | ✅ |
| Compute shaders in photos | ✅ | ✅ |
| Compute shaders in video | ✅ | ✅ |
| Skia overlay in preview | ✅ zero-copy | ✅ +0.6ms round-trip |
| Skia overlay in photos | ✅ zero-copy | ✅ +0.6ms round-trip |
| Skia overlay in video | ✅ zero-copy | ✅ +0.6ms round-trip |
| Skia ↔ WebGPU bidirectional | ✅ (Skia can draw onto compute textures) | ❌ (separate layers, composited at end) |
| 60fps achievable | ✅ easily (~20ms headroom) | ✅ (~14ms headroom at 1080p) |

The "bidirectional" row is the only real feature loss without Graphite. With
Graphite, a Skia path could use a compute shader output as a fill pattern, or
a compute shader could read Skia's rendered text as an input texture. Without
Graphite, they're independent layers that get alpha-composited at the end but
can't interact mid-pipeline. For the vast majority of use cases (bounding boxes,
labels, annotations drawn over a processed frame), this doesn't matter.

**Optimization for Ganesh fallback:** The overlay layer is mostly transparent
pixels with sparse drawing (a few boxes and text labels). The round-trip cost
can be reduced by rendering Skia at lower resolution (720p overlay on 1080p
frame — visually identical, 4× fewer pixels to copy) or by only reading back
the dirty region.

---

## 3. Native Module (Rust + UniFFI)

### 3.1 Toolchain

The native code is written in Rust and exposed to TypeScript via
`uniffi-bindgen-react-native` (Mozilla/Filament, released Dec 2024).

UniFFI generates TypeScript + JSI C++ from Rust proc macro annotations.
You annotate your Rust API once and get type-safe bindings for iOS, Android,
and WASM — no hand-written Objective-C, Kotlin, or C++ FFI glue.

```
rust/
├── Cargo.toml
├── src/
│   ├── lib.rs                  ← UniFFI annotated public API
│   ├── camera/
│   │   ├── mod.rs              ← Platform-agnostic camera trait
│   │   ├── ios.rs              ← AVCaptureSession wrapper (~150 lines)
│   │   └── android.rs          ← Camera2 wrapper (~250 lines)
│   ├── gpu_bridge/
│   │   ├── mod.rs              ← Zero-copy texture import
│   │   ├── ios.rs              ← IOSurface → Dawn MTLTexture
│   │   └── android.rs          ← HardwareBuffer → Dawn VkImage
│   └── encoder/
│       ├── mod.rs              ← Encoding trait
│       ├── ios.rs              ← AVAssetWriter for video, CGImage for photo
│       └── android.rs          ← MediaCodec for video, Bitmap for photo
└── uniffi.toml                 ← UniFFI config for react-native bindings
```

### 3.2 UniFFI API Surface

```rust
// lib.rs — the entire public API exposed to TypeScript

#[uniffi::export]
pub fn create_camera_session(config: CameraConfig) -> Arc<CameraSession>;

#[uniffi::export]
impl CameraSession {
    /// Start streaming frames. Calls on_frame with a GPU texture handle.
    pub fn start(&self);
    pub fn stop(&self);

    /// Camera controls — thin wrappers around single platform API calls
    pub fn set_zoom(&self, factor: f32);
    pub fn set_focus_point(&self, x: f32, y: f32);
    pub fn set_exposure_compensation(&self, ev: f32);
    pub fn set_torch(&self, enabled: bool);

    /// Returns the GPU texture handle for the current frame.
    /// TypeScript passes this to WebGPU as an external texture.
    pub fn get_current_frame_texture(&self) -> Option<GpuTextureHandle>;

    /// Returns the recorder Surface handle that Dawn should render to
    /// during active recording. Null when not recording.
    pub fn get_recorder_surface(&self) -> Option<PlatformSurfaceHandle>;

    /// Capture photo from the current processed GPU texture.
    /// Injects cached EXIF metadata from the most recent camera frame
    /// and optionally embeds GPS coordinates.
    pub fn capture_photo(&self, texture: GpuTextureHandle, config: PhotoConfig) -> PhotoResult;

    /// Start/stop video recording.
    /// On Android: configures MediaRecorder, returns its Surface for Dawn to render to.
    /// On iOS: configures AVAssetWriter, returns IOSurface-backed pixel buffer adaptor.
    /// Audio capture and A/V sync are handled internally by the platform framework.
    pub fn start_recording(&self, config: RecordingConfig);
    pub fn stop_recording(&self) -> VideoResult;

    /// Enable/disable GPS location tracking for photo EXIF and video metadata.
    /// Requires location permission granted by the app.
    pub fn set_location_tracking(&self, enabled: bool);

    /// Device enumeration
    pub fn available_devices() -> Vec<CameraDeviceInfo>;
}

#[derive(uniffi::Record)]
pub struct CameraConfig {
    pub device_id: String,       // "back", "front", or specific ID
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub pixel_format: PixelFormat,
}

#[derive(uniffi::Record)]
pub struct PhotoConfig {
    pub quality: f32,              // JPEG quality 0.0-1.0
    pub format: PhotoFormat,       // JPEG or PNG
    pub embed_exif: bool,          // Include camera metadata (exposure, ISO, etc.)
    pub embed_location: bool,      // Include GPS (requires location permission)
}

#[derive(uniffi::Enum)]
pub enum PhotoFormat {
    Jpeg,
    Png,
}

#[derive(uniffi::Record)]
pub struct PhotoResult {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub exif: Option<ExifData>,    // Returned so JS can display/use metadata
}

#[derive(uniffi::Record)]
pub struct ExifData {
    pub exposure_time: Option<f64>,      // seconds
    pub iso: Option<u32>,
    pub focal_length: Option<f32>,       // mm
    pub aperture: Option<f32>,           // f-number
    pub white_balance: Option<String>,
    pub lens_model: Option<String>,
    pub device_model: String,
    pub orientation: u16,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude: Option<f64>,
    pub timestamp: String,               // ISO 8601
}

#[derive(uniffi::Record)]
pub struct RecordingConfig {
    pub output_path: String,
    pub video_bitrate: u32,        // bps
    pub audio_enabled: bool,
    pub audio_source: AudioSource,
    pub embed_location: bool,      // GPS in video container metadata
}

#[derive(uniffi::Enum)]
pub enum AudioSource {
    Microphone,
    None,
}

#[derive(uniffi::Enum)]
pub enum PixelFormat {
    Yuv420,
    Bgra8,
}
```

### 3.3 EXIF Metadata & GPS

**The problem:** In VisionCamera, the platform camera API embeds EXIF
automatically because it captures directly from the camera pipeline. In our
architecture, photo capture is a GPU readback of processed pixels — raw RGBA
data with zero metadata. We lose everything: exposure, ISO, focal length,
white balance, lens info, device model, orientation, GPS.

**The solution:** The platform camera APIs provide all this metadata alongside
each frame — it's just on a separate path from the pixel data.

**iOS:** `CMSampleBuffer` has `CMGetAttachment` for EXIF metadata.
`AVCaptureDevice` exposes current exposure/ISO/focus state. GPS from
`CLLocationManager`.

**Android:** Camera2's `CaptureResult` (delivered with each frame) contains
all EXIF-equivalent fields: `SENSOR_EXPOSURE_TIME`, `SENSOR_SENSITIVITY`,
`LENS_FOCAL_LENGTH`, `LENS_APERTURE`, etc. GPS from `LocationManager`.

**Implementation in the Rust module:**

1. **Per-frame metadata caching:** Each frame delivery callback extracts
   metadata from `CaptureResult` (Android) or `CMSampleBuffer` attachments
   (iOS) and caches it in the `CameraSession` struct. This is lightweight
   structured data — no pixel copies.

2. **Optional GPS subscription:** When `set_location_tracking(true)` is called,
   the module subscribes to platform location updates (`CLLocationManager` /
   `LocationManager`). Latest location is cached and updated asynchronously.
   Requires the app to have location permission — the module does not request
   permission itself, only consumes it.

3. **Photo EXIF injection:** When `capture_photo()` is called, the cached
   metadata + GPS is injected into the JPEG/PNG output. For JPEG, EXIF
   injection uses a Rust crate like `kamadak-exif` or `img-parts` to write
   EXIF segments without re-encoding the image data.

4. **Video metadata:** `MediaRecorder` (Android) and `AVAssetWriter` (iOS)
   handle video container metadata natively — creation date, device info, GPS.
   We pass the location to the recorder config when `embed_location` is true.

5. **ExifData returned to JS:** The `PhotoResult` includes the `ExifData`
   struct so the JS side can display metadata, store it in a database, or
   use it for organizing photos. This is the same data that was injected
   into the file — no separate query needed.

### 3.4 Zero-Copy Frame Import (the critical path)

**iOS (~120 lines):**
```
CMSampleBuffer
  → CMSampleBufferGetImageBuffer() → CVPixelBuffer
  → CVPixelBufferGetIOSurface() → IOSurface
  → Dawn imports IOSurface as MTLTexture (shared GPU memory)
  → Returns WGPUTexture handle to JS
```

**Android (~150 lines):**
```
ImageReader.acquireLatestImage()
  → Image.getHardwareBuffer() → HardwareBuffer
  → Dawn imports via VK_ANDROID_external_memory_android_hardware_buffer
  → Returns WGPUTexture handle to JS
```

The texture handle is passed to the JS worklet thread where it's used as a
`GPUExternalTexture` or sampled in compute/render passes. The pixel data
never touches CPU memory.

### 3.5 Video Recording (platform-native, zero-copy, framework-managed A/V sync)

Unlike VisionCamera (which uses manual AudioRecord + MediaCodec + MediaMuxer on
Android with hand-managed timestamp alignment), we use each platform's high-level
recording API and just render our processed frames to their input Surface.

**Android: MediaRecorder with Surface input (~80 lines Rust)**

The elegant path that Android's built-in camera app uses:

```kotlin
// What our Rust module does (via JNI):
val mediaRecorder = MediaRecorder().apply {
    setAudioSource(MediaRecorder.AudioSource.MIC)
    setVideoSource(MediaRecorder.SURFACE)  // Surface input, not Camera
    setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
    setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
    setVideoEncoder(MediaRecorder.VideoEncoder.H264)
    setVideoSize(width, height)
    setVideoFrameRate(fps)
    setOutputFile(outputPath)
    prepare()
}

// MediaRecorder provides a Surface — Dawn renders to it
val recorderSurface: Surface = mediaRecorder.surface
// Our final render pass outputs to BOTH preview canvas AND this surface
```

MediaRecorder handles audio capture, video encoding, muxing, and A/V sync
internally. We never touch `AudioRecord`, `MediaCodec`, `MediaMuxer`, or
timestamps. We just render processed frames to the Surface at our frame rate.

**iOS: AVAssetWriter with AVCaptureAudioDataOutput (~100 lines Rust)**

```swift
// Camera session captures audio from the same session (shared clock)
let audioOutput = AVCaptureAudioDataOutput()
captureSession.addOutput(audioOutput)

// Asset writer with both video and audio inputs
let assetWriter = AVAssetWriter(url: outputURL, fileType: .mp4)
let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: videoInput,
    sourcePixelBufferAttributes: nil
)
assetWriter.add(videoInput)
assetWriter.add(audioInput)

// Render processed frame → IOSurface → adaptor.append(pixelBuffer, timestamp)
// Audio samples from AVCaptureAudioDataOutput → audioInput.append(sampleBuffer)
// Timestamps share the same session clock — A/V sync is automatic
```

**Recording architecture comparison:**

| Aspect | VisionCamera (Android) | Our approach (Android) |
|--------|----------------------|----------------------|
| Video encoding | Manual MediaCodec setup | MediaRecorder (internal) |
| Audio capture | Manual AudioRecord | MediaRecorder (internal) |
| Muxing | Manual MediaMuxer | MediaRecorder (internal) |
| A/V sync | Manual timestamp alignment (error-prone) | Framework-managed (automatic) |
| Native code | ~400+ lines Kotlin | ~80 lines Rust |
| Processes recorded | Raw camera frames only | Fully processed + Skia overlays |

**Photo capture:** Single-frame GPU readback via `readbackBuffer.mapAsync()`.
Encode to JPEG/PNG using platform APIs. Only happens on user-triggered capture.
No audio involvement.

---

## 4. TypeScript API Design

### 4.1 Drop-in Replacement API

The API mirrors VisionCamera's patterns as closely as possible to minimize
migration effort. Developers should be able to port existing code with
predictable, mechanical changes.

#### Current VisionCamera + Skia:

```typescript
import { Camera, useCameraDevice, useSkiaFrameProcessor } from 'react-native-vision-camera';
import { Skia, PaintStyle } from '@shopify/react-native-skia';

function CameraScreen() {
  const device = useCameraDevice('back');

  const frameProcessor = useSkiaFrameProcessor((frame) => {
    'worklet';
    frame.render(); // render camera frame
    
    const faces = detectFaces(frame); // native plugin call
    for (const face of faces) {
      const paint = Skia.Paint();
      paint.setColor(Skia.Color('red'));
      paint.setStyle(PaintStyle.Stroke);
      paint.setStrokeWidth(3);
      frame.drawRect(face.bounds, paint);
    }
  }, []);

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={true}
      frameProcessor={frameProcessor}
    />
  );
}
```

#### Migration to react-native-webgpu-camera:

```typescript
import { GPUCamera, useCameraDevice, useGPUFrameProcessor }
  from 'react-native-webgpu-camera';
import { useSkiaOverlay } from 'react-native-webgpu-camera/skia';
import { useCapture } from 'react-native-webgpu-camera/capture';
import { useFaceDetection } from '@webgpu-camera/examples/face-detection';
import { Skia, PaintStyle } from '@shopify/react-native-skia';

function CameraScreen() {
  const device = useCameraDevice('back');
  const capture = useCapture();
  const skia = useSkiaOverlay();

  const frameProcessor = useGPUFrameProcessor(({ frame, gpu }) => {
    'worklet';

    // Compute passes (optional — skip if you only need Skia drawing)
    gpu.dispatch(colorGradeShader, { temperature: 6500 });

    // Skia 2D drawing — SAME API as VisionCamera's Skia frame processor
    const canvas = skia.getCanvas();
    const faces = detectFaces(frame); // WebGPU ML inference, not native plugin
    for (const face of faces) {
      const paint = Skia.Paint();
      paint.setColor(Skia.Color('red'));
      paint.setStyle(PaintStyle.Stroke);
      paint.setStrokeWidth(3);
      canvas.drawRect(face.bounds, paint);
    }

    // Present composites compute + Skia → preview AND capture pipeline
    gpu.present();
  }, []);

  return (
    <View style={{ flex: 1 }}>
      <GPUCamera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
      />
      <CaptureButton
        onPhoto={() => capture.takePhoto()}
        onVideoStart={() => capture.startRecording()}
        onVideoStop={() => capture.stopRecording()}
      />
    </View>
  );
}
```

**Key differences:**
1. `<Camera>` → `<GPUCamera>` (our component, not VisionCamera)
2. `useSkiaFrameProcessor` → `useGPUFrameProcessor` (receives `{ frame, gpu }`)
3. `frame.render()` is implicit (GPU texture is always available)
4. `frame.drawRect()` → `canvas.drawRect()` (via `skia.getCanvas()`)
5. Native `detectFaces()` plugin → WebGPU ML `detectFaces()` from examples package
6. Skia drawings appear in photos/video (the whole point)

### 4.2 Compute Pipeline API

```typescript
import { createComputeShader } from 'react-native-webgpu-camera/compute';

// Define a compute shader — just a WGSL string
const motionDetect = createComputeShader({
  name: 'motion_detect',
  wgsl: `
    @group(0) @binding(0) var current_frame: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> prev_frame: array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> result: atomic<u32>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3u) {
        let idx = id.y * uniforms.width + id.x;
        let current = textureLoad(current_frame, id.xy, 0);
        let previous = prev_frame[idx];
        let diff = length(current.rgb - previous.rgb);
        if (diff > uniforms.threshold) {
            atomicAdd(&result, 1u);
        }
        prev_frame[idx] = current;
    }
  `,
  // Persistent buffers survive across frames (temporal state!)
  persistentBuffers: {
    prev_frame: { type: 'storage', size: 'frame' }, // auto-sized to frame dims
  },
  // Result buffers are read back to CPU each frame
  resultBuffers: {
    result: { type: 'atomic<u32>', size: 4 },
  },
  // Uniforms configurable from JS
  uniforms: {
    threshold: { type: 'f32', default: 0.15 },
  },
});

// Use in frame processor
const frameProcessor = useGPUFrameProcessor(({ frame, gpu }) => {
  'worklet';
  const { result } = gpu.dispatch(motionDetect, { threshold: 0.2 });
  // result.motion_pixel_count is available immediately (previous frame's result)
  // or after await (current frame's result, slight latency)
}, []);
```

### 4.3 Pre-built Compute Shaders (processors subpath)

```typescript
import {
  motionDetect,       // Frame differencing with persistent previous frame
  colorHistogram,     // 256-bin RGB histogram via atomic workgroup reduction
  edgeSobel,          // Sobel edge detection (3x3 convolution)
  gaussianBlur,       // Separable Gaussian blur (two-pass)
  colorGrade,         // Temperature, tint, exposure, contrast, saturation
  yuvToRgb,           // YUV420 → RGBA conversion
  threshold,          // Binary threshold with configurable cutoff
  backgroundSubtract, // Running average background model (persistent)
  opticalFlow,        // Lucas-Kanade sparse optical flow
  segmentationRefine, // Guided filter upscaling of low-res segmentation mask
} from 'react-native-webgpu-camera/processors';
```

Each of these is a `ComputeShader` object containing the WGSL source,
buffer declarations, and uniform definitions. They're just TypeScript
objects — tree-shakeable, inspectable, modifiable.

### 4.4 ML Inference (via `@webgpu-camera/examples` package)

ML inference is NOT part of the core library. It lives in the separate
`@webgpu-camera/examples` package in the monorepo. All ML inference uses
WebGPU compute. No native frame processor plugins.

```typescript
// Install the examples package alongside the core library
// npm install @webgpu-camera/examples

import { useFaceDetection } from '@webgpu-camera/examples/face-detection';
import { usePoseEstimation } from '@webgpu-camera/examples/pose-estimation';
import { useObjectDetection } from '@webgpu-camera/examples/object-detection';
import { useTextRecognition } from '@webgpu-camera/examples/text-recognition';
import { useBarcodeScanner } from '@webgpu-camera/examples/barcode-scanning';
import { useImageClassification } from '@webgpu-camera/examples/image-classification';
import { useSegmentation } from '@webgpu-camera/examples/segmentation';

// Example: face detection
const faces = useFaceDetection({ maxFaces: 5, minConfidence: 0.8 });

const frameProcessor = useGPUFrameProcessor(({ frame, gpu }) => {
  'worklet';
  
  const detectedFaces = faces.detect(frame);
  // detectedFaces: Array<{ bounds: Rect, landmarks: Point[], confidence: number }>
  
  const canvas = skia.getCanvas();
  for (const face of detectedFaces) {
    canvas.drawRect(face.bounds, redPaint);
    for (const point of face.landmarks) {
      canvas.drawCircle(point.x, point.y, 3, greenPaint);
    }
  }
  
  gpu.present();
}, []);
```

**These are reference implementations, not library code.** Users can:
1. Install `@webgpu-camera/examples` and use the hooks directly
2. Copy the source into their own project and adapt it
3. Use them as templates to integrate other ML models

**Mapping of VisionCamera native plugins to WebGPU examples:**

| Use Case | VisionCamera Native Plugin | WebGPU Example |
|----------|---------------------------|----------------------|
| Face detection | MLKit (native) | MediaPipe Face Detection (TF.js WebGPU) |
| Pose estimation | MLKit Pose (native) | MediaPipe Pose (TF.js WebGPU) |
| Object detection | TFLite (native C++) | EfficientDet (TF.js WebGPU) |
| Text recognition (OCR) | MLKit Text (native) | PaddleOCR (ONNX WebGPU) |
| Barcode scanning | MLKit Barcode (native) | ZXing-wasm or custom compute shader |
| Image classification | CoreML/TFLite (native) | MobileNet (TF.js WebGPU) |
| Segmentation | MLKit Selfie (native) | MediaPipe Selfie Segmentation (TF.js WebGPU) |
| Custom ONNX model | N/A | ONNX Runtime Web (WebGPU execution provider) |

The preprocessing (resize, normalize, color convert) and postprocessing
(decode boxes, NMS, render overlays) are ALL compute shaders from the core
library's `/processors` subpath — no platform-specific code for any of it.

**Do we still need native frame processor plugin support?**

**No.** Every common native frame processor use case has a WebGPU-based
replacement in the examples package. The native module (Rust/UniFFI) handles
ONLY camera management and video recording — never frame processing or ML
inference. This is a deliberate architectural boundary: pixel processing
happens exclusively on the GPU via TypeScript/WGSL.

If a developer has an edge case requiring a native-only API with no WebGPU
equivalent (e.g., a proprietary SDK that only exposes a native API), they
can write a separate Expo Module that receives raw pixel data. But this is
outside our library's scope — we don't provide a native plugin system.

---

## 5. VisionCamera Frame Processor → WebGPU Migration Guide

This section maps every common VisionCamera frame processor pattern to its
`react-native-webgpu-camera` equivalent.

### 5.1 Basic Skia Drawing (bounding boxes, text, shapes)

**Before (preview-only):**
```typescript
const frameProcessor = useSkiaFrameProcessor((frame) => {
  'worklet';
  frame.render();
  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  frame.drawRect({ x: 10, y: 10, width: 100, height: 100 }, paint);
  frame.drawCircle(50, 50, 20, paint);
}, []);
```

**After (works in preview + capture):**
```typescript
const frameProcessor = useGPUFrameProcessor(({ gpu }) => {
  'worklet';
  const canvas = skia.getCanvas();
  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  canvas.drawRect({ x: 10, y: 10, width: 100, height: 100 }, paint);
  canvas.drawCircle(50, 50, 20, paint);
  gpu.present();
}, []);
```

**Change:** `frame.render()` is gone (implicit). `frame.drawX()` → `canvas.drawX()`.
The Skia canvas API is identical — same `Paint`, same `Path`, same `drawRect/Circle/Text`.

### 5.2 RuntimeEffect Shader (color filter)

**Before (Skia SKSL fragment shader):**
```typescript
const invertShader = Skia.RuntimeEffect.Make(`
  uniform shader image;
  half4 main(vec2 pos) {
    vec4 color = image.eval(pos);
    return vec4((1.0 - color).rgb, 1.0);
  }
`);

const frameProcessor = useSkiaFrameProcessor((frame) => {
  'worklet';
  const paint = Skia.Paint();
  paint.setImageFilter(Skia.ImageFilter.MakeRuntimeShader(shaderBuilder, null, null));
  frame.render(paint);
}, []);
```

**After (WGSL compute shader):**
```typescript
const invertShader = createComputeShader({
  name: 'invert_colors',
  wgsl: `
    @group(0) @binding(0) var input: texture_2d<f32>;
    @group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      let color = textureLoad(input, id.xy, 0);
      textureStore(output, id.xy, vec4f(1.0 - color.rgb, 1.0));
    }
  `,
});

const frameProcessor = useGPUFrameProcessor(({ gpu }) => {
  'worklet';
  gpu.dispatch(invertShader);
  gpu.present();
}, []);
```

**Change:** SKSL → WGSL syntax. Fragment shader (per-pixel output only) → compute
shader (can write to ANY pixel, use storage buffers, do reductions). More powerful.

### 5.3 Face Detection + Overlay

**Before (native MLKit plugin):**
```typescript
import { scanFaces } from 'vision-camera-face-detector';

const frameProcessor = useSkiaFrameProcessor((frame) => {
  'worklet';
  frame.render();
  const faces = scanFaces(frame); // native Obj-C/Kotlin plugin
  for (const face of faces) {
    frame.drawRect(face.bounds, redPaint);
  }
}, []);
```

**After (WebGPU ML inference):**
```typescript
import { useFaceDetection } from '@webgpu-camera/examples/face-detection';

const faces = useFaceDetection({ maxFaces: 5 });

const frameProcessor = useGPUFrameProcessor(({ frame, gpu }) => {
  'worklet';
  const detected = faces.detect(frame); // TF.js WebGPU inference
  const canvas = skia.getCanvas();
  for (const face of detected) {
    canvas.drawRect(face.bounds, redPaint);
  }
  gpu.present();
}, []);
```

**Change:** `scanFaces(frame)` (native) → `faces.detect(frame)` (WebGPU).
Same result type. Drawing code identical. No native plugin dependency.

### 5.4 Frame Resize + ML Preprocessing

**Before (native resize plugin + TFLite):**
```typescript
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTensorflowModel } from 'react-native-fast-tflite';

const { resize } = useResizePlugin();
const model = useTensorflowModel(require('./model.tflite'));

const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  const resized = resize(frame, {
    scale: { width: 320, height: 320 },
    pixelFormat: 'rgb',
    dataType: 'uint8',
  });
  const result = model.runSync([resized]);
}, []);
```

**After (all compute shaders):**
```typescript
import { yuvToRgb } from 'react-native-webgpu-camera/processors';
import { useOnnxModel } from '@webgpu-camera/examples/custom-onnx-template';

const model = useOnnxModel(require('./model.onnx'));

const frameProcessor = useGPUFrameProcessor(({ frame, gpu }) => {
  'worklet';
  // YUV→RGB + resize + normalize in chained compute passes
  // Input and output stay on GPU — no CPU copy
  gpu.dispatch(yuvToRgb);
  gpu.dispatch(resizeShader, { width: 320, height: 320 });
  gpu.dispatch(normalizeShader, { mean: [0.485, 0.456, 0.406] });

  // Model runs on WebGPU — preprocessed tensor already on GPU
  const result = model.run(gpu.getOutputTexture());
  gpu.present();
}, []);
```

**Change:** Two native dependencies (resize plugin + TFLite) → zero native
dependencies. Entire preprocessing pipeline is chained compute shaders.
Data never leaves GPU between preprocessing and inference.

### 5.5 Temporal Processing (motion detection)

**Before: NOT POSSIBLE in VisionCamera without native code.**
Skia RuntimeEffects are stateless. Previous frame data requires a native plugin
that manages GPU buffers per-platform.

**After:**
```typescript
import { motionDetect } from 'react-native-webgpu-camera/processors';

const frameProcessor = useGPUFrameProcessor(({ gpu }) => {
  'worklet';
  const { motionPixelCount } = gpu.dispatch(motionDetect, { threshold: 0.15 });
  // motionPixelCount is available from the persistent prev_frame buffer
  // that the compute shader maintains across frames automatically

  if (motionPixelCount > 1000) {
    // Motion detected — trigger recording, alert, etc.
  }
  gpu.present();
}, []);
```

This is the clearest win. Temporal state via persistent storage buffers is
a capability that simply doesn't exist in VisionCamera's architecture.

---

## 6. Agent Task Breakdown

### 6.1 Agent 1: Rust Native Module (Camera + GPU Bridge)

**Scope:** Camera session management, zero-copy frame import, photo/video encoding.

**Tech:** Rust, UniFFI (`uniffi-bindgen-react-native`), platform camera APIs.

**Tasks:**

1. **Scaffold Rust project with UniFFI for React Native**
   - Set up `Cargo.toml` with `uniffi` dependency
   - Configure `uniffi.toml` for React Native bindings generation
   - Set up cross-compilation targets (iOS aarch64, Android aarch64/armv7/x86_64)
   - Verify TypeScript bindings generate and compile

2. **iOS camera module** (~150 lines Rust + ObjC interop)
   - `AVCaptureSession` + `AVCaptureVideoDataOutput` setup
   - Device enumeration (`AVCaptureDevice.DiscoverySession`)
   - Frame delivery callback → extract `CVPixelBuffer` → `IOSurface`
   - Camera controls: zoom, focus, exposure, torch (single API calls each)
   - Error handling: retry on `CameraAccessException`, cleanup on disconnect

3. **Android camera module** (~250 lines Rust + JNI)
   - Camera2 `CameraManager` + `ImageReader` setup
   - Device enumeration (`getCameraIdList()`)
   - Frame delivery → `Image.getHardwareBuffer()`
   - Camera controls
   - Error handling: state machine for Camera2 lifecycle, retry on open failure

4. **Zero-copy GPU bridge — iOS** (~120 lines)
   - `IOSurface` → Dawn `MTLTexture` import
   - Texture handle creation and caching
   - Buffer pooling (reuse texture objects across frames)

5. **Zero-copy GPU bridge — Android** (~150 lines)
   - `HardwareBuffer` → Dawn `VkImage` via
     `VK_ANDROID_external_memory_android_hardware_buffer`
   - Same handle/pooling pattern as iOS

6. **Photo capture + EXIF/GPS metadata** (both platforms)
   - GPU readback: map processed texture → pixel buffer → platform image encode
   - JPEG quality config, PNG support
   - **Per-frame metadata caching:** extract EXIF-equivalent data from
     `CaptureResult` (Android) / `CMSampleBuffer` attachments (iOS) on each
     frame delivery callback. Cache in `CameraSession` struct. Lightweight
     structured data — no pixel involvement.
   - **GPS subscription:** optional `set_location_tracking(true)` subscribes to
     `CLLocationManager` (iOS) / `LocationManager` (Android). Latest coordinates
     cached asynchronously. Module does not request permission — app must handle.
   - **EXIF injection into JPEG:** use Rust crate (`kamadak-exif` or `img-parts`)
     to write EXIF segments into encoded JPEG without re-encoding image data.
     Includes: exposure time, ISO, focal length, aperture, white balance, lens
     model, device model, orientation, GPS coordinates, timestamp.
   - **ExifData returned to JS:** `PhotoResult` includes the injected metadata
     so the JS side can display it without re-reading the file.

7. **Video recording** (both platforms — elegantly simple)
   - iOS: `AVAssetWriter` + `AVAssetWriterInputPixelBufferAdaptor` (~100 lines)
     Dawn renders processed frames → IOSurface → adaptor.append()
     Audio: `AVCaptureAudioDataOutput` on same session (shared clock = auto sync)
   - Android: `MediaRecorder` with Surface input (~80 lines)
     Configure MediaRecorder → get its Surface → Dawn renders to that Surface
     Audio: MediaRecorder captures mic internally (framework A/V sync)
   - `start_recording()` returns platform Surface handle to JS
   - `stop_recording()` finalizes file, returns path + metadata
   - GPS in video container metadata when `embed_location` is true
     (both MediaRecorder and AVAssetWriter handle this natively)
   - No manual AudioRecord, MediaCodec, MediaMuxer, or timestamp alignment

8. **Per-frame metrics instrumentation**
   - Inject high-resolution timestamps at each native pipeline stage:
     `camera_timestamp`, `native_receive_ns`, `gpu_import_ns`
   - Expose `FrameMetrics` struct via UniFFI so JS can read timing data
   - Optional: compile-time flag to enable/disable metrics (zero overhead
     in production builds)

**Deliverables:** Rust crate with UniFFI annotations, generated TypeScript
bindings, working on iOS simulator + Android emulator. E2E test: open camera,
stream frames, capture photo, record 5 seconds of video with audio.

### 6.2 Agent 2: WebGPU Compute Pipeline

**Scope:** Compute shader orchestration, pipeline management, buffer lifecycle.

**Tech:** TypeScript, WGSL, `react-native-wgpu` API.

**Tasks:**

1. **Core pipeline manager**
   - `GPUDevice` initialization and caching
   - Shader module compilation from WGSL strings
   - Compute pipeline creation and caching
   - Bind group management (frame texture + persistent buffers + uniforms)
   - Command encoder / dispatch / submit lifecycle

2. **`createComputeShader()` factory**
   - Parse shader declaration (WGSL, buffer defs, uniforms)
   - Auto-create persistent storage buffers (sized to frame dimensions)
   - Auto-create result readback buffers
   - Uniform buffer management with typed updates from JS

3. **`useGPUFrameProcessor()` hook**
   - Receive frame texture handle from Rust native module
   - Create bind group with current frame + persistent buffers
   - `gpu.dispatch()` — execute a named shader with optional uniform overrides
   - `gpu.present()` — trigger final render pass to BOTH preview canvas AND
     recorder Surface (if recording is active). Single command buffer, two targets.
   - Double-buffered result readback (return previous frame's results immediately)

4. **Frame texture management**
   - Receive `GpuTextureHandle` from Rust module per frame
   - Create/update `GPUExternalTexture` or `GPUTexture` from handle
   - Intermediate textures for multi-pass pipelines (ping-pong buffers)
   - Cleanup / release per frame

5. **Pre-built compute shaders** (WGSL implementations)
   - `yuvToRgb` — planar YUV420 → RGBA conversion
   - `motionDetect` — frame differencing with persistent prev_frame buffer
   - `colorHistogram` — 256-bin RGB histogram via atomic reduction
   - `edgeSobel` — 3×3 Sobel convolution
   - `gaussianBlur` — separable two-pass Gaussian
   - `colorGrade` — temperature, tint, exposure, contrast, saturation
   - `threshold` — binary threshold
   - `backgroundSubtract` — running average model (persistent)

6. **GPU timestamp query infrastructure**
   - Create `GPUQuerySet` with timestamp type for each pipeline
   - Wrap compute dispatches with `writeTimestamp()` before and after
   - Resolve query results to a readback buffer
   - Expose per-pass GPU execution time (nanoseconds) in `FrameMetrics`
   - Optional: compile-time flag to disable queries in production
     (timestamp queries have minimal but nonzero overhead)

**Deliverables:** TypeScript library that chains compute shaders on a WebGPU
device, running at 60fps on a worklet thread. Unit tests for each shader
(render test frames, verify output). Working in react-native-wgpu example app
(initially with static image input, not live camera — that requires Agent 1).

### 6.3 Agent 3: Skia Graphite Compositing

**Scope:** Skia 2D overlay drawing on WebGPU textures, Graphite integration.

**Tech:** `@shopify/react-native-skia` (Graphite build), TypeScript.

**Tasks:**

1. **Skia Graphite detection and fallback**
   - Runtime check: is Graphite/Dawn available?
   - If yes: create Skia Graphite surface backed by Dawn GPU context
   - If no: warn user, fall back to Ganesh offscreen + texture upload

2. **`useSkiaOverlay()` hook**
   - Create Skia canvas that shares the Dawn GPU context
   - `getCanvas()` returns a `SkCanvas` that draws onto the current frame's
     compositing layer
   - Canvas is cleared each frame (overlay is per-frame, not accumulated)
   - Expose full Skia drawing API: drawRect, drawCircle, drawPath, drawText,
     drawImage, drawParagraph, etc.

3. **Compositing render pass**
   - After all compute passes + Skia drawing:
   - Render pass that samples: processed frame texture + Skia overlay texture
   - Alpha-composite Skia layer over computed frame
   - Output to preview canvas AND capture pipeline simultaneously

4. **Coordinate system management**
   - Frame coordinates (raw frame dimensions) → preview coordinates (screen)
   - Provide conversion utilities matching VisionCamera's coordinate API
   - Handle device orientation, mirroring (front camera)

5. **Migration compatibility layer**
   - `DrawableFrame`-like interface that wraps `SkCanvas`
   - Ensure `drawRect`, `drawText`, `drawPath` signatures match VisionCamera's
     Skia frame processor API as closely as possible
   - Document the differences explicitly

**Deliverables:** Skia overlay rendering on a WebGPU canvas, verified that
drawings composite correctly with compute shader output. Working with both
Graphite (primary) and Ganesh (fallback) backends. Performance benchmarks
showing <3ms for typical overlay workloads (text + 10 bounding boxes).

### 6.4 Agent 4: Examples Package (`@webgpu-camera/examples`)

**Scope:** Reference implementations of common native frame processor use cases,
reimplemented using WebGPU ML inference. These are NOT part of the core library —
they live in a separate package in the monorepo that users can install optionally,
copy-paste from, or use as templates for their own ML integrations.

**Tech:** TensorFlow.js (WebGPU backend), ONNX Runtime Web, TypeScript.

**Tasks:**

1. **Package scaffold and infrastructure**
   - Set up `@webgpu-camera/examples` in the monorepo
   - Peer dependency on `react-native-webgpu-camera` (core)
   - Each example is a subpath export (tree-shakeable)
   - README per example with: what it replaces, how to use, model details

2. **TF.js WebGPU backend integration**
   - Initialize `@tensorflow/tfjs` with WebGPU backend
   - Verify it shares the Dawn device with the core compute pipeline
   - Benchmark inference latency on common models
   - Document GPU device sharing (or lack thereof) with workarounds

3. **Reference implementations (one per native frame processor use case):**

   | Example | Replaces VisionCamera Plugin | ML Runtime | Model |
   |---------|----------------------------|------------|-------|
   | `face-detection` | `vision-camera-face-detector` (MLKit) | TF.js WebGPU | MediaPipe Face Detection |
   | `pose-estimation` | `react-native-vision-camera-pose` (MLKit) | TF.js WebGPU | MediaPipe Pose |
   | `object-detection` | `react-native-fast-tflite` + EfficientDet | TF.js WebGPU | SSD MobileNet / EfficientDet |
   | `segmentation` | MLKit Selfie Segmentation | TF.js WebGPU | MediaPipe Selfie Segmentation |
   | `image-classification` | CoreML/TFLite wrappers | TF.js WebGPU | MobileNet v3 |
   | `text-recognition` | `vision-camera-ocr` (MLKit Text) | ONNX Runtime WebGPU | PaddleOCR |
   | `barcode-scanning` | `vision-camera-barcode-scanner` (MLKit/ZXing) | WASM | ZXing compiled to WASM |
   | `custom-onnx-template` | N/A | ONNX Runtime WebGPU | User-provided ONNX model |

4. **ONNX Runtime Web integration (for text-recognition and custom models)**
   - Initialize with WebGPU execution provider
   - `useOnnxModel(modelPath)` hook — load any ONNX model
   - Preprocessing compute shader integration (resize + normalize on GPU,
     feed tensor to ONNX inference without CPU copy)

5. **Result type standardization**
   - All examples return consistent types:
     - `BoundingBox: { x, y, width, height, confidence, label }`
     - `Landmark: { x, y, z?, confidence }`
     - `SegmentationMask: GPUTexture` (stays on GPU for compute shader refinement)
   - Coordinate system matches frame coordinates (convertible to preview coords
     via the core library's coordinate conversion utilities)

6. **Each example includes:**
   - Working hook implementation (e.g., `useFaceDetection()`)
   - Complete demo component showing the hook + Skia overlay drawing
   - Performance benchmarks vs the VisionCamera native plugin it replaces
   - Migration guide section: "If you used X, replace with Y"

**Deliverables:** Published `@webgpu-camera/examples` package with at least
face detection, pose estimation, and object detection working at >15fps
alongside the camera pipeline. Side-by-side comparison with VisionCamera's
native MLKit plugins (feature parity, latency benchmarks).

**Note:** This agent depends on Agent 2 (compute pipeline) for preprocessing
shaders and Agent 3 (Skia) for overlay rendering. Can start with TF.js
integration and model benchmarking while waiting for the core pipeline.

### 6.5 Agent 5: Integration, API Surface, Documentation

**Scope:** Public API, `<GPUCamera>` component, example app, migration guide.

**Tasks:**

1. **`<GPUCamera>` React component**
   - Props: `device`, `isActive`, `frameProcessor`, `zoom`, `torch`, `style`
   - Manages lifecycle: init camera session → start → stop → cleanup
   - Contains the WebGPU preview canvas
   - Handles permissions (camera, microphone)

2. **`useCapture()` hook**
   - `takePhoto({ quality?, format?, embedExif?, embedLocation? })` → `Promise<PhotoResult>`
     PhotoResult includes path, dimensions, and ExifData (if embedded)
   - `startRecording({ bitrate?, audioEnabled?, audioSource?, embedLocation? })`
   - `stopRecording()` → `Promise<VideoResult>`
   - `setLocationTracking(enabled)` — toggle GPS for both photo and video
   - Events: `onRecordingProgress`, `onRecordingError`

3. **`useCameraDevice()` hook**
   - Enumerate available devices
   - Filter by position ('back', 'front')
   - Return format capabilities (resolutions, FPS options)

4. **Example app**
   - Basic camera preview with compute shader (color grade)
   - Motion detection with visual overlay
   - Face detection with bounding boxes (from `@webgpu-camera/examples`)
   - Photo capture (verify Skia overlays appear in saved photo)
   - Photo EXIF verification (display captured metadata: ISO, exposure, GPS)
   - Video recording with audio (verify Skia overlays appear in video)
   - Side-by-side with VisionCamera showing the capture limitation
   - Graphite vs Ganesh fallback visual comparison

5. **Benchmark app** (`apps/benchmark/`)
   - Implement all 7 benchmark scenarios (see §8.4):
     baseline, color-filter, motion-detect, face-detection,
     multi-pass, heavy-overlay, capture
   - Each scenario runs in 3 modes: VisionCamera + native plugin,
     VisionCamera + Skia frame processor, react-native-webgpu-camera
   - `MetricsCollector.ts` — aggregates per-frame timestamps from
     both the Rust native module (§8.2) and GPU timestamp queries
   - `StatsDisplay.tsx` — real-time overlay showing FPS, processor ms,
     CPU%, GPU compute ms during a benchmark run
   - `ThermalGuard.ts` — monitors device thermal state, blocks next
     run until device returns to nominal temperature
   - `Reporter.ts` — exports per-run JSON results (§8.6 format)
   - `report/generate.ts` — produces comparison tables and charts
     from collected JSON data across all scenarios and devices
   - VisionCamera installed as a dev dependency for the benchmark app
     only (not a dependency of the core library)

6. **Setup documentation**
   - Install instructions (including `SK_GRAPHITE=1` build flag)
   - Migration guide from VisionCamera + Skia frame processors
   - Custom compute shader authoring guide
   - Performance tuning guide
   - Troubleshooting (Graphite detection, Android device issues)
   - Benchmark methodology and how to reproduce results

**Deliverables:** Published npm package with complete API, working example app,
benchmark app with reproducible results on at least 3 devices (see §8.7),
comprehensive README, migration guide document.

---

## 7. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Dawn cannot import IOSurface/HardwareBuffer as external texture | Critical — no zero-copy | Medium | Spike this first (Agent 1, Task 4-5). Fallback: `writeTexture()` copy path works, ~0.3ms overhead |
| Skia Graphite not stable enough | High — overlay compositing degrades | Medium | Ganesh fallback path (one extra copy). Monitor react-native-skia releases |
| Worklet thread context incompatibility between react-native-wgpu and our native module | High — can't dispatch compute from frame callback | Low | We control threading (standalone). Use Reanimated worklets exclusively. Spike early |
| TF.js WebGPU backend doesn't share Dawn device | Medium — separate GPU context, extra memory | Medium | Test in spike. Fallback: TF.js creates its own device, we copy the result texture |
| UniFFI overhead for frame handle passing | Low — control plane only | Low | Pixel data never goes through UniFFI. Only tiny handles and config structs |
| Android Camera2 device fragmentation | Medium — crashes on specific devices | Medium | Start with Pixel + Samsung. Community testing. Narrow camera config (no exotic formats) |
| react-native-wgpu breaking changes | Medium — our entire pipeline depends on it | Low | Pin version, contribute upstream, maintain patches if needed |

### 7.1 First Spike Priority

Before building anything else, Agent 1 and Agent 2 collaborate on a proof-of-concept:

1. Can we import a platform camera buffer as a Dawn/WebGPU texture? (zero-copy)
2. Can we dispatch a compute shader on that texture from a Reanimated worklet?
3. Can we render the compute output to a WebGPU canvas at 60fps?

If all three work, the rest is engineering. If #1 fails, we fall back to the
copy path. If #2 fails, we restructure the threading model. If #3 fails,
the project is not viable (extremely unlikely — this is demonstrated by
react-native-wgpu already).

---

## 8. Benchmark Suite

### 8.1 What We Measure

Three levels of comparison, each telling a different story:

**Level 1: Base pipeline overhead** (camera → preview, no processing)
"What does the library itself cost?" Same camera, same frames, different
pipeline path. Establishes that our base overhead isn't worse than VisionCamera.

**Level 2: Frame processor throughput** (processing pipeline comparison)
"How fast can each library process frames?" Same operation, different
implementation. This is where compute shaders vs JS/native diverges.

**Level 3: Feature-specific comparisons** (same task, different approach)
"Face detection via MLKit native plugin vs TF.js WebGPU." Same visible result,
completely different implementation. Includes accuracy metrics, not just speed.

### 8.2 Metrics Collected Per Frame

```rust
// Injected by the Rust native module at each pipeline stage
pub struct FrameMetrics {
    pub camera_timestamp: u64,      // Platform camera's timestamp on the frame
    pub native_receive_ns: u64,     // When Rust module received the frame
    pub gpu_import_ns: u64,         // When zero-copy import completed
    pub js_dispatch_ns: u64,        // When JS worklet received the texture handle
    pub compute_start_ns: u64,      // When first compute dispatch began
    pub compute_end_ns: u64,        // When last compute dispatch completed
    pub skia_overlay_ns: u64,       // When Skia overlay completed
    pub present_ns: u64,            // When render pass submitted to GPU queue
    pub display_ns: u64,            // When frame appeared on screen (estimated)
}
```

Additionally, WebGPU GPU timestamp queries measure actual GPU execution time —
not when the command was submitted, but when the GPU started and finished work:

```typescript
const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });

const encoder = device.createCommandEncoder();
encoder.writeTimestamp(querySet, 0);          // GPU records start time
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.dispatchWorkgroups(width / 16, height / 16);
pass.end();
encoder.writeTimestamp(querySet, 1);          // GPU records end time
encoder.resolveQuerySet(querySet, 0, 2, resultBuffer, 0);
device.queue.submit([encoder.finish()]);

// Readback: two 64-bit nanosecond timestamps
// Difference = actual GPU compute time, no JS overhead noise
```

VisionCamera has no equivalent — their Skia RuntimeEffects and native plugins
run on opaque GPU paths with no user-accessible timing. For VisionCamera, we
measure wall-clock time from JS (includes scheduling overhead).

### 8.3 Benchmark App Structure

```
apps/
└── benchmark/
    ├── scenarios/
    │   ├── baseline.tsx          → Camera preview, no processing
    │   ├── color-filter.tsx      → Simple per-pixel color transform
    │   ├── motion-detect.tsx     → Temporal processing (our clear win)
    │   ├── face-detection.tsx    → ML inference comparison
    │   ├── multi-pass.tsx        → 3 chained operations
    │   ├── heavy-overlay.tsx     → Skia drawing with 50+ elements
    │   └── capture.tsx           → Photo + video capture with overlays
    ├── harness/
    │   ├── MetricsCollector.ts   → Collects per-frame timing data
    │   ├── StatsDisplay.tsx      → Real-time FPS/latency/CPU overlay
    │   ├── Reporter.ts           → Export results as JSON/CSV
    │   └── ThermalGuard.ts       → Wait for thermal cooldown between runs
    └── report/
        └── generate.ts           → Produce comparison charts from data
```

### 8.4 Benchmark Scenarios

Each scenario runs in three modes for direct comparison:
1. **VisionCamera + native plugin** (the baseline people migrate from)
2. **VisionCamera + Skia frame processor** (current Skia path)
3. **react-native-webgpu-camera** (our implementation)

**Scenario: baseline (no processing)**
```
Metric: frame-to-screen latency, CPU%, memory
Expected: roughly similar — our WebGPU render pass vs their native preview.
Proves our base overhead doesn't regress.
```

**Scenario: color-filter (simple per-pixel)**
```
VisionCamera: Skia RuntimeEffect (SKSL fragment shader)
Ours: WGSL compute shader

Metric: processor execution time, FPS, CPU%
Expected: similar GPU time (both trivial per-pixel on GPU).
Our CPU% should be lower (less JS→native bridge overhead).
Tests bridge efficiency, not GPU capability.
```

**Scenario: motion-detect (temporal — the showcase)**
```
VisionCamera: NOT POSSIBLE without native code. Compare against:
  (a) JS pixel iteration via toArrayBuffer() (~50ms+ per 1080p frame)
  (b) Custom native plugin (if built for fair comparison)
Ours: WGSL compute shader with persistent storage buffer (~2-5ms)

Metric: processor execution time, FPS, CPU%
Expected: 10-25× speedup. JS iteration is CPU-bound at 50ms+.
Our compute shader is 2-5ms with near-zero CPU.
This is the "why this library exists" demo.
```

**Scenario: face-detection (ML inference)**
```
VisionCamera: MLKit native plugin (platform-optimized)
Ours: TF.js WebGPU (MediaPipe Face Detection)

Metric: inference latency, detection accuracy (IoU vs ground truth), FPS
Expected: MLKit may win on raw latency (highly platform-optimized).
Our accuracy should be comparable (same MediaPipe models).
Our CPU% should be lower (GPU inference vs CPU dispatch).
Be honest if MLKit wins — the comparison includes accuracy, not just speed.
```

**Scenario: multi-pass (3 chained: color grade → edge detect → overlay)**
```
VisionCamera: 3 Skia RuntimeEffects chained, or native plugin
Ours: 3 compute dispatches in one command buffer

Metric: total pipeline time, FPS
Expected: clear win. Chained compute dispatches share one command buffer —
no JS round-trips between passes. VisionCamera's Skia path requires
multiple RuntimeEffects with intermediate surface allocations.
```

**Scenario: heavy-overlay (50+ Skia draw calls)**
```
VisionCamera: useSkiaFrameProcessor with extensive drawing
Ours: useSkiaOverlay with same drawing code

Metric: overlay render time, total FPS
Expected: similar Skia performance (same drawing API).
Tests that our Skia integration doesn't add overhead vs VisionCamera's.
Both with and without Graphite for our side.
```

**Scenario: capture (feature comparison, not just speed)**
```
VisionCamera: capture photo → overlays NOT in output
Ours: capture photo → overlays IN output

Metric: capture latency, output correctness (overlay visible? screenshot both)
Expected: this is a feature comparison. Our capture includes overlays.
Theirs doesn't. The benchmark app displays both outputs side-by-side.
Also: verify EXIF data present in our captured photos.
```

### 8.5 Benchmark Protocol (critical for fairness)

```
For each scenario:
  1. Kill all background apps, enable airplane mode
  2. Wait 30 seconds for thermal baseline
  3. Run VisionCamera version for 60 seconds, collect per-frame metrics
  4. Stop, wait 60 seconds for thermal cooldown
  5. Run our version for 60 seconds, collect per-frame metrics
  6. Stop, wait 60 seconds
  7. Repeat steps 3-6 two more times (3 runs each, alternating order)
  8. Report: median, p95, p99 for each metric, with confidence intervals
```

The thermal cooldown between runs is non-negotiable. Without it, the second
library always looks worse because the SoC throttles from accumulated heat.
Alternating run order across repetitions cancels thermal bias.

`ThermalGuard.ts` monitors `ProcessInfo.thermalState` (iOS) or thermal
service (Android) and blocks the next run until the device returns to nominal.

### 8.6 Report Format

Per-run JSON output:

```json
{
  "device": "iPhone 15 Pro",
  "os": "iOS 18.2",
  "scenario": "motion-detect",
  "library": "react-native-webgpu-camera",
  "duration_seconds": 60,
  "frames_total": 1800,
  "frames_dropped": 0,
  "fps_median": 60.0,
  "fps_p5": 59.2,
  "processor_ms_median": 3.2,
  "processor_ms_p95": 4.8,
  "processor_ms_p99": 6.1,
  "gpu_compute_ms_median": 2.1,
  "gpu_compute_ms_p95": 3.0,
  "cpu_percent_median": 8.2,
  "memory_mb": 142,
  "thermal_state_start": "nominal",
  "thermal_state_end": "nominal"
}
```

`report/generate.ts` produces comparison tables:

```
┌─────────────────────────────────────────────────────────────┐
│ Motion Detection — iPhone 15 Pro — 1080p @ 60fps target     │
│                                                              │
│                    VisionCamera (JS)  │  WebGPU Camera       │
│ Processor time     52.3ms median     │  3.2ms median        │
│ FPS achieved       18 (frame drops)  │  60 (no drops)       │
│ CPU usage          87%               │  8%                  │
│ GPU compute        N/A (CPU-bound)   │  2.1ms               │
│ Overlay in capture ❌ No             │  ✅ Yes              │
└─────────────────────────────────────────────────────────────┘
```

### 8.7 Target Devices for Benchmarking

Run the full suite on at least these devices to cover the hardware spectrum:

| Device | SoC | GPU | Why |
|--------|-----|-----|-----|
| iPhone 15 Pro | A17 Pro | 6-core Apple GPU | Current high-end iOS |
| iPhone 13 | A15 | 5-core Apple GPU | Mid-range iOS baseline |
| Pixel 8 | Tensor G3 | Mali-G715 | Reference Android (Google) |
| Samsung S24 | Snapdragon 8 Gen 3 | Adreno 750 | High-end Android |
| Samsung A54 | Exynos 1380 | Mali-G68 | Mid-range Android (mass market) |

Results published per-device. Performance claims in the README reference the
lowest-performing device that meets 60fps, not just the flagship.

---

## 9. Performance Targets

Validated via the benchmark suite (§8). All measurements taken using the
benchmark protocol with thermal cooldown. Targets reference the mid-range
device floor (iPhone 13 / Samsung A54), not flagships.

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Frame delivery latency | <1ms | Rust timestamps: `gpu_import_ns` - `native_receive_ns` |
| Single compute shader dispatch | <5ms | GPU timestamp queries (§8.2) |
| Full pipeline (3 compute + Skia + render) | <12ms | GPU timestamps: first dispatch start → present end |
| Preview frame rate | 60fps | MetricsCollector sustained FPS over 60s |
| Photo capture latency | <100ms | JS timestamp: capture call → PhotoResult returned |
| Video recording overhead | <5% battery vs raw camera | A/B test over 5 minutes (benchmark app) |
| ML inference (face detection) | >20fps | From `@webgpu-camera/examples`, alongside full pipeline |
| Memory overhead vs VisionCamera | <50MB additional | Benchmark app memory profiling |
| Ganesh fallback overhead | <1ms per frame | Skia overlay round-trip measured in fallback path |

---

## 10. Open Questions for Development

1. **Dawn external texture import API:** What is the exact Dawn C++ API for
   importing an `IOSurface` as a `WGPUTexture`? Is this exposed through
   `react-native-wgpu`'s JS API, or do we need to call Dawn C++ from Rust?

2. **Graphite shared GPU context:** When Skia Graphite and react-native-wgpu
   both use Dawn, is it automatically the same `GPUDevice`? Or do we need
   to explicitly share the device between them?

3. **App backgrounding:** When the app goes to background, both the camera and
   GPU context need graceful suspend/resume. What's the cleanup sequence?

4. **Hot reload during development:** Can WGSL shader strings hot-reload via
   Metro fast refresh? (Likely yes — they're just JS strings.)

5. **MediaRecorder Surface + Dawn rendering:** Can Dawn render to a Surface
   obtained from `MediaRecorder.getSurface()` on Android? This is the
   zero-copy video recording path. If not, fallback is rendering to a
   shared `HardwareBuffer` and feeding it to MediaCodec (more complex).

6. **GPU timestamp query availability:** Are timestamp queries supported on
   all target devices via Dawn? Some mobile GPUs have limited query support.
   Need to detect capability and gracefully disable benchmarking metrics
   on devices that don't support timestamps.

---

## 11. Development Environment Guidelines

### 11.1 Monorepo Setup

Bun workspaces monorepo. Single `package.json` at the root with workspace
definitions. No Nx, no Turborepo — bun workspaces handle dependency hoisting
and script orchestration natively.

```
react-native-webgpu-camera/
├── package.json                  ← bun workspace root
├── bun.lockb
├── .gitignore
├── tsconfig.base.json            ← shared TypeScript config
├── .eslintrc.js                  ← shared lint config
├── .prettierrc
├── packages/
│   ├── react-native-webgpu-camera/
│   │   ├── package.json          ← "name": "react-native-webgpu-camera"
│   │   ├── tsconfig.json         ← extends ../../tsconfig.base.json
│   │   ├── src/
│   │   │   ├── camera/
│   │   │   ├── compute/
│   │   │   ├── skia/
│   │   │   ├── capture/
│   │   │   ├── processors/
│   │   │   └── index.ts
│   │   ├── rust/                 ← Rust crate (camera + encoder + GPU bridge)
│   │   │   ├── Cargo.toml
│   │   │   ├── src/
│   │   │   └── uniffi.toml
│   │   ├── ios/                  ← generated by create-expo-module + UniFFI output
│   │   ├── android/              ← generated by create-expo-module + UniFFI output
│   │   └── expo-module.config.json
│   │
│   └── @webgpu-camera/examples/
│       ├── package.json          ← "name": "@webgpu-camera/examples"
│       ├── tsconfig.json
│       └── src/
│           ├── face-detection/
│           ├── pose-estimation/
│           ├── object-detection/
│           └── ...
│
├── apps/
│   ├── example/                  ← Expo app (demo)
│   │   ├── package.json
│   │   ├── app.json
│   │   ├── eas.json
│   │   └── app/
│   └── benchmark/                ← Expo app (performance comparison)
│       ├── package.json
│       ├── app.json
│       ├── eas.json
│       └── app/
│
└── scripts/
    ├── build-rust.sh             ← Cross-compile Rust for all targets
    ├── generate-bindings.sh      ← Run uniffi-bindgen-react-native
    └── build-skia-graphite.sh    ← Build Skia with SK_GRAPHITE=1
```

Root `package.json`:

```json
{
  "name": "react-native-webgpu-camera-monorepo",
  "private": true,
  "workspaces": [
    "packages/*",
    "apps/*"
  ],
  "scripts": {
    "build": "bun run --filter 'react-native-webgpu-camera' build",
    "build:rust": "./scripts/build-rust.sh",
    "build:bindings": "./scripts/generate-bindings.sh",
    "build:skia": "SK_GRAPHITE=1 yarn build-skia",
    "lint": "bun run --filter '*' lint",
    "typecheck": "bun run --filter '*' typecheck",
    "test": "bun run --filter '*' test",
    "example:ios": "cd apps/example && bun expo run:ios",
    "example:android": "cd apps/example && bun expo run:android",
    "benchmark:ios": "cd apps/benchmark && bun expo run:ios",
    "benchmark:android": "cd apps/benchmark && bun expo run:android"
  }
}
```

### 11.2 Scaffolding the Native Module

Use `create-expo-module` to scaffold the Expo Module even though the native
code lives in the core package, not the examples package. The generated
boilerplate gives us the correct `ios/`, `android/`, `expo-module.config.json`,
and podspec structure that Expo's autolinking expects.

```bash
# From the monorepo root
cd packages
bunx create-expo-module react-native-webgpu-camera

# This generates:
# packages/react-native-webgpu-camera/
#   ├── ios/
#   ├── android/
#   ├── src/
#   ├── expo-module.config.json
#   ├── package.json
#   └── tsconfig.json
```

Then modify the generated structure:
- Replace the generated Swift/Kotlin stubs with UniFFI-generated bindings
- Add the `rust/` directory with Cargo.toml
- Configure `expo-module.config.json` to include the compiled Rust libraries
- Update `package.json` with the correct entry points and subpath exports

The `expo-module.config.json` needs to reference the Rust-compiled shared
libraries. On iOS this is a `.framework` or `.a` file linked via the podspec.
On Android this is a `.so` loaded via `System.loadLibrary()` in the generated
Kotlin module.

### 11.3 Rust Cross-Compilation

The Rust crate needs to compile for all target architectures. Set up
`scripts/build-rust.sh`:

```bash
#!/bin/bash
set -euo pipefail

# iOS targets
rustup target add aarch64-apple-ios        # Device
rustup target add aarch64-apple-ios-sim    # Simulator (Apple Silicon)
rustup target add x86_64-apple-ios         # Simulator (Intel)

# Android targets
rustup target add aarch64-linux-android    # ARM64 (most devices)
rustup target add armv7-linux-androideabi  # ARM32 (older devices)
rustup target add x86_64-linux-android     # Emulator (Intel)
rustup target add i686-linux-android       # Emulator (32-bit)

# Android requires NDK — set ANDROID_NDK_HOME
export ANDROID_NDK_HOME="${ANDROID_HOME}/ndk/$(ls ${ANDROID_HOME}/ndk | sort -V | tail -1)"

cd packages/react-native-webgpu-camera/rust

# Build iOS (universal binary for sim)
cargo build --release --target aarch64-apple-ios
cargo build --release --target aarch64-apple-ios-sim

# Build Android
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 build --release

echo "Built for all targets. Run generate-bindings.sh next."
```

UniFFI bindings generation (`scripts/generate-bindings.sh`):

```bash
#!/bin/bash
set -euo pipefail

cd packages/react-native-webgpu-camera

# Generate TypeScript + JSI C++ + Swift + Kotlin from Rust annotations
bunx uniffi-bindgen-react-native \
  --config rust/uniffi.toml \
  --out-dir generated/
```

### 11.4 Apps Setup

Both `apps/example` and `apps/benchmark` are Expo apps created with:

```bash
cd apps
bunx create-expo-app example --template blank-typescript
bunx create-expo-app benchmark --template blank-typescript
```

Each app's `package.json` references the local packages:

```json
{
  "dependencies": {
    "react-native-webgpu-camera": "workspace:*",
    "@webgpu-camera/examples": "workspace:*",
    "react-native-wgpu": "^x.x.x",
    "@shopify/react-native-skia": "^x.x.x",
    "react-native-reanimated": "^x.x.x",
    "expo-media-library": "^x.x.x"
  }
}
```

The benchmark app additionally depends on VisionCamera for comparison:

```json
{
  "devDependencies": {
    "react-native-vision-camera": "^4.x.x",
    "react-native-worklets-core": "^x.x.x"
  }
}
```

### 11.5 Testing App Requirements

Both the example and benchmark apps MUST test the full capture pipeline:

**Photo capture → save to device media library:**
```typescript
import * as MediaLibrary from 'expo-media-library';

async function captureAndSave() {
  const { status } = await MediaLibrary.requestPermissionsAsync();
  if (status !== 'granted') return;

  const photo = await capture.takePhoto({
    quality: 0.95,
    format: 'jpeg',
    embedExif: true,
    embedLocation: true,
  });

  // Save to device photo library — user can verify in Photos/Gallery
  const asset = await MediaLibrary.saveToLibraryAsync(photo.path);
  console.log('Saved to library:', asset.uri);
}
```

**Video recording → save to device media library:**
```typescript
async function recordAndSave() {
  capture.startRecording({
    bitrate: 8_000_000,
    audioEnabled: true,
    audioSource: 'microphone',
    embedLocation: true,
  });

  // Record for 5 seconds
  await new Promise(resolve => setTimeout(resolve, 5000));

  const video = await capture.stopRecording();
  const asset = await MediaLibrary.saveToLibraryAsync(video.path);
  console.log('Saved to library:', asset.uri);
}
```

This is the critical end-to-end test: capture a photo with Skia overlays →
open the device's Photos app → verify the overlays are visible in the saved
image. Same for video. If this works, the entire pipeline is validated. If it
doesn't, nothing else matters.

The apps should include a UI that:
- Shows a live camera preview with a compute shader effect and Skia overlays
- Has a capture button that saves to media library
- Has a record button that starts/stops video recording and saves to library
- Displays a thumbnail of the last captured photo/video with EXIF metadata
- Shows a "open in Photos" button that uses `Linking.openURL` to jump to
  the saved asset in the system gallery

### 11.6 EAS Build Configuration

Both apps use EAS Build for device testing. Native Rust compilation and Skia
Graphite builds require custom build configurations.

`apps/example/eas.json`:

```json
{
  "cli": { "version": ">= 12.0.0" },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": {
        "buildConfiguration": "Debug"
      },
      "android": {
        "buildType": "apk"
      },
      "env": {
        "SK_GRAPHITE": "1"
      }
    },
    "preview": {
      "distribution": "internal",
      "env": {
        "SK_GRAPHITE": "1"
      }
    },
    "production": {
      "env": {
        "SK_GRAPHITE": "1"
      }
    }
  }
}
```

**EAS Build custom prebuild steps** — the Rust compilation and UniFFI binding
generation need to run before the native build. Use `eas.json` hooks or an
`eas-build-pre-install` script:

```json
{
  "build": {
    "development": {
      "prebuildCommand": "cd ../../ && ./scripts/build-rust.sh && ./scripts/generate-bindings.sh"
    }
  }
}
```

Alternatively, precompile the Rust libraries locally and commit the binaries
to the repo (common pattern for small native libraries in Expo modules). This
avoids needing Rust toolchain on EAS Build workers:

```
packages/react-native-webgpu-camera/
  ├── prebuilt/
  │   ├── ios/
  │   │   ├── libwebgpu_camera.a          (aarch64)
  │   │   └── libwebgpu_camera_sim.a      (aarch64-sim + x86_64 universal)
  │   └── android/
  │       ├── arm64-v8a/libwebgpu_camera.so
  │       ├── armeabi-v7a/libwebgpu_camera.so
  │       └── x86_64/libwebgpu_camera.so
  └── ...
```

This is the approach `react-native-wgpu` uses for Dawn prebuilts — it's
proven to work with EAS Build.

### 11.7 Local Development Workflow

**Day-to-day development cycle:**

```bash
# Initial setup (once)
bun install
./scripts/build-rust.sh        # Compile Rust for all targets
./scripts/generate-bindings.sh  # Generate TypeScript + native bindings
SK_GRAPHITE=1 yarn build-skia  # Build Skia with Graphite backend

# Run example app on device
cd apps/example
bun expo prebuild               # Generate native projects
bun expo run:ios                # Build and run on iOS simulator/device
# or
bun expo run:android            # Build and run on Android emulator/device

# After changing TypeScript (hot reload works automatically via Metro)
# After changing WGSL shaders (hot reload works — they're JS strings)
# After changing Rust code:
cd ../../
./scripts/build-rust.sh
./scripts/generate-bindings.sh
cd apps/example
bun expo run:ios                # Rebuild native

# Run tests
bun test                        # Unit tests across all packages

# EAS Build for physical device testing
cd apps/example
eas build --platform ios --profile development
eas build --platform android --profile development
```

**WGSL shader development:**
Shaders are strings in TypeScript files. Metro fast refresh picks up changes
immediately — no native rebuild needed. This is one of the core developer
experience advantages over VisionCamera's native frame processor plugins.

```typescript
// Change this string → save → Metro hot reloads → see result instantly
const myShader = createComputeShader({
  name: 'my_filter',
  wgsl: `
    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      // Edit this, save, see the result on the camera preview immediately
    }
  `,
});
```

### 11.8 Git Repository

Single git repo at the workspace root. All packages, apps, scripts, and
documentation live in one repository.

```bash
# .gitignore (key entries)
node_modules/
.expo/
ios/Pods/
android/.gradle/
android/build/
*.jsbundle
*.hbc

# Rust build artifacts (NOT the prebuilts — those are committed)
packages/react-native-webgpu-camera/rust/target/

# Generated UniFFI bindings (regenerated from Rust source)
packages/react-native-webgpu-camera/generated/

# Prebuilt Rust libraries ARE committed (for EAS Build compatibility)
# !packages/react-native-webgpu-camera/prebuilt/

# Benchmark results (committed for historical comparison)
# !apps/benchmark/results/
```

**Branch strategy:**
- `main` — stable, all tests pass, prebuilt binaries up to date
- `develop` — integration branch for feature PRs
- `agent/*` — per-agent feature branches (e.g., `agent/rust-camera`,
  `agent/compute-pipeline`, `agent/skia-graphite`, etc.)
- `spike/*` — for technical spikes (e.g., `spike/dawn-iosurface-import`)

**Commit conventions:**
Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `perf:`, `test:`).
Scope per package: `feat(camera):`, `fix(compute):`, `perf(skia):`,
`feat(examples):`, `test(benchmark):`.

### 11.9 CI Considerations

**Minimum CI checks (GitHub Actions):**
- `bun install` — verify lockfile is up to date
- `bun run typecheck` — TypeScript compilation across all packages
- `bun run lint` — ESLint across all packages
- `bun run test` — unit tests (compute shader tests run against test images)
- Verify prebuilt Rust binaries match the Rust source (hash check)

**Native builds via EAS:**
Full native builds (iOS + Android) run via EAS Build, not GitHub Actions.
Native compilation (Xcode, Gradle, Rust cross-compilation, Skia Graphite
build) is too slow and complex for CI runners. EAS Build handles the native
toolchains.

Trigger EAS builds from CI on:
- Every PR to `main` or `develop` (preview profile, internal distribution)
- Every merge to `main` (production profile)

**Benchmark runs:**
Manual. Benchmarks require physical devices with controlled thermal state —
they can't run in emulators or CI. The benchmark app is built via EAS,
installed on target devices, and run manually following the protocol in §8.5.
Results are committed to `apps/benchmark/results/` for historical tracking.
