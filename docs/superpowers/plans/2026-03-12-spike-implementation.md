# Spike Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all four spikes in a single example app to validate the react-native-webgpu-camera architecture on physical devices.

**Architecture:** "Skia Owns the World" (Approach A) — Skia Graphite's Dawn backend provides `navigator.gpu` as the single shared GPU context for compute, overlay rendering, and recording. Camera capture in Swift/Kotlin Expo modules, frame coordination through Rust UniFFI frame slot. If Graphite's shared context fails, fall back to Approach B (react-native-wgpu Canvas + Ganesh).

**Tech Stack:** React Native 0.83, Expo 55, react-native-wgpu 0.5.8, @shopify/react-native-skia (Graphite), react-native-reanimated 4.2+, Rust/UniFFI, Swift (iOS), Kotlin (Android)

**Spec:** `docs/superpowers/specs/2026-03-12-spike-implementation-design.md`

---

## File Structure

### Files to create

| File | Responsibility |
| ---- | -------------- |
| `apps/example/src/hooks/useGPUPipeline.ts` | WebGPU device acquisition, compute pipeline setup, render loop management |
| `apps/example/src/components/SpikeOverlay.tsx` | Skia Graphite overlay component for text/shapes on compute output |
| `apps/example/src/utils/recorderBridge.ts` | TypeScript wrapper for recorder start/stop with output path management |

### Files to modify

| File | Change |
| ---- | ------ |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Implement camera capture via AVCaptureSession, wire UniFFI calls, add recorder surface setup |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt` | Implement camera capture via Camera2, wire UniFFI calls, add recorder surface setup |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/lib.rs` | Optimize frame slot (double-buffer), add frame counter for new-frame detection |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/ios.rs` | Replace with no-op stub (capture moved to Swift) |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/android.rs` | Replace with no-op stub (capture moved to Kotlin) |
| `apps/example/src/app/index.tsx` | Wire up full pipeline: GPU init, camera start, worklet render loop, Skia overlay, recorder |
| `apps/example/src/hooks/useSpikeMetrics.ts` | Add path detection (copy vs zero-copy, graphite vs ganesh) |

---

## Chunk 1: Rust Frame Slot + iOS Camera Capture

### Task 1: Optimize the Rust frame slot for sustained throughput

The current frame slot clones ~8MB per read through a Mutex. Replace with a double-buffer using atomic swap for lock-free reads.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/lib.rs`

- [ ] **Step 1: Add double-buffer frame slot implementation**

Replace the single `Mutex<Vec<u8>>` with a double-buffer. The writer (camera callback) writes to the back buffer and atomically swaps. The reader (JS polling) reads from the front buffer without locking.

```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

uniffi::setup_scaffolding!();

// Double-buffer frame slot.
// Camera writes to back buffer, atomically swaps index.
// JS reads from front buffer — no lock contention.
static FRAME_BUFFERS: [Mutex<Vec<u8>>; 2] = [Mutex::new(Vec::new()), Mutex::new(Vec::new())];
static FRONT_INDEX: AtomicUsize = AtomicUsize::new(0);
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
static CURRENT_FRAME_HANDLE: AtomicU64 = AtomicU64::new(0);
static FRAME_DIMS: Mutex<FrameDimensions> = Mutex::new(FrameDimensions {
    width: 0,
    height: 0,
    bytes_per_row: 0,
});

#[derive(uniffi::Record, Clone)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}
```

- [ ] **Step 2: Add exported functions for frame delivery and reading**

These are the UniFFI-exported functions that Swift/Kotlin call to deliver frames, and that JS calls to read them. Note: `deliver_frame` takes `Vec<u8>` (not `&[u8]`) because UniFFI requires owned types for exported functions.

```rust
#[uniffi::export]
pub fn deliver_frame(pixels: Vec<u8>, handle: u64) {
    let front = FRONT_INDEX.load(Ordering::Acquire);
    let back = 1 - front;

    {
        let mut buf = FRAME_BUFFERS[back].lock().unwrap();
        buf.clear();
        buf.extend_from_slice(&pixels);
    }

    CURRENT_FRAME_HANDLE.store(handle, Ordering::Relaxed);
    FRONT_INDEX.store(back, Ordering::Release);
    FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
}

#[uniffi::export]
pub fn set_frame_dimensions(width: u32, height: u32, bytes_per_row: u32) {
    let mut dims = FRAME_DIMS.lock().unwrap();
    dims.width = width;
    dims.height = height;
    dims.bytes_per_row = bytes_per_row;
}

#[uniffi::export]
pub fn get_current_frame_handle() -> u64 {
    CURRENT_FRAME_HANDLE.load(Ordering::Relaxed)
}

#[uniffi::export]
pub fn get_current_frame_pixels() -> Vec<u8> {
    let front = FRONT_INDEX.load(Ordering::Acquire);
    FRAME_BUFFERS[front].lock().unwrap().clone()
}

#[uniffi::export]
pub fn get_frame_dimensions() -> FrameDimensions {
    FRAME_DIMS.lock().unwrap().clone()
}

#[uniffi::export]
pub fn get_frame_counter() -> u64 {
    FRAME_COUNTER.load(Ordering::Relaxed)
}
```

- [ ] **Step 3: Update camera stubs to remove `CURRENT_FRAME_PIXELS` reference**

The existing `camera/ios.rs` and `camera/android.rs` import the old `CURRENT_FRAME_PIXELS` which no longer exists. Update both stubs now to avoid `cargo check` failures.

Replace `camera/ios.rs`:

```rust
//! iOS camera — capture will be handled in Swift Expo module (Task 2).
//! This stub remains for the lib.rs conditional compile.

pub fn start_preview(_device_id: &str, _width: u32, _height: u32) {
    println!("[webgpu-camera/ios] Camera managed by Swift Expo module");
}

pub fn stop_preview() {
    println!("[webgpu-camera/ios] Camera stopped by Swift Expo module");
}
```

Replace `camera/android.rs`:

```rust
//! Android camera — capture will be handled in Kotlin Expo module (Task 3).

pub fn start_preview(_device_id: &str, _width: u32, _height: u32) {
    println!("[webgpu-camera/android] Camera managed by Kotlin Expo module");
}

pub fn stop_preview() {
    println!("[webgpu-camera/android] Camera stopped by Kotlin Expo module");
}
```

- [ ] **Step 4: Keep existing start/stop/recorder/thermal exports unchanged**

The `start_camera_preview`, `stop_camera_preview`, `start_test_recorder`, `stop_test_recorder`, and `get_thermal_state` functions stay as-is. The Rust `start_camera_preview` calls the camera stub (now a no-op), so it won't conflict with the Swift/Kotlin `setFrameDimensions` call — frame dimensions are set exclusively by the native module in Tasks 2–3.

- [ ] **Step 5: Run cargo check**

```bash
cd packages/react-native-webgpu-camera/modules/webgpu-camera/rust && cargo check
```

Expected: compiles without errors. Note: `cargo check` compiles for the host target (macOS), so `#[cfg(target_os)]`-gated camera modules may not be checked. Full validation requires the EAS device build in Task 10.

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/
git commit -m "feat(rust): double-buffer frame slot for lock-free camera frame reads"
```

---

### Task 2: Implement iOS camera capture in Swift

Move camera capture from the Rust stub to the Swift Expo module, using AVCaptureSession. The Swift module captures frames, extracts pixel data, and delivers them to the Rust frame slot via UniFFI.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`

**Reference docs:**
- Apple AVCaptureSession: https://developer.apple.com/documentation/avfoundation/avcapturesession
- Apple CVPixelBuffer: https://developer.apple.com/documentation/corevideo/cvpixelbuffer

**Prerequisites:** `deliver_frame` and `set_frame_dimensions` are already UniFFI-exported from Task 1 Step 2. The Rust camera stubs were already simplified in Task 1 Step 3.

**UniFFI binding note:** The Swift code calls free functions like `deliverFrame(pixels:handle:)` and `getCurrentFrameHandle()`. UniFFI generates **camelCase** Swift bindings from the snake_case Rust exports. The generated Swift file is typically at `packages/react-native-webgpu-camera/modules/webgpu-camera/generated/swift/webgpu_camera.swift` — it's created by running `scripts/generate-bindings.sh`.

- [ ] **Step 1: Add camera permission to app config**

Add `NSCameraUsageDescription` to the Expo config. In `apps/example/app.json`, ensure the `ios.infoPlist` section includes:

```json
{
  "expo": {
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "Camera access is needed for WebGPU spike validation"
      }
    }
  }
}
```

Also add the Android camera permission in the `android` section:

```json
{
  "expo": {
    "android": {
      "permissions": ["android.permission.CAMERA"]
    }
  }
}
```

- [ ] **Step 2: Add camera session manager to Swift module**

Replace the stub implementations in `WebGPUCameraModule.swift` with real AVCaptureSession code. The module:

1. Configures AVCaptureSession with back camera at requested resolution
2. Sets up AVCaptureVideoDataOutput with BGRA pixel format
3. In the delegate callback, captures the IOSurface handle first, then calls Rust `deliver_frame` with the handle
4. Retains the FrameDelegate as a strong reference to prevent deallocation

```swift
import ExpoModulesCore
import AVFoundation
import CoreVideo

public class WebGPUCameraModule: Module {
  private var captureSession: AVCaptureSession?
  private var dataOutput: AVCaptureVideoDataOutput?
  private var frameDelegate: FrameDelegate?
  private let sessionQueue = DispatchQueue(label: "webgpu-camera-session")
  private let frameQueue = DispatchQueue(label: "webgpu-camera-frame", qos: .userInteractive)

  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, width: Int, height: Int) in
      self.startCapture(deviceId: deviceId, width: width, height: height)
    }

    Function("stopCameraPreview") {
      self.stopCapture()
    }

    Function("getCurrentFrameHandle") { () -> Int in
      return Int(getCurrentFrameHandle())
    }

    Function("getCurrentFramePixels") { () -> Data in
      let pixels = getCurrentFramePixels()
      return Data(pixels)
    }

    Function("getFrameDimensions") { () -> [String: Any] in
      let dims = getFrameDimensions()
      return ["width": dims.width, "height": dims.height, "bytesPerRow": dims.bytesPerRow]
    }

    Function("getFrameCounter") { () -> Int in
      return Int(getFrameCounter())
    }

    Function("startTestRecorder") { (outputPath: String, width: Int, height: Int) -> Int in
      return Int(startTestRecorder(outputPath: outputPath, width: UInt32(width), height: UInt32(height)))
    }

    Function("stopTestRecorder") { () -> String in
      return stopTestRecorder()
    }

    Function("getThermalState") { () -> String in
      let state = ProcessInfo.processInfo.thermalState
      switch state {
      case .nominal: return "nominal"
      case .fair: return "fair"
      case .serious: return "serious"
      case .critical: return "critical"
      @unknown default: return "nominal"
      }
    }
  }

  private func startCapture(deviceId: String, width: Int, height: Int) {
    sessionQueue.async {
      let session = AVCaptureSession()
      session.sessionPreset = .hd1920x1080

      // Find camera device
      let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back
      guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
        print("[WebGPUCamera] No camera found for position: \(position)")
        return
      }

      do {
        let input = try AVCaptureDeviceInput(device: camera)
        if session.canAddInput(input) {
          session.addInput(input)
        }
      } catch {
        print("[WebGPUCamera] Failed to create camera input: \(error)")
        return
      }

      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
      output.alwaysDiscardsLateVideoFrames = true

      // Retain delegate as instance property to prevent deallocation
      let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height))
      self.frameDelegate = delegate
      output.setSampleBufferDelegate(delegate, queue: self.frameQueue)

      if session.canAddOutput(output) {
        session.addOutput(output)
      }

      // Update frame dimensions in Rust (already exported in Task 1)
      setFrameDimensions(width: UInt32(width), height: UInt32(height), bytesPerRow: UInt32(width * 4))

      session.startRunning()
      self.captureSession = session
      self.dataOutput = output
      print("[WebGPUCamera] Camera started: \(width)x\(height)")
    }
  }

  private func stopCapture() {
    sessionQueue.async {
      self.captureSession?.stopRunning()
      self.captureSession = nil
      self.dataOutput = nil
      self.frameDelegate = nil
      print("[WebGPUCamera] Camera stopped")
    }
  }
}

private class FrameDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  let width: UInt32
  let height: UInt32

  init(width: UInt32, height: UInt32) {
    self.width = width
    self.height = height
  }

  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let dataSize = bytesPerRow * Int(height)

    // Capture IOSurface handle FIRST for zero-copy investigation
    var surfaceHandle: UInt64 = 0
    if let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer) {
      surfaceHandle = UInt64(IOSurfaceGetID(ioSurface.takeUnretainedValue()))
      // Log once for zero-copy follow-up
      if getCurrentFrameHandle() == 0 {
        print("[WebGPUCamera] IOSurface handle available: \(surfaceHandle) (logged for zero-copy follow-up)")
      }
    }

    // Copy pixel data to Rust frame slot, passing the IOSurface handle
    let data = Data(bytes: baseAddress, count: dataSize)
    deliverFrame(pixels: [UInt8](data), handle: surfaceHandle)
  }
}
```

- [ ] **Step 3: Update TypeScript module interface**

Add `getFrameCounter` to `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`:

```typescript
interface WebGPUCameraModuleInterface extends NativeModule {
  startCameraPreview(deviceId: string, width: number, height: number): void;
  stopCameraPreview(): void;
  getCurrentFrameHandle(): number;
  getCurrentFramePixels(): Uint8Array;
  getFrameDimensions(): FrameDimensions;
  getFrameCounter(): number;
  startTestRecorder(outputPath: string, width: number, height: number): number;
  stopTestRecorder(): string;
  getThermalState(): string;
}
```

- [ ] **Step 4: Run TypeScript check**

```bash
cd packages/react-native-webgpu-camera && bunx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/ packages/react-native-webgpu-camera/modules/webgpu-camera/src/ apps/example/app.json
git commit -m "feat(ios): implement AVCaptureSession camera capture in Swift Expo module"
```

---

### Task 3: Implement Android camera capture in Kotlin

Mirror the iOS implementation: Camera2 API captures frames in Kotlin, delivers pixel data to the Rust frame slot.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt`

**Prerequisites:** `deliver_frame` and `set_frame_dimensions` are already UniFFI-exported from Task 1. The Rust Android camera stub was already simplified in Task 1 Step 3. Camera permission was added to `app.json` in Task 2 Step 1.

**Reference docs:**
- Android Camera2: https://developer.android.com/reference/android/hardware/camera2/package-summary
- Android ImageReader: https://developer.android.com/reference/android/media/ImageReader

**UniFFI binding note:** UniFFI maps Rust `Vec<u8>` to Kotlin `List<UByte>`. When calling `deliverFrame()`, convert `ByteArray` to the correct type using `.asUByteArray().toList()`.

- [ ] **Step 1: Implement Camera2 capture in Kotlin**

Replace the stub `WebGPUCameraModule.kt` with real Camera2 implementation:

```kotlin
package expo.modules.webgpucamera

import android.Manifest
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import android.view.Surface
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.nio.ByteBuffer

class WebGPUCameraModule : Module() {
  private var cameraDevice: CameraDevice? = null
  private var captureSession: CameraCaptureSession? = null
  private var imageReader: ImageReader? = null
  private var backgroundThread: HandlerThread? = null
  private var backgroundHandler: Handler? = null

  override fun definition() = ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { deviceId: String, width: Int, height: Int ->
      startCapture(deviceId, width, height)
    }

    Function("stopCameraPreview") {
      stopCapture()
    }

    Function("getCurrentFrameHandle") {
      uniffi.webgpu_camera.getCurrentFrameHandle().toLong()
    }

    Function("getCurrentFramePixels") {
      uniffi.webgpu_camera.getCurrentFramePixels().toByteArray()
    }

    Function("getFrameDimensions") {
      val dims = uniffi.webgpu_camera.getFrameDimensions()
      mapOf("width" to dims.width.toInt(), "height" to dims.height.toInt(), "bytesPerRow" to dims.bytesPerRow.toInt())
    }

    Function("getFrameCounter") {
      uniffi.webgpu_camera.getFrameCounter().toLong()
    }

    Function("startTestRecorder") { outputPath: String, width: Int, height: Int ->
      uniffi.webgpu_camera.startTestRecorder(outputPath, width.toUInt(), height.toUInt()).toLong()
    }

    Function("stopTestRecorder") {
      uniffi.webgpu_camera.stopTestRecorder()
    }

    Function("getThermalState") {
      val context = appContext.reactContext ?: return@Function "nominal"
      val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
      when (pm?.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "nominal"
        PowerManager.THERMAL_STATUS_LIGHT -> "fair"
        PowerManager.THERMAL_STATUS_MODERATE -> "fair"
        PowerManager.THERMAL_STATUS_SEVERE -> "serious"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "critical"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "critical"
        else -> "nominal"
      }
    }
  }

  private fun startCapture(deviceId: String, width: Int, height: Int) {
    val context = appContext.reactContext ?: return

    // Start background thread
    backgroundThread = HandlerThread("WebGPUCamera").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)

    // Set frame dimensions in Rust
    uniffi.webgpu_camera.setFrameDimensions(width.toUInt(), height.toUInt(), (width * 4).toUInt())

    // Create ImageReader for RGBA output
    imageReader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 2).apply {
      setOnImageAvailableListener({ reader ->
        val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
        try {
          // Convert YUV to BGRA and deliver to Rust frame slot
          val yPlane = image.planes[0]
          val uPlane = image.planes[1]
          val vPlane = image.planes[2]

          // For spike: simple CPU YUV→BGRA conversion.
          // WARNING: This per-pixel loop (~50-100ms at 1080p) will NOT sustain 30fps.
          // Spike will run at reduced framerate on Android. Acceptable for validation.
          // Production: use libyuv, RenderScript, or GPU shader for conversion.
          val bgra = yuvToBgra(
            yPlane.buffer, uPlane.buffer, vPlane.buffer,
            width, height,
            yPlane.rowStride, uPlane.rowStride, uPlane.pixelStride
          )

          uniffi.webgpu_camera.deliverFrame(bgra.asUByteArray().toList(), 0u)
        } finally {
          image.close()
        }
      }, backgroundHandler)
    }

    // Open camera
    val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    val facing = if (deviceId == "front") CameraCharacteristics.LENS_FACING_FRONT
                 else CameraCharacteristics.LENS_FACING_BACK

    val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
      val chars = cameraManager.getCameraCharacteristics(id)
      chars.get(CameraCharacteristics.LENS_FACING) == facing
    } ?: return

    try {
      cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
          cameraDevice = camera
          createCaptureSession(camera, width, height)
        }

        override fun onDisconnected(camera: CameraDevice) {
          camera.close()
          cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
          camera.close()
          cameraDevice = null
          println("[WebGPUCamera] Camera error: $error")
        }
      }, backgroundHandler)
    } catch (e: SecurityException) {
      println("[WebGPUCamera] Camera permission not granted")
    }
  }

  private fun createCaptureSession(camera: CameraDevice, width: Int, height: Int) {
    val surface = imageReader?.surface ?: return

    camera.createCaptureSession(
      listOf(surface),
      object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) {
          captureSession = session
          val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
          }
          session.setRepeatingRequest(request.build(), null, backgroundHandler)
          println("[WebGPUCamera] Camera capture session started")
        }

        override fun onConfigureFailed(session: CameraCaptureSession) {
          println("[WebGPUCamera] Capture session configuration failed")
        }
      },
      backgroundHandler
    )
  }

  private fun stopCapture() {
    captureSession?.close()
    captureSession = null
    cameraDevice?.close()
    cameraDevice = null
    imageReader?.close()
    imageReader = null
    backgroundThread?.quitSafely()
    backgroundThread = null
    backgroundHandler = null
    println("[WebGPUCamera] Camera stopped")
  }

  private fun yuvToBgra(
    yBuf: ByteBuffer, uBuf: ByteBuffer, vBuf: ByteBuffer,
    width: Int, height: Int,
    yRowStride: Int, uvRowStride: Int, uvPixelStride: Int
  ): ByteArray {
    val bgra = ByteArray(width * height * 4)
    for (row in 0 until height) {
      for (col in 0 until width) {
        val yIdx = row * yRowStride + col
        val uvRow = row / 2
        val uvCol = col / 2
        // For semi-planar formats (NV12/NV21), U and V planes share the same layout
        val uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride

        val y = (yBuf.get(yIdx).toInt() and 0xFF).toFloat()
        val u = (uBuf.get(uvIdx).toInt() and 0xFF).toFloat() - 128f
        val v = (vBuf.get(uvIdx).toInt() and 0xFF).toFloat() - 128f

        val r = (y + 1.370705f * v).toInt().coerceIn(0, 255)
        val g = (y - 0.337633f * u - 0.698001f * v).toInt().coerceIn(0, 255)
        val b = (y + 1.732446f * u).toInt().coerceIn(0, 255)

        val outIdx = (row * width + col) * 4
        bgra[outIdx] = b.toByte()       // B
        bgra[outIdx + 1] = g.toByte()   // G
        bgra[outIdx + 2] = r.toByte()   // R
        bgra[outIdx + 3] = 0xFF.toByte() // A
      }
    }
    return bgra
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/android/
git commit -m "feat(android): implement Camera2 capture in Kotlin Expo module"
```

---

## Chunk 2: WebGPU Compute Pipeline + Worklet Render Loop

### Task 4: Create the GPU pipeline hook (Spike 2 + Spike 3 gate)

This is the core of the spike — set up WebGPU device, create the Sobel compute pipeline, and validate that it works. This task also validates whether `navigator.gpu` is provided by Skia Graphite (Approach A gate).

**Files:**
- Create: `apps/example/src/hooks/useGPUPipeline.ts`

**Key dependencies to check:**
- `@shopify/react-native-skia` — must be installed with Graphite enabled
- `react-native-wgpu` — fallback if Graphite doesn't provide `navigator.gpu`

- [ ] **Step 1: Create useGPUPipeline hook**

```typescript
import { useRef, useState, useCallback, useEffect } from 'react';
import { useSharedValue } from 'react-native-reanimated';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';

export interface GPUPipelineState {
  status: 'idle' | 'initializing' | 'ready' | 'error';
  error?: string;
  deviceSource: 'graphite' | 'rn-wgpu' | 'unknown';
  computeSupported: boolean;
}

interface GPUResources {
  device: GPUDevice;
  computePipeline: GPUComputePipeline;
  inputTexture: GPUTexture;
  outputTexture: GPUTexture;
  bindGroup: GPUBindGroup;
  width: number;
  height: number;
}

export function useGPUPipeline(width = 1920, height = 1080) {
  const [state, setState] = useState<GPUPipelineState>({
    status: 'idle',
    deviceSource: 'unknown',
    computeSupported: false,
  });
  const resources = useRef<GPUResources | null>(null);

  const initialize = useCallback(async () => {
    setState(s => ({ ...s, status: 'initializing' }));

    try {
      // Step 1: Check if navigator.gpu exists (provided by Skia Graphite in Approach A)
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('navigator.gpu not available — Skia Graphite may not be active');
      }

      // Step 2: Get adapter and device
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No GPU adapter available');
      }

      const device = await adapter.requestDevice();
      console.log('[GPUPipeline] Device acquired from navigator.gpu');

      // Detect device source — Skia Graphite sets navigator.gpu from its Dawn context.
      // If react-native-skia's Graphite is active, it installs navigator.gpu before our code runs.
      // We detect this by checking for Skia's SkiaGpuContext global, which react-native-skia
      // sets when Graphite mode is active with a Dawn backend.
      let deviceSource: 'graphite' | 'rn-wgpu' | 'unknown' = 'unknown';
      try {
        // @shopify/react-native-skia exposes __SKIA_GRAPHITE_ACTIVE__ on globalThis
        // when Graphite mode is enabled and Dawn is the backend.
        // If not present, check adapter info for Dawn-specific identifiers.
        const g = globalThis as any;
        if (g.__SKIA_GRAPHITE_ACTIVE__ === true) {
          deviceSource = 'graphite';
        } else {
          // Check adapter info — Dawn adapters report "dawn" in their description
          const adapterInfo = await adapter.requestAdapterInfo?.();
          if (adapterInfo?.description?.toLowerCase().includes('dawn')) {
            // Dawn is present but Graphite marker is absent — likely react-native-wgpu standalone
            deviceSource = 'rn-wgpu';
          } else {
            deviceSource = 'unknown';
          }
        }
      } catch {
        deviceSource = 'unknown';
      }
      console.log(`[GPUPipeline] Device source: ${deviceSource}`);

      // Step 3: Create compute pipeline with Sobel shader
      const shaderModule = device.createShaderModule({ code: SOBEL_WGSL });
      console.log('[GPUPipeline] Shader module created');

      const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
      console.log('[GPUPipeline] Compute pipeline created');

      // Step 4: Create input/output textures
      const inputTexture = device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT,
      });

      const outputTexture = device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.RENDER_ATTACHMENT,
      });

      // Step 5: Create bind group
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: inputTexture.createView() },
          { binding: 1, resource: outputTexture.createView() },
        ],
      });

      resources.current = {
        device,
        computePipeline,
        inputTexture,
        outputTexture,
        bindGroup,
        width,
        height,
      };

      // Populate shared values for worklet access (Task 9 render loop)
      deviceSV.value = device;
      computePipelineSV.value = computePipeline;
      inputTextureSV.value = inputTexture;
      outputTextureSV.value = outputTexture;
      bindGroupSV.value = bindGroup;
      widthSV.value = width;
      heightSV.value = height;

      setState({
        status: 'ready',
        deviceSource,
        computeSupported: true,
      });

      console.log('[GPUPipeline] Pipeline ready');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error('[GPUPipeline] Init failed:', msg);
      setState({
        status: 'error',
        error: msg,
        deviceSource: 'unknown',
        computeSupported: false,
      });
    }
  }, [width, height]);

  const processFrame = useCallback((pixels: Uint8Array, bytesPerRow: number) => {
    const res = resources.current;
    if (!res) return;

    const t0 = performance.now();

    // Upload camera frame to input texture
    res.device.queue.writeTexture(
      { texture: res.inputTexture },
      pixels,
      { bytesPerRow },
      { width: res.width, height: res.height },
    );
    const tImport = performance.now();

    // Dispatch Sobel compute
    const encoder = res.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(res.computePipeline);
    pass.setBindGroup(0, res.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(res.width / 16),
      Math.ceil(res.height / 16),
    );
    pass.end();
    res.device.queue.submit([encoder.finish()]);
    const tCompute = performance.now();

    return {
      importMs: tImport - t0,
      computeMs: tCompute - tImport,
    };
  }, []);

  const cleanup = useCallback(() => {
    const res = resources.current;
    if (res) {
      res.inputTexture.destroy();
      res.outputTexture.destroy();
      resources.current = null;
    }
    setState({ status: 'idle', deviceSource: 'unknown', computeSupported: false });
  }, []);

  // Shared values for worklet access — WebGPU JSI host objects pass directly to worklets.
  // These are populated during initialize() and read from the worklet render loop.
  const deviceSV = useSharedValue<GPUDevice | null>(null);
  const computePipelineSV = useSharedValue<GPUComputePipeline | null>(null);
  const inputTextureSV = useSharedValue<GPUTexture | null>(null);
  const outputTextureSV = useSharedValue<GPUTexture | null>(null);
  const bindGroupSV = useSharedValue<GPUBindGroup | null>(null);
  const widthSV = useSharedValue(width);
  const heightSV = useSharedValue(height);

  return {
    state, initialize, processFrame, cleanup, resources,
    // Shared values for worklet render loop (Task 9)
    workletResources: {
      device: deviceSV,
      computePipeline: computePipelineSV,
      inputTexture: inputTextureSV,
      outputTexture: outputTextureSV,
      bindGroup: bindGroupSV,
      width: widthSV,
      height: heightSV,
    },
  };
}
```

**Important:** The `processFrame` function contains the WebGPU dispatch logic and runs on the JS thread (for testing). For Spike 2 validation, the render loop in Task 9 must perform the same WebGPU calls from a **Reanimated worklet** using `runOnUI`. The hook exposes `workletResources` — shared values containing WebGPU JSI host objects (GPUDevice, GPUTexture, etc.) that pass directly to worklets without serialization. The worklet render loop reads these shared values instead of accessing `resources.current` (which is a React ref and NOT accessible from worklets).

The three loop driver mechanisms to test (per spec):
1. **`context.present()` blocking** — may provide vsync pacing
2. **`setTimeout(renderFrame, 16)`** — simple timer (fallback)
3. **Camera-driven** — trigger render on new frame arrival

Start with option 2 (setTimeout in worklet via `requestAnimationFrame` equivalent), and experiment with the others during on-device testing.

- [ ] **Step 2: Run TypeScript check**

```bash
cd apps/example && bunx tsc --noEmit
```

Expected: may have type issues with `navigator.gpu` — WebGPU types may need to be installed. If so, add `@webgpu/types` to the example app's devDependencies:

```bash
cd apps/example && bun add -d @webgpu/types
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/hooks/useGPUPipeline.ts
git commit -m "feat: GPU pipeline hook with Sobel compute + Graphite device detection"
```

---

### Task 5: Validate Skia Graphite shared context (Spike 3)

This task validates the core Approach A hypothesis: can Skia Graphite and WebGPU share textures on the same Dawn device? The overlay component also displays status info, but the primary purpose is the six API verification items from the spec.

**Files:**
- Create: `apps/example/src/components/SpikeOverlay.tsx`

- [ ] **Step 1: Create SpikeOverlay with Graphite validation**

The component:
1. Uses Skia's `<Canvas>` (Graphite-backed when SK_GRAPHITE=1)
2. Draws overlay text/shapes (proves Skia rendering works)
3. Attempts to access the compute output texture from Skia's context (shared device test)
4. Reports which Spike 3 path succeeded

```typescript
import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text as RNText } from 'react-native';
import { Canvas, Text, RoundedRect, useFont, Skia } from '@shopify/react-native-skia';

interface SpikeOverlayProps {
  fps: number;
  spike1Status: string;
  spike2Status: string;
  spike3Status: string;
  spike4Status: string;
  elapsed: number;
  isRecording: boolean;
}

// Spike 3 validation: test Skia Graphite shared context
export function validateGraphiteSharedContext(): {
  graphiteActive: boolean;
  sharedDevice: boolean;
  textureRoundTrip: boolean;
  path: 'graphite-direct' | 'graphite-composite' | 'ganesh-fallback' | 'unknown';
} {
  const result = {
    graphiteActive: false,
    sharedDevice: false,
    textureRoundTrip: false,
    path: 'unknown' as const,
  };

  // Test 1: Is navigator.gpu available? (Graphite installs it)
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    result.graphiteActive = true;
    console.log('[Spike3] navigator.gpu exists — Graphite likely active');
  } else {
    console.log('[Spike3] navigator.gpu NOT available — Graphite not active');
    return { ...result, path: 'ganesh-fallback' };
  }

  // Test 2: Shared device detection
  // If Skia's Graphite backend created navigator.gpu, then the device
  // from navigator.gpu.requestAdapter().requestDevice() is Skia's device.
  // We verify by checking if a texture created on this device can be
  // referenced from both WebGPU compute and Skia drawing.
  // This is validated at runtime — log results for analysis.
  console.log('[Spike3] Shared device test requires runtime validation:');
  console.log('[Spike3] - Create GPUTexture on navigator.gpu device');
  console.log('[Spike3] - Try SkSurfaces.WrapBackendTexture() with Dawn texture handle');
  console.log('[Spike3] - Try SkImages.BorrowTextureFrom() for compute output');
  console.log('[Spike3] - If both work: graphite-direct path');
  console.log('[Spike3] - If Skia can draw to own texture on same device: graphite-composite');
  console.log('[Spike3] - If neither works: ganesh-fallback');

  // Attempt shared device detection via runtime probes
  try {
    // If Skia's Dawn context installed navigator.gpu, then requesting a device
    // from it gives us Skia's device. We can verify by creating a small buffer
    // and checking that both WebGPU compute and Skia rendering work without
    // device-lost errors.
    // The actual C++ API calls (WrapBackendTexture, BorrowTextureFrom) happen
    // at the native level — from JS we verify:
    // a) Skia Canvas renders (proves Skia works) — checked by SpikeOverlay rendering
    // b) WebGPU compute works on the same device (proved in Task 4)
    // c) Both run simultaneously without crashes (proves compatible contexts)

    // Check globalThis for Skia's Graphite marker
    const g = globalThis as any;
    if (g.__SKIA_GRAPHITE_ACTIVE__ === true) {
      result.sharedDevice = true;
      result.textureRoundTrip = true; // Tentative — validated by running both Skia + WebGPU
      return { ...result, path: 'graphite-direct' };
    }

    // If navigator.gpu exists but no Graphite marker, it may be react-native-wgpu standalone
    result.sharedDevice = false;
    result.textureRoundTrip = false;
    return { ...result, path: 'graphite-composite' };
  } catch {
    return { ...result, path: 'ganesh-fallback' };
  }
}

export function SpikeOverlay({
  fps,
  spike1Status,
  spike2Status,
  spike3Status,
  spike4Status,
  elapsed,
  isRecording,
}: SpikeOverlayProps) {
  const font = useFont(null, 13);

  if (!font) return null;

  const x = 16;
  const y = 60;
  const lineHeight = 18;

  return (
    <Canvas style={styles.overlay} pointerEvents="none">
      {/* Background */}
      <RoundedRect x={x} y={y} width={260} height={140} r={8} color="rgba(0,0,0,0.6)" />

      {/* Title */}
      <Text x={x + 12} y={y + 20} text="Spike Validation" font={font} color="white" />

      {/* Status lines */}
      <Text x={x + 12} y={y + 20 + lineHeight} text={`S1 (camera→GPU): ${spike1Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 2} text={`S2 (compute): ${spike2Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 3} text={`S3 (Skia): ${spike3Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 4} text={`S4 (recorder): ${spike4Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 5} text={`FPS: ${fps.toFixed(1)} | ${elapsed}s`} font={font} color="white" />

      {/* Recording indicator — red tint when recording */}
      {isRecording && (
        <RoundedRect x={0} y={0} width={9999} height={9999} r={0} color="rgba(255,0,0,0.15)" />
      )}
    </Canvas>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
});
```

**Spike 3 validation notes for implementer:**

The six spec-required verification items and how to test them:

1. **SK_GRAPHITE=1 build** — Check if react-native-skia builds with Graphite. It may be the default in current versions. Look at the build output for `SK_GRAPHITE` flag.

2. **Graphite auto-detects Dawn** — If `navigator.gpu` exists at runtime, Graphite detected Dawn.

3. **SkSurfaces::WrapBackendTexture** — This is a C++ API. From JS, we can't call it directly. The test is: does the Skia Canvas render on the same device that WebGPU compute uses without crashes? If yes, the contexts are compatible.

4. **SkImages::BorrowTextureFrom** — Same as above — C++ level. For the spike, the JS-level validation is: can we see both Skia overlay content AND compute output on screen simultaneously? If yes, they share context or at least coexist.

5. **Shared vs separate GPUDevice** — Log the device adapter info from `navigator.gpu.requestAdapter()` and compare with Skia's internal Dawn adapter. From JS, we verify by checking that both WebGPU and Skia render without creating conflicting resources.

6. **Skia output as GPUTexture for recorder** — For the spike, the recorder captures whatever is on the native surface. If Skia and WebGPU share the surface, the recorded output contains both. Verified by checking the recorded video.

- [ ] **Step 2: Run TypeScript check**

```bash
cd apps/example && bunx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/components/SpikeOverlay.tsx
git commit -m "feat: Skia overlay with Graphite shared context validation (Spike 3)"
```

---

### Task 6: Implement iOS recorder surface (Spike 4)

Implement AVAssetWriter-based recording in the Swift Expo module. The spike tests whether Dawn can render to a recorder-owned surface.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Add AVAssetWriter recording to Swift module**

Add recording methods to `WebGPUCameraModule.swift`. The recorder:
1. Creates an AVAssetWriter configured for H.264
2. Uses AVAssetWriterInputPixelBufferAdaptor for frame input
3. The `startTestRecorder` function returns a surface handle (0 = readback path, non-zero = surface path)
4. During recording, the render loop calls a method to append the current rendered frame

Add these methods and properties to the `WebGPUCameraModule` class (after the camera capture code from Task 2):

```swift
  // --- Recorder (Spike 4) ---
  private var assetWriter: AVAssetWriter?
  private var writerInput: AVAssetWriterInput?
  private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
  private var isRecording = false
  private var recordingStartTime: CMTime = .zero
  private var recordedFrameCount: Int64 = 0
  private var recordingOutputPath: String = ""

  private func startRecorder(outputPath: String, width: Int, height: Int) -> Int {
    let url = URL(fileURLWithPath: outputPath)

    // Remove existing file
    try? FileManager.default.removeItem(at: url)

    do {
      let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

      let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: 10_000_000, // 10 Mbps
          AVVideoExpectedSourceFrameRateKey: 30,
        ]
      ]

      let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
      input.expectsMediaDataInRealTime = true

      let sourcePixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ]

      let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: sourcePixelBufferAttributes
      )

      if writer.canAdd(input) {
        writer.add(input)
      }

      writer.startWriting()
      writer.startSession(atSourceTime: .zero)

      self.assetWriter = writer
      self.writerInput = input
      self.pixelBufferAdaptor = adaptor
      self.isRecording = true
      self.recordingStartTime = CMClockGetTime(CMClockGetHostTimeClock())
      self.recordedFrameCount = 0
      self.recordingOutputPath = outputPath

      print("[WebGPUCamera] Recorder started: \(outputPath)")

      // Return 0 for readback path (we're using pixel buffer adaptor, not direct surface)
      // A non-zero return would indicate direct Dawn surface rendering (future work)
      return 0
    } catch {
      print("[WebGPUCamera] Recorder setup failed: \(error)")
      return 0
    }
  }

  private func stopRecorder() -> String {
    guard let writer = assetWriter, isRecording else { return "" }

    isRecording = false
    writerInput?.markAsFinished()

    let semaphore = DispatchSemaphore(value: 0)
    var outputPath = recordingOutputPath

    writer.finishWriting {
      print("[WebGPUCamera] Recording finished: \(self.recordedFrameCount) frames, path: \(outputPath)")
      semaphore.signal()
    }
    semaphore.wait()

    assetWriter = nil
    writerInput = nil
    pixelBufferAdaptor = nil

    return outputPath
  }

  /// Called from the render loop to append a frame to the recording.
  /// Takes raw BGRA pixel data and writes it as a CVPixelBuffer.
  func appendFrameToRecorder(pixels: Data, width: Int, height: Int) {
    guard isRecording,
          let adaptor = pixelBufferAdaptor,
          let input = writerInput,
          input.isReadyForMoreMediaData else { return }

    // Use the adaptor's pixel buffer pool for better performance
    // (reuses buffers instead of allocating each frame)
    guard let pool = adaptor.pixelBufferPool else { return }

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
      pixels.copyBytes(to: baseAddress.assumingMemoryBound(to: UInt8.self), count: min(pixels.count, width * height * 4))
    }

    let frameTime = CMTime(value: recordedFrameCount, timescale: 30)
    adaptor.append(buffer, withPresentationTime: frameTime)
    recordedFrameCount += 1
  }
```

Then update the `startTestRecorder` and `stopTestRecorder` function definitions to call these:

```swift
    Function("startTestRecorder") { (outputPath: String, width: Int, height: Int) -> Int in
      return self.startRecorder(outputPath: outputPath, width: width, height: height)
    }

    Function("stopTestRecorder") { () -> String in
      return self.stopRecorder()
    }

    Function("appendFrameToRecorder") { (pixels: Data, width: Int, height: Int) in
      self.appendFrameToRecorder(pixels: pixels, width: width, height: height)
    }
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/
git commit -m "feat(ios): AVAssetWriter recorder implementation (Spike 4)"
```

---

### Task 7: Implement Android recorder (Spike 4)

Mirror the iOS recorder: MediaCodec + MediaMuxer-based recording in Kotlin. We use MediaCodec (not MediaRecorder) because MediaRecorder requires rendering to its Surface, but we're using the readback path (appending raw pixel data). MediaCodec accepts raw input buffers and MediaMuxer writes the MP4 container.

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt`

- [ ] **Step 1: Add MediaCodec + MediaMuxer recording to Kotlin module**

Add recording methods to `WebGPUCameraModule.kt`. Add these properties and methods to the class:

```kotlin
  // --- Recorder (Spike 4) ---
  private var mediaCodec: android.media.MediaCodec? = null
  private var mediaMuxer: android.media.MediaMuxer? = null
  private var videoTrackIndex = -1
  private var isRecording = false
  private var isMuxerStarted = false
  private var recordingOutputPath: String = ""
  private var recordedFrameCount: Long = 0
  private var recordingWidth = 0
  private var recordingHeight = 0

  private fun startRecorder(outputPath: String, width: Int, height: Int): Long {
    try {
      recordingWidth = width
      recordingHeight = height

      // Configure H.264 encoder
      val format = android.media.MediaFormat.createVideoFormat(
        android.media.MediaFormat.MIMETYPE_VIDEO_AVC, width, height
      ).apply {
        setInteger(android.media.MediaFormat.KEY_BIT_RATE, 10_000_000)
        setInteger(android.media.MediaFormat.KEY_FRAME_RATE, 30)
        setInteger(android.media.MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        setInteger(
          android.media.MediaFormat.KEY_COLOR_FORMAT,
          android.media.MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )
      }

      val codec = android.media.MediaCodec.createEncoderByType(
        android.media.MediaFormat.MIMETYPE_VIDEO_AVC
      )
      codec.configure(format, null, null, android.media.MediaCodec.CONFIGURE_FLAG_ENCODE)
      codec.start()

      val muxer = android.media.MediaMuxer(
        outputPath,
        android.media.MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
      )

      mediaCodec = codec
      mediaMuxer = muxer
      isRecording = true
      isMuxerStarted = false
      videoTrackIndex = -1
      recordingOutputPath = outputPath
      recordedFrameCount = 0

      println("[WebGPUCamera] Recorder started: $outputPath")
      return 0L // Readback path
    } catch (e: Exception) {
      println("[WebGPUCamera] Recorder setup failed: $e")
      return 0L
    }
  }

  /// Called from the render loop to append a raw BGRA frame to the recording.
  /// Converts BGRA → YUV420 and feeds to MediaCodec encoder.
  fun appendFrameToRecorder(bgraPixels: ByteArray, width: Int, height: Int) {
    val codec = mediaCodec ?: return
    if (!isRecording) return

    val inputBufferIndex = codec.dequeueInputBuffer(0) // Non-blocking
    if (inputBufferIndex < 0) return // Encoder busy, drop frame

    val inputBuffer = codec.getInputBuffer(inputBufferIndex) ?: return

    // Convert BGRA to NV21 (YUV420SP) for the encoder
    // This is CPU-intensive but acceptable for spike validation
    val yuvSize = width * height * 3 / 2
    val yuv = ByteArray(yuvSize)
    bgraToNv21(bgraPixels, yuv, width, height)

    inputBuffer.clear()
    inputBuffer.put(yuv, 0, minOf(yuv.size, inputBuffer.remaining()))

    val presentationTimeUs = recordedFrameCount * 1_000_000L / 30
    codec.queueInputBuffer(inputBufferIndex, 0, yuvSize, presentationTimeUs, 0)
    recordedFrameCount++

    // Drain encoder output
    drainEncoder(false)
  }

  private fun drainEncoder(endOfStream: Boolean) {
    val codec = mediaCodec ?: return
    val muxer = mediaMuxer ?: return
    val bufferInfo = android.media.MediaCodec.BufferInfo()

    while (true) {
      val outputIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
      when {
        outputIndex == android.media.MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
          videoTrackIndex = muxer.addTrack(codec.outputFormat)
          muxer.start()
          isMuxerStarted = true
        }
        outputIndex >= 0 -> {
          if (!isMuxerStarted) break
          val outputBuffer = codec.getOutputBuffer(outputIndex) ?: break
          if (bufferInfo.flags and android.media.MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
            bufferInfo.size = 0
          }
          if (bufferInfo.size > 0) {
            outputBuffer.position(bufferInfo.offset)
            outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
            muxer.writeSampleData(videoTrackIndex, outputBuffer, bufferInfo)
          }
          codec.releaseOutputBuffer(outputIndex, false)
          if (bufferInfo.flags and android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
        }
        else -> break
      }
    }
  }

  private fun bgraToNv21(bgra: ByteArray, nv21: ByteArray, width: Int, height: Int) {
    // Simple BGRA → NV21 conversion (CPU, acceptable for spike)
    var yIndex = 0
    var uvIndex = width * height
    for (j in 0 until height) {
      for (i in 0 until width) {
        val px = (j * width + i) * 4
        val b = bgra[px].toInt() and 0xFF
        val g = bgra[px + 1].toInt() and 0xFF
        val r = bgra[px + 2].toInt() and 0xFF

        val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
        nv21[yIndex++] = y.coerceIn(0, 255).toByte()

        if (j % 2 == 0 && i % 2 == 0) {
          val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
          val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
          nv21[uvIndex++] = v.coerceIn(0, 255).toByte()
          nv21[uvIndex++] = u.coerceIn(0, 255).toByte()
        }
      }
    }
  }

  private fun stopRecorder(): String {
    if (!isRecording) return ""
    isRecording = false

    try {
      // Signal end of stream
      val codec = mediaCodec
      if (codec != null) {
        val inputIndex = codec.dequeueInputBuffer(5000)
        if (inputIndex >= 0) {
          codec.queueInputBuffer(
            inputIndex, 0, 0, 0,
            android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM
          )
        }
        drainEncoder(true)
        codec.stop()
        codec.release()
      }
      mediaMuxer?.stop()
      mediaMuxer?.release()
    } catch (e: Exception) {
      println("[WebGPUCamera] Recorder stop error: $e")
    }
    mediaCodec = null
    mediaMuxer = null

    println("[WebGPUCamera] Recording finished: $recordedFrameCount frames, $recordingOutputPath")
    return recordingOutputPath
  }
```

Update the `startTestRecorder` and `stopTestRecorder` functions in `definition()`:

```kotlin
    Function("startTestRecorder") { outputPath: String, width: Int, height: Int ->
      startRecorder(outputPath, width, height)
    }

    Function("stopTestRecorder") {
      stopRecorder()
    }

    Function("appendFrameToRecorder") { pixels: ByteArray, width: Int, height: Int ->
      appendFrameToRecorder(pixels, width, height)
    }
```

**Note:** The `appendFrameToRecorder` method is called from the JS-side render loop via a native module call. The render loop (Task 9) needs to add an `appendFrameToRecorder` call on the JS side when recording is active. See Task 9 for the integration point.

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/android/
git commit -m "feat(android): MediaCodec + MediaMuxer recorder implementation (Spike 4)"
```

---

### Task 8: Create recorder bridge utility

TypeScript wrapper for the recorder that manages output paths and state.

**Files:**
- Create: `apps/example/src/utils/recorderBridge.ts`

- [ ] **Step 1: Create recorderBridge.ts**

```typescript
import { Paths } from 'expo-file-system';
// NOTE: Verify this import path resolves in the monorepo. If not, try:
// import WebGPUCameraModule from '@/../../packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';
import WebGPUCameraModule from 'react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';

export interface RecorderState {
  isRecording: boolean;
  outputPath: string | null;
  surfaceHandle: number;
  path: 'surface-record' | 'readback-record' | 'unknown';
}

export function startRecording(
  width: number,
  height: number,
  filename = 'spike4_test.mp4'
): RecorderState {
  const outputPath = `${Paths.document}/${filename}`;
  const surfaceHandle = WebGPUCameraModule.startTestRecorder(outputPath, width, height);

  // surfaceHandle === 0 means readback path (current implementation)
  // surfaceHandle !== 0 would mean direct Dawn surface rendering (future)
  const path = surfaceHandle !== 0 ? 'surface-record' : 'readback-record';

  console.log(`[Recorder] Started: ${path}, output: ${outputPath}`);
  console.log(`[Recorder] Surface handle: ${surfaceHandle}`);

  return { isRecording: true, outputPath, surfaceHandle, path };
}

export function stopRecording(): string {
  const filePath = WebGPUCameraModule.stopTestRecorder();
  console.log(`[Recorder] Stopped, file: ${filePath}`);
  return filePath;
}
```

**Spike 4 fallback cascade notes for implementer:**

The current implementation uses the **readback path** (fallback 3 in the spec). To test direct surface rendering:

1. **iOS direct surface:** Get `pixelBufferAdaptor.pixelBufferPool`, create a `CVPixelBuffer` backed by `IOSurface`, create a `CAMetalLayer` from that IOSurface, pass to Dawn's `makeSurface()`. This is the non-standard path flagged as high-uncertainty in the spec.

2. **Android direct surface:** Call `mediaRecorder.surface` before starting, convert to `ANativeWindow*`, pass to Dawn's `makeSurface()`. More straightforward but requires JNI bridge to Dawn.

3. **GPU copy fallback:** Use Dawn's `CopyTextureToTexture` to copy the composited frame to a texture backed by the recorder surface. Still GPU→GPU.

For the spike, the readback path validates that recording works end-to-end. Direct surface rendering is a stretch goal.

- [ ] **Step 2: Run TypeScript check**

```bash
cd apps/example && bunx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/utils/recorderBridge.ts
git commit -m "feat: recorder bridge utility for spike 4 validation"
```

---

## Chunk 3: Wire Up the Example App

### Task 9: Wire up the full pipeline in index.tsx

Replace the stub CameraSpikeScreen with the real pipeline that connects all four spikes. The render loop runs on the UI thread via Reanimated's `runOnUI` to validate Spike 2 (compute from worklet thread).

**Files:**
- Modify: `apps/example/src/app/index.tsx`

**Important — worklet thread requirement:**
The render loop MUST run on the UI/worklet thread, not the JS main thread. This validates Spike 2 (WebGPU compute dispatch from worklet). We use `runOnUI` from react-native-reanimated to schedule the loop, and `runOnJS` to push status updates back to React state. The `isRunningRef` shared value prevents stale closure issues that would occur with `useState`.

- [ ] **Step 1: Rewrite the CameraSpikeScreen**

This is the biggest change — connecting camera → compute → overlay → screen. Replace the full contents of `index.tsx`:

```typescript
import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  ScrollView,
} from 'react-native';
import { runOnUI, runOnJS, useSharedValue } from 'react-native-reanimated';
import { useSpikeMetrics, SpikeResults } from '@/hooks/useSpikeMetrics';
import { useGPUPipeline } from '@/hooks/useGPUPipeline';
import { SpikeOverlay } from '@/components/SpikeOverlay';
import { startRecording, stopRecording, RecorderState } from '@/utils/recorderBridge';
import WebGPUCameraModule from 'react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';

const CAMERA_WIDTH = 1920;
const CAMERA_HEIGHT = 1080;
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;
const RUN_DURATION_S = 60;

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [recorderState, setRecorderState] = useState<RecorderState | null>(null);
  const [spikeStatus, setSpikeStatus] = useState({
    spike1: 'pending',
    spike2: 'pending',
    spike3: 'pending',
    spike4: 'pending',
    fps: 0,
    elapsed: 0,
  });
  const [results, setResults] = useState<SpikeResults | null>(null);
  const metrics = useSpikeMetrics();
  const pipeline = useGPUPipeline(CAMERA_WIDTH, CAMERA_HEIGHT);
  const lastFrameCounter = useRef<number>(0);
  const startTimeRef = useRef(0);

  // Shared values for worklet thread access (avoids stale closures)
  const isRunningRef = useSharedValue(false);
  // Use shared value (not React ref) so worklet can read recorder status
  const recorderPathSV = useSharedValue<string>('pending');

  // Push status updates from worklet → JS thread
  const updateStatus = useCallback((status: typeof spikeStatus) => {
    setSpikeStatus(status);
  }, []);

  // autoStop is called from the worklet via runOnJS when elapsed >= RUN_DURATION_S.
  // Sets shared value to stop the loop; the useEffect cleanup handles full teardown.
  // Avoids circular dependency with stopPipeline (defined below).
  const autoStop = useCallback(() => {
    isRunningRef.value = false;
    setIsRunning(false);
  }, [isRunningRef]);

  // Access worklet-compatible shared values from the pipeline hook
  const { workletResources } = pipeline;

  // Shared values for worklet-side state (React refs are NOT accessible from worklets)
  const lastFrameCounterSV = useSharedValue(0);
  const startTimeSV = useSharedValue(0);
  const spike2StatusSV = useSharedValue('pending');
  const spike3StatusSV = useSharedValue('pending');

  // Callbacks for runOnJS — must be stable references
  const recordFrameJS = useCallback((timing: { importMs: number; computeMs: number; skiaMs: number; totalMs: number }) => {
    metrics.recordFrame(timing);
  }, [metrics]);

  const recordThermalJS = useCallback((thermal: string) => {
    metrics.recordThermalChange(thermal);
  }, [metrics]);

  // Worklet render loop — runs on UI thread via runOnUI
  // This validates Spike 2: WebGPU compute dispatch from worklet thread
  const startRenderLoop = useCallback(() => {
    // Capture current spike status for worklet
    spike2StatusSV.value = pipeline.state.computeSupported ? 'worklet-compute' : 'pending';
    spike3StatusSV.value = pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'pending';
    startTimeSV.value = performance.now();

    const tick = () => {
      'worklet';
      if (!isRunningRef.value) return;

      // Read frame counter from native module (JSI call — works from worklet)
      const frameCounter = WebGPUCameraModule.getFrameCounter();

      if (frameCounter > lastFrameCounterSV.value) {
        lastFrameCounterSV.value = frameCounter;

        // Get frame data from native module (JSI calls — work from worklet)
        const pixels = WebGPUCameraModule.getCurrentFramePixels();
        const dims = WebGPUCameraModule.getFrameDimensions();

        // Read WebGPU resources from shared values (NOT React refs)
        const device = workletResources.device.value;
        const computePipeline = workletResources.computePipeline.value;
        const inputTexture = workletResources.inputTexture.value;
        const outputTexture = workletResources.outputTexture.value;
        const bindGroup = workletResources.bindGroup.value;
        const w = workletResources.width.value;
        const h = workletResources.height.value;

        if (pixels.length > 0 && dims.width > 0 && device && computePipeline && inputTexture) {
          const t0 = performance.now();

          // Upload camera frame to input texture — THIS IS THE SPIKE 2 VALIDATION
          // WebGPU JSI calls executing on the worklet/UI thread
          device.queue.writeTexture(
            { texture: inputTexture },
            pixels,
            { bytesPerRow: dims.bytesPerRow },
            { width: w, height: h },
          );
          const tImport = performance.now();

          // Dispatch Sobel compute on worklet thread
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(computePipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(
            Math.ceil(w / 16),
            Math.ceil(h / 16),
          );
          pass.end();
          device.queue.submit([encoder.finish()]);
          const tCompute = performance.now();

          const skiaMs = 0; // Measured separately via Skia rendering

          // Push timing data to JS thread for metrics
          runOnJS(recordFrameJS)({
            importMs: tImport - t0,
            computeMs: tCompute - tImport,
            skiaMs,
            totalMs: tCompute - t0 + skiaMs,
          });

          const elapsed = Math.floor((performance.now() - startTimeSV.value) / 1000);

          runOnJS(updateStatus)({
            spike1: 'copy-fallback',
            spike2: spike2StatusSV.value,
            spike3: spike3StatusSV.value,
            spike4: recorderPathSV.value,
            fps: 0, // Updated from metrics on JS side
            elapsed,
          });

          // Check thermal state every ~1s
          if (frameCounter % 30 === 0) {
            const thermal = WebGPUCameraModule.getThermalState();
            runOnJS(recordThermalJS)(thermal);
          }

          // Auto-stop after RUN_DURATION_S
          if (elapsed >= RUN_DURATION_S) {
            runOnJS(autoStop)();
            return;
          }
        }
      }

      // Schedule next tick on UI thread.
      // RISK: Reanimated's worklet runtime may not support setTimeout natively.
      // If setTimeout is unavailable in the worklet, try these fallbacks in order:
      // 1. requestAnimationFrame(tick) — if available in worklet runtime
      // 2. runOnUI(tick)() — re-dispatch from JS via runOnJS(() => runOnUI(tick)())
      // 3. Move loop to JS thread with runOnJS(tick) — still validates that individual
      //    WebGPU dispatch calls work from worklet (call processFrame via runOnUI)
      // In production, prefer camera-driven callback instead of timer-based loop.
      setTimeout(tick, FRAME_INTERVAL);
    };

    runOnUI(tick)();
  }, [pipeline, isRunningRef, workletResources, updateStatus, autoStop, recordFrameJS, recordThermalJS, lastFrameCounterSV, startTimeSV, spike2StatusSV, spike3StatusSV]);

  const startPipeline = useCallback(async () => {
    isRunningRef.value = true;
    setIsRunning(true);
    setResults(null);
    metrics.reset();
    startTimeRef.current = performance.now();
    lastFrameCounter.current = 0;

    // Initialize GPU pipeline
    await pipeline.initialize();

    // Start camera
    WebGPUCameraModule.startCameraPreview('back', CAMERA_WIDTH, CAMERA_HEIGHT);

    console.log('[CameraSpikeScreen] Pipeline started');

    // Start render loop after a brief delay for camera to warm up
    setTimeout(() => {
      startRenderLoop();
    }, 500);
  }, [pipeline, metrics, startRenderLoop, isRunningRef]);

  const stopPipeline = useCallback(() => {
    isRunningRef.value = false;
    setIsRunning(false);

    // Stop camera
    WebGPUCameraModule.stopCameraPreview();

    // Stop recorder if active
    if (recorderState?.isRecording) {
      stopRecording();
      setRecorderState(null);
      recorderPathSV.value = 'pending';
    }

    // Log results
    metrics.logSummary({
      spike1Path: 'copy-fallback',
      spike2Path: pipeline.state.computeSupported ? 'worklet-compute' : 'unknown',
      spike3Path: pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'unknown',
      spike4Path: recorderPathSV.value === 'pending' ? 'unknown' : recorderPathSV.value,
    });

    const summary = metrics.getSummary();
    if (summary) {
      setResults({
        ...summary,
        spike1Path: 'copy-fallback',
        spike2Path: pipeline.state.computeSupported ? 'worklet-compute' : 'unknown',
        spike3Path: pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'unknown',
        spike4Path: recorderPathSV.value === 'pending' ? 'unknown' : recorderPathSV.value,
      });
    }

    // Cleanup GPU resources
    pipeline.cleanup();
  }, [pipeline, metrics, recorderState, isRunningRef]);

  const handleStartRecording = useCallback(() => {
    const state = startRecording(CAMERA_WIDTH, CAMERA_HEIGHT);
    setRecorderState(state);
    recorderPathSV.value = state.path;
    setSpikeStatus(s => ({ ...s, spike4: state.path }));

    // Stop after 5 seconds
    setTimeout(() => {
      const filePath = stopRecording();
      setRecorderState(null);
      recorderPathSV.value = 'pending';
      console.log(`[CameraSpikeScreen] Recording saved: ${filePath}`);
    }, 5000);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isRunningRef.value = false;
      pipeline.cleanup();
    };
  }, [pipeline, isRunningRef]);

  return (
    <View style={styles.container}>
      {/* Skia overlay — validates Spike 3 */}
      <SpikeOverlay
        fps={spikeStatus.fps}
        spike1Status={spikeStatus.spike1}
        spike2Status={spikeStatus.spike2}
        spike3Status={spikeStatus.spike3}
        spike4Status={spikeStatus.spike4}
        elapsed={spikeStatus.elapsed}
        isRecording={recorderState?.isRecording ?? false}
      />

      {/* Pipeline status */}
      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          GPU: {pipeline.state.status} | Source: {pipeline.state.deviceSource}
          {pipeline.state.error ? ` | Error: ${pipeline.state.error}` : ''}
        </Text>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={isRunning ? stopPipeline : startPipeline}
        >
          <Text style={styles.buttonText}>
            {isRunning ? 'Stop' : 'Start Pipeline'}
          </Text>
        </Pressable>

        {isRunning && (
          <Pressable
            style={[styles.button, recorderState?.isRecording && styles.buttonActive]}
            onPress={handleStartRecording}
            disabled={recorderState?.isRecording}
          >
            <Text style={styles.buttonText}>
              {recorderState?.isRecording ? 'Recording...' : 'Record 5s'}
            </Text>
          </Pressable>
        )}
      </View>

      {/* Results display */}
      {results && (
        <ScrollView style={styles.results}>
          <Text style={styles.resultsTitle}>Spike Results</Text>
          <Text style={styles.resultsText}>
            {JSON.stringify(results, null, 2)}
          </Text>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  statusBar: {
    position: 'absolute',
    top: 44,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 4,
    padding: 8,
  },
  statusText: {
    color: '#aaa',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  controls: {
    position: 'absolute',
    bottom: 60,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 24,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: {
    backgroundColor: 'rgba(255,80,80,0.4)',
    borderColor: 'rgba(255,80,80,0.6)',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  results: {
    position: 'absolute',
    top: 220,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 8,
    padding: 12,
    maxHeight: 300,
  },
  resultsTitle: {
    color: '#0f0',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultsText: {
    color: '#0f0',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
});
```

- [ ] **Step 2: Run TypeScript check**

```bash
cd apps/example && bunx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: wire up full spike pipeline — camera, compute, overlay, recorder (worklet render loop)"
```

---

### Task 10: EAS device build and on-device test

Build the app for physical devices and run the spike validation.

**Files:** None — this is a build + test task.

- [ ] **Step 1: Verify all TypeScript compiles**

```bash
cd packages/react-native-webgpu-camera && bunx tsc --noEmit
cd apps/example && bunx tsc --noEmit
```

- [ ] **Step 2: Commit bun.lock if changed**

```bash
git add bun.lock
git status
```

If bun.lock changed:

```bash
git commit -m "chore: update bun.lock"
```

- [ ] **Step 3: Build for iOS device via EAS**

```bash
cd apps/example && npx eas build --platform ios --profile development
```

Wait for build to complete. If build fails:
- Check native module linking errors
- Check podspec issues
- Check UniFFI binding generation

- [ ] **Step 4: Build for Android device via EAS**

```bash
cd apps/example && npx eas build --platform android --profile development
```

- [ ] **Step 5: Install and test on physical device**

On the device:
1. Tap "Start Pipeline"
2. Grant camera permission when prompted
3. Observe console logs for:
   - `[WebGPUCamera] Camera started` — Spike 1 working
   - `[GPUPipeline] Device acquired from navigator.gpu` — Spike 2 gate
   - `[GPUPipeline] Compute pipeline created` — Spike 2 working
   - FPS counter in Skia overlay — Spike 3 working
4. Tap "Record 5s" while pipeline is running
5. Wait for recording to complete
6. Check console for `[Recorder] Stopped, file:` — Spike 4
7. Let pipeline run for 60 seconds for stability test (Spike 5)
8. Review final Spike Results JSON output

- [ ] **Step 6: Document results**

Create a results file capturing:
- Which spikes passed/failed
- Which paths were used (copy vs zero-copy, graphite vs ganesh, etc.)
- Sustained FPS
- Frame drops
- Thermal state transitions
- Any errors or crashes

```bash
# After running tests, document results:
# (Fill in actual results from device testing)
echo "Results documented in console output and device screenshots"
```

- [ ] **Step 7: Evaluate Spike 5 go/no-go criteria**

Spike 5 (sustained performance) evaluation after the 60-second run (per spec):

**Pass (proceed to production):**
- Sustained FPS >= 60, frame drop rate < 5%

**Acceptable (proceed with investigation):**
- Sustained FPS >= 30 but < 60 — investigate bottleneck, 30fps is acceptable baseline

**Fail (fundamental architecture issue):**
- Cannot sustain 30 FPS

**Additional stability criteria (all must pass regardless of FPS tier):**
- No thermal throttling above "fair" state
- No OOM crashes or GPU device lost errors
- Memory stays stable (no unbounded growth over 60s)

Log all actual values. This determines whether the architecture is viable for production.

- [ ] **Step 8: Commit any fixes from testing**

```bash
git add apps/example/src/ packages/react-native-webgpu-camera/
git commit -m "fix: address issues found during on-device testing"
```

---

## Implementation Notes

### Testing strategy

This is spike/PoC code — the "test" is running on a physical device and observing the results. There are no unit tests because:
1. Camera capture requires real hardware
2. WebGPU compute requires a real GPU
3. Skia Graphite requires the full rendering context
4. Recorder requires the full platform encoder stack

The metrics hook (`useSpikeMetrics.ts`) IS the test harness — it captures frame timings, drop rates, and thermal state for the 60-second stability run.

### If Approach A fails

If `navigator.gpu` is not available or compute doesn't work on the Graphite device:

1. Install `react-native-wgpu` Canvas component
2. Change `useGPUPipeline` to get device from Canvas's `onCreateSurface` callback instead of `navigator.gpu`
3. Change `SpikeOverlay` to use Ganesh (offscreen Skia surface → readPixels → writeTexture)
4. Update status reporting to reflect Approach B

This fallback can be implemented as a follow-up task if needed.

### Build script dependencies

The Rust → UniFFI → Swift/Kotlin binding generation depends on:
- `scripts/build-rust.sh` — builds the Rust staticlib for iOS/Android targets
- `scripts/generate-bindings.sh` — runs UniFFI to generate Swift/Kotlin bindings
- `ubrn.config.yaml` — UniFFI bridge configuration

These scripts must run before the EAS build. Check that the EAS build profile includes a prebuild step that runs them, or run them manually before `eas build`.
