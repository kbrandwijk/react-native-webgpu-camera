# Codec & Color Space Selection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded camera resolution/fps with a format enumeration + selection API, and add color space control.

**Architecture:** New `useCameraFormats` hook queries native for available `AVCaptureDevice.Format`s. `useCamera` accepts an optional format object (with opaque `nativeHandle`) and `colorSpace` string. Native stores format references per-device, activates by index. Default: 1080p/30fps/sRGB.

**Tech Stack:** TypeScript, Swift (AVFoundation), Expo Modules

**Spec:** `docs/superpowers/specs/2026-03-15-codec-colorspace-selection-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/react-native-webgpu-camera/src/types.ts` | Modify | Add `CameraFormat`, `ColorSpace`, `StabilizationMode` types; update `CameraConfig` |
| `packages/react-native-webgpu-camera/src/useCameraFormats.ts` | Create | New hook: queries native for formats, returns `CameraFormat[]` |
| `packages/react-native-webgpu-camera/src/useCamera.ts` | Modify | Accept `format` + `colorSpace`, derive width/height/fps from format, pass `nativeHandle` to native |
| `packages/react-native-webgpu-camera/src/index.ts` | Modify | Export new hook + types |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts` | Modify | Add `getFormats` to native module interface, update `startCameraPreview` signature |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Modify | Add `getFormats()` function, update `startCapture()` to accept `nativeHandle` + `colorSpace`, add default selection logic |
| `apps/example/src/app/index.tsx` | Modify | Update to use new API |

---

## Chunk 1: Types and Native Module Interface

### Task 1: Add TypeScript types

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/types.ts`

- [ ] **Step 1: Add format and color space types to types.ts**

Add after the `CameraHandle` interface (after line 20):

```ts
// Format enumeration

export type ColorSpace = 'sRGB' | 'p3D65' | 'hlgBT2020' | 'appleLog';
export type StabilizationMode = 'off' | 'standard' | 'cinematic' | 'cinematicExtended';

export interface CameraFormat {
  width: number;
  height: number;
  minFps: number;
  maxFps: number;
  pixelFormat: 'yuv420' | 'yuv422' | 'yuv444' | 'bgra';
  supportedColorSpaces: ColorSpace[];
  isHDR: boolean;
  stabilizationModes: StabilizationMode[];
  fieldOfView: number;
  isBinned: boolean;
  isMultiCamSupported: boolean;
  /** Opaque index into native format array — pass back to useCamera */
  nativeHandle: number;
}
```

- [ ] **Step 2: Update CameraConfig to use format + colorSpace**

Replace the existing `CameraConfig` interface (lines 6-11) with:

```ts
export interface CameraConfig {
  device: 'back' | 'front';
  /** Selected format from useCameraFormats(). Default: best 1080p/30fps match. */
  format?: CameraFormat;
  /** Color space to activate. Default: 'sRGB'. Must be in format.supportedColorSpaces. */
  colorSpace?: ColorSpace;
}
```

- [ ] **Step 3: Verify types compile**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`

This will show errors in `useCamera.ts` and `useGPUFrameProcessor.ts` because they still reference `width`/`height`/`fps` on config — that's expected. The types themselves should be valid.

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/src/types.ts
git commit -m "feat: add CameraFormat, ColorSpace types and update CameraConfig"
```

---

### Task 2: Update native module TypeScript interface

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`

- [ ] **Step 1: Add getFormats and update startCameraPreview signature**

The current interface (lines 3-18) has `startCameraPreview(deviceId, width, height, fps)`. Update to:

```ts
interface WebGPUCameraModuleInterface extends NativeModule {
  getFormats(deviceId: string): Record<string, any>[];
  startCameraPreview(
    deviceId: string,
    nativeHandle: number,
    colorSpace: string,
  ): void;
  stopCameraPreview(): void;
  getThermalState(): string;
  setupMultiPassPipeline(config: {
    shaders: string[];
    width: number;
    height: number;
    buffers: [number, number, number][];
    useCanvas: boolean;
    sync: boolean;
  }): boolean;
  getActiveFormatInfo(): { width: number; height: number; fps: number } | null;
  cleanupComputePipeline(): void;
  isComputeReady(): boolean;
}
```

Key changes:
- Added `getFormats(deviceId)` — returns array of dictionaries
- `startCameraPreview` now takes `(deviceId, nativeHandle, colorSpace)` instead of `(deviceId, width, height, fps)`
- Added `getActiveFormatInfo()` — returns resolved dimensions after camera starts (used by `useCamera` when no format specified)

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts
git commit -m "feat: add getFormats and update startCameraPreview in native module interface"
```

---

## Chunk 2: Native Implementation (Swift)

### Task 3: Add getFormats() to Swift module

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Add stored format arrays and getFormats function**

Add instance properties after line 14 (after `var computeSetup: Bool = false`):

```swift
/// Stored AVCaptureDevice.Format arrays, keyed by device position.
/// Rebuilt on each getFormats() call. nativeHandle indexes into these.
var storedBackFormats: [AVCaptureDevice.Format] = []
var storedFrontFormats: [AVCaptureDevice.Format] = []
```

Add the `getFormats` function definition inside the module definition block. Add it right after `stopCameraPreview` (after line 25). This is an Expo module function:

```swift
Function("getFormats") { (deviceId: String) -> [[String: Any]] in
    let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back

    guard let camera = AVCaptureDevice.default(
        .builtInWideAngleCamera,
        for: .video,
        position: position
    ) else {
        return []
    }

    let formats = camera.formats
    // Store for later lookup by nativeHandle
    if position == .back {
        self.storedBackFormats = formats
    } else {
        self.storedFrontFormats = formats
    }

    return formats.enumerated().map { (index, format) in
        let desc = format.formatDescription
        let dims = CMVideoFormatDescriptionGetDimensions(desc)

        // FPS: min of all minFrameRates, max of all maxFrameRates
        var minFps: Float64 = .greatestFiniteMagnitude
        var maxFps: Float64 = 0
        for range in format.videoSupportedFrameRateRanges {
            minFps = min(minFps, range.minFrameRate)
            maxFps = max(maxFps, range.maxFrameRate)
        }
        if minFps == .greatestFiniteMagnitude { minFps = 0 }

        // Color spaces
        let colorSpaces: [String] = format.supportedColorSpaces.map { cs in
            switch cs {
            case .sRGB: return "sRGB"
            case .P3_D65: return "p3D65"
            case .HLG_BT2020: return "hlgBT2020"
            case .appleLog: return "appleLog"
            @unknown default: return "unknown"
            }
        }

        // Stabilization modes
        var stabilizationModes: [String] = ["off"]
        if format.isVideoStabilizationModeSupported(.standard) {
            stabilizationModes.append("standard")
        }
        if format.isVideoStabilizationModeSupported(.cinematic) {
            stabilizationModes.append("cinematic")
        }
        if format.isVideoStabilizationModeSupported(.cinematicExtended) {
            stabilizationModes.append("cinematicExtended")
        }

        let isHDR = colorSpaces.contains("hlgBT2020")

        return [
            "width": Int(dims.width),
            "height": Int(dims.height),
            "minFps": minFps,
            "maxFps": maxFps,
            "pixelFormat": "bgra",
            "supportedColorSpaces": colorSpaces,
            "isHDR": isHDR,
            "stabilizationModes": stabilizationModes,
            "fieldOfView": format.videoFieldOfView,
            "isBinned": format.isVideoBinned,
            "isMultiCamSupported": format.isMultiCamSupported,
            "nativeHandle": index,
        ] as [String: Any]
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: add getFormats() native function for format enumeration"
```

---

### Task 4: Update startCapture() for format selection + color space

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Update startCameraPreview function signature**

Replace the current `startCameraPreview` function definition (lines 19-21):

```swift
// Old:
Function("startCameraPreview") { (deviceId: String, width: Int, height: Int, fps: Int) in
    self.startCapture(deviceId: deviceId, width: width, height: height, fps: fps)
}
```

With:

```swift
Function("startCameraPreview") { (deviceId: String, nativeHandle: Int, colorSpace: String) in
    self.startCapture(deviceId: deviceId, nativeHandle: nativeHandle, colorSpace: colorSpace)
}
```

- [ ] **Step 2: Add colorSpace mapping helper**

Add this helper method to the class (outside the module definition, near `stopCapture()`):

```swift
private func mapColorSpace(_ name: String) -> AVCaptureColorSpace {
    switch name {
    case "p3D65": return .P3_D65
    case "hlgBT2020": return .HLG_BT2020
    case "appleLog": return .appleLog
    default: return .sRGB
    }
}
```

- [ ] **Step 3: Rewrite startCapture() to use nativeHandle + colorSpace**

Replace the entire `startCapture` method (lines 97-169) with:

```swift
private func startCapture(deviceId: String, nativeHandle: Int, colorSpace: String) {
    sessionQueue.async { [weak self] in
        guard let self = self else { return }
        let session = AVCaptureSession()
        session.sessionPreset = .inputPriority

        let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back

        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: position
        ) else {
            NSLog("[WebGPUCamera] No camera found for position: \(deviceId)")
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: camera) else {
            NSLog("[WebGPUCamera] Could not create camera input")
            return
        }
        if session.canAddInput(input) {
            session.addInput(input)
        }

        // Format selection
        do {
            try camera.lockForConfiguration()

            if nativeHandle >= 0 {
                // User selected a specific format
                let storedFormats = position == .back ? self.storedBackFormats : self.storedFrontFormats
                if nativeHandle < storedFormats.count {
                    let selectedFormat = storedFormats[nativeHandle]
                    camera.activeFormat = selectedFormat

                    // Set FPS to format's max
                    var bestMaxFps: Float64 = 30
                    for range in selectedFormat.videoSupportedFrameRateRanges {
                        bestMaxFps = max(bestMaxFps, range.maxFrameRate)
                    }
                    camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(bestMaxFps))
                    camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(bestMaxFps))
                    self.activeFps = Int(bestMaxFps)

                    NSLog("[WebGPUCamera] Set format: \(CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription).width)x\(CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription).height) @ \(bestMaxFps)fps")
                } else {
                    NSLog("[WebGPUCamera] nativeHandle \(nativeHandle) out of range (\(storedFormats.count) formats), using default")
                    self.applyDefaultFormat(camera: camera)
                }
            } else {
                // No format specified — apply default (1080p/30fps)
                self.applyDefaultFormat(camera: camera)
            }

            // Color space — validate against format, then set explicitly (some formats default to P3)
            let targetColorSpace = self.mapColorSpace(colorSpace)
            if camera.activeFormat.supportedColorSpaces.contains(targetColorSpace) {
                camera.activeColorSpace = targetColorSpace
            } else {
                NSLog("[WebGPUCamera] Color space '\(colorSpace)' not supported by active format, falling back to sRGB")
                camera.activeColorSpace = .sRGB
            }

            camera.unlockForConfiguration()
        } catch {
            NSLog("[WebGPUCamera] Failed to lock camera for configuration: \(error)")
        }

        // Frame output setup
        let dims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
        let width = Int(dims.width)
        let height = Int(dims.height)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true

        let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
        self.frameDelegate = delegate
        output.setSampleBufferDelegate(delegate, queue: self.frameQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.startRunning()

        self.captureSession = session
        self.dataOutput = output

        NSLog("[WebGPUCamera] Camera started: \(width)x\(height), colorSpace=\(colorSpace)")
    }
}
```

- [ ] **Step 4: Add applyDefaultFormat() helper**

Add this helper method to the class:

```swift
/// Default format selection: closest to 1080p, maxFps >= 30, prefer non-binned, sRGB-capable
private func applyDefaultFormat(camera: AVCaptureDevice) {
    var bestFormat: AVCaptureDevice.Format?
    var bestScore = Int.max

    for format in camera.formats {
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)

        // Must support sRGB
        let supportsSRGB = format.supportedColorSpaces.contains(.sRGB)
        if !supportsSRGB { continue }

        // Must support >= 30fps
        var maxFps: Float64 = 0
        for range in format.videoSupportedFrameRateRanges {
            maxFps = max(maxFps, range.maxFrameRate)
        }
        if maxFps < 30 { continue }

        // Score: distance from 1080p (lower is better)
        let resScore = abs(Int(dims.width) - 1920) + abs(Int(dims.height) - 1080)
        // Prefer lower maxFps (don't overshoot) — small penalty
        let fpsScore = Int(maxFps - 30)
        // Prefer non-binned
        let binScore = format.isVideoBinned ? 100 : 0

        let score = resScore + fpsScore + binScore
        if score < bestScore {
            bestScore = score
            bestFormat = format
        }
    }

    if let format = bestFormat {
        camera.activeFormat = format

        // Default to 30fps
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        self.activeFps = 30

        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        NSLog("[WebGPUCamera] Default format: \(dims.width)x\(dims.height) @ 30fps")
    } else {
        NSLog("[WebGPUCamera] No suitable default format found, using device default")
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: update startCapture for format selection + color space + default logic"
```

---

## Chunk 3: JS Hooks

### Task 5: Create useCameraFormats hook

**Files:**
- Create: `packages/react-native-webgpu-camera/src/useCameraFormats.ts`

- [ ] **Step 1: Write useCameraFormats hook**

```ts
import { useEffect, useState } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraFormat, ColorSpace, StabilizationMode } from './types';

function mapNativeFormat(raw: Record<string, any>): CameraFormat {
  return {
    width: raw.width as number,
    height: raw.height as number,
    minFps: raw.minFps as number,
    maxFps: raw.maxFps as number,
    pixelFormat: (raw.pixelFormat as CameraFormat['pixelFormat']) ?? 'bgra',
    supportedColorSpaces: (raw.supportedColorSpaces as ColorSpace[]) ?? ['sRGB'],
    isHDR: (raw.isHDR as boolean) ?? false,
    stabilizationModes: (raw.stabilizationModes as StabilizationMode[]) ?? ['off'],
    fieldOfView: (raw.fieldOfView as number) ?? 0,
    isBinned: (raw.isBinned as boolean) ?? false,
    isMultiCamSupported: (raw.isMultiCamSupported as boolean) ?? false,
    nativeHandle: raw.nativeHandle as number,
  };
}

export function useCameraFormats(device: 'back' | 'front'): CameraFormat[] {
  const [formats, setFormats] = useState<CameraFormat[]>([]);

  useEffect(() => {
    const raw = WebGPUCameraModule.getFormats(device);
    setFormats(raw.map(mapNativeFormat));
  }, [device]);

  return formats;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useCameraFormats.ts
git commit -m "feat: add useCameraFormats hook"
```

---

### Task 6: Update useCamera hook + getActiveFormatInfo

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/useCamera.ts`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

This task combines useCamera rewrite with resolved-dimensions support so the repo is never in a broken intermediate state.

- [ ] **Step 1: Add getActiveFormatInfo to Swift module**

In `WebGPUCameraModule.swift`, add instance properties (near other stored properties):

```swift
var activeWidth: Int = 0
var activeHeight: Int = 0
var activeFps: Int = 0
```

At the end of `startCapture()`, just before `session.startRunning()`, add:

```swift
self.activeWidth = width
self.activeHeight = height
```

(Note: `self.activeFps` is already set during format selection — either `Int(bestMaxFps)` for user-selected formats or `30` from `applyDefaultFormat`.)

Add the Expo module function inside the module definition:

```swift
Function("getActiveFormatInfo") { () -> [String: Any]? in
    if self.activeWidth == 0 { return nil }
    return [
        "width": self.activeWidth,
        "height": self.activeHeight,
        "fps": self.activeFps,
    ]
}
```

- [ ] **Step 2: Rewrite useCamera to accept format + colorSpace and query resolved dimensions**

Replace the entire `useCamera.ts` file with:

```ts
import { useEffect, useState } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraConfig, CameraHandle } from './types';

export function useCamera(config: CameraConfig): CameraHandle {
  const [isReady, setIsReady] = useState(false);
  const [resolvedWidth, setResolvedWidth] = useState(config.format?.width ?? 0);
  const [resolvedHeight, setResolvedHeight] = useState(config.format?.height ?? 0);
  const [resolvedFps, setResolvedFps] = useState(config.format?.maxFps ?? 30);

  const device = config.device;
  const nativeHandle = config.format?.nativeHandle ?? -1;
  const colorSpace = config.colorSpace ?? 'sRGB';

  useEffect(() => {
    WebGPUCameraModule.startCameraPreview(device, nativeHandle, colorSpace);

    // Camera needs time to produce first frame
    const timeout = setTimeout(() => {
      // Query actual dimensions (important when using default format)
      const info = WebGPUCameraModule.getActiveFormatInfo();
      if (info) {
        setResolvedWidth(info.width);
        setResolvedHeight(info.height);
        setResolvedFps(info.fps);
      }
      setIsReady(true);
    }, 500);

    return () => {
      clearTimeout(timeout);
      WebGPUCameraModule.stopCameraPreview();
      setIsReady(false);
    };
  }, [device, nativeHandle, colorSpace]);

  return {
    isReady,
    width: resolvedWidth,
    height: resolvedHeight,
    fps: resolvedFps,
  };
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useCamera.ts \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: update useCamera for format selection + getActiveFormatInfo"
```

---

## Chunk 4: Exports and Example App

### Task 8: Update package exports

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/index.ts`

- [ ] **Step 1: Add exports for new hook and types**

Add to the exports in `index.ts`:

```ts
export { useCameraFormats } from './useCameraFormats';
```

And add to the type exports:

```ts
export type { CameraFormat, ColorSpace, StabilizationMode } from './types';
```

- [ ] **Step 2: Verify types compile**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`

Expected: no errors (or only pre-existing ones unrelated to this feature).

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/src/index.ts
git commit -m "feat: export useCameraFormats hook and format types"
```

---

### Task 9: Update example app

**Files:**
- Modify: `apps/example/src/app/index.tsx`

- [ ] **Step 1: Update imports**

Add `useCameraFormats` and `CameraFormat` to the import from `react-native-webgpu-camera`:

```ts
import { useCamera, useGPUFrameProcessor, useCameraFormats } from 'react-native-webgpu-camera';
import type { CameraFormat } from 'react-native-webgpu-camera';
```

- [ ] **Step 2: Remove hardcoded camera constants**

Remove these constants at the top of the file:

```ts
// DELETE:
const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;
```

- [ ] **Step 3: Add format selection to CameraSpikeScreen**

In `CameraSpikeScreen`, add format selection state and the `useCameraFormats` hook:

```ts
export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [shaderIndex, setShaderIndex] = useState(0);
  const [showDepth, setShowDepth] = useState(false);
  const shader = SHADERS[shaderIndex];

  // Format enumeration
  const formats = useCameraFormats('back');
  const [selectedFormat, setSelectedFormat] = useState<CameraFormat | undefined>();

  // Auto-select best format: 4K 120fps if available, otherwise highest res
  useEffect(() => {
    if (formats.length === 0) return;
    const best =
      formats.find(f => f.width >= 3840 && f.maxFps >= 120) ??
      formats.find(f => f.width >= 1920 && f.maxFps >= 60) ??
      formats[0];
    setSelectedFormat(best);
  }, [formats]);

  // ... rest of component
```

- [ ] **Step 4: Update CameraPreview to use format**

Change `CameraPreview` to accept a `format` prop instead of `shaderChain` only:

```ts
function CameraPreview({ shaderChain, format }: { shaderChain: readonly string[]; format?: CameraFormat }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
  });

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    for (const wgsl of shaderChain) {
      frame.runShader(wgsl);
    }
  });
  // ... rest unchanged
```

- [ ] **Step 5: Update HistogramPreview and HistogramOnFramePreview similarly**

Add `format` prop to both components and replace `useCamera` calls:

```ts
function HistogramPreview({ format }: { format?: CameraFormat }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
  });
  // ... rest unchanged, but remove references to CAMERA_WIDTH/CAMERA_HEIGHT
  // Use camera.width and camera.height instead
```

Replace any remaining `CAMERA_WIDTH` with `camera.width` and `CAMERA_HEIGHT` with `camera.height`.

For HistogramPreview's overlay coordinate calculation:
```ts
const texPortraitW = camera.height; // e.g. 2160 in landscape
const texPortraitH = camera.width;  // e.g. 3840 in landscape
```

- [ ] **Step 6: Pass format through render calls**

In `CameraSpikeScreen`, pass `selectedFormat` to each preview component:

```ts
{isRunning && shader.type === 'histogram' && (
  <HistogramPreview key={shader.name} format={selectedFormat} />
)}
{isRunning && shader.type === 'histogram-onframe' && (
  <HistogramOnFramePreview key={shader.name} format={selectedFormat} />
)}
{isRunning && shader.type === 'simple' && (
  <CameraPreview key={shader.name} shaderChain={shader.wgsl} format={selectedFormat} />
)}
```

- [ ] **Step 7: Add format info to status bar**

Update the status bar to show current format:

```ts
<Text style={styles.statusText}>
  {error
    ? `Error: ${error}`
    : camera.isReady
    ? `${camera.width}x${camera.height} @ ${camera.fps}fps`
    : 'Starting camera...'}
</Text>
```

- [ ] **Step 8: Verify types compile**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: update example app to use format selection API"
```

---

## Chunk 5: Verification

### Task 10: Final verification

- [ ] **Step 1: Full type check**

Run: `cd /Users/kim/dev/rn-webgpu-camera && bunx tsc --noEmit`

Expected: clean (no errors).

- [ ] **Step 2: Review all changes**

Run: `git diff main --stat` to see all files changed.

Verify:
- `types.ts` has `CameraFormat`, `ColorSpace`, `StabilizationMode`, updated `CameraConfig`
- `useCameraFormats.ts` exists with the hook
- `useCamera.ts` accepts `format` + `colorSpace`, queries resolved dimensions
- `WebGPUCameraModule.ts` has `getFormats` and updated `startCameraPreview`
- `WebGPUCameraModule.swift` has `getFormats()`, `applyDefaultFormat()`, `mapColorSpace()`, updated `startCapture()`
- `index.ts` exports the new hook and types
- `index.tsx` uses `useCameraFormats` and passes format objects

- [ ] **Step 3: Build on device**

Run: `cd apps/example && eas build --platform ios --profile development --local`

This is required to test native changes. TypeScript changes alone don't validate the Swift code.
