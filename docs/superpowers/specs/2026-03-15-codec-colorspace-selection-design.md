# Codec & Color Space Selection — Design Spec

## Summary

Add format enumeration and color space selection to the camera hook, replacing hardcoded resolution/fps/color-space with a query-and-select API. Users query available formats per device, pick one (or rely on sane defaults), and optionally set a color space.

## API Surface

### `useCameraFormats(device)`

Returns all available capture formats for a device.

```ts
function useCameraFormats(device: 'back' | 'front'): CameraFormat[];
```

Calls native `getFormats()` once per device value. Returns a stable array of `CameraFormat` objects.

### `useCamera(config)`

Updated signature — `width`, `height`, `fps` replaced by optional `format` + `colorSpace`.

```ts
function useCamera(config: {
  device: 'back' | 'front';
  format?: CameraFormat;      // default: best 1080p/30fps match
  colorSpace?: ColorSpace;    // default: 'sRGB'
}): CameraHandle;
```

Breaking change from `{ device, width, height, fps }`. Acceptable in spike phase.

## Data Model

```ts
interface CameraFormat {
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
  nativeHandle: number; // index into native format array
}

type ColorSpace = 'sRGB' | 'p3D65' | 'hlgBT2020' | 'appleLog';
type StabilizationMode = 'off' | 'standard' | 'cinematic' | 'cinematicExtended';
```

`nativeHandle` is an opaque index into a native-side array of `AVCaptureDevice.Format` objects. It avoids serializing the full native format and is passed back to native when setting the active format.

## Default Format Selection

When no `format` is passed to `useCamera`:

1. Filter formats to those supporting sRGB
2. Find closest to 1920×1080 resolution
3. Among matches, prefer `maxFps >= 30` with lowest `maxFps` (don't overshoot)
4. Prefer non-binned over binned
5. Fallback: device's first format

Default color space: `'sRGB'`.

## Validation

- If `colorSpace` is set but the format doesn't support it → log warning, fall back to sRGB
- Fallback to sRGB explicitly sets `device.activeColorSpace = .sRGB` (some formats default to P3 if unset)
- If `format.nativeHandle` references a different device than `device` → throw
- `nativeHandle` is scoped per `getFormats()` call per device — each call rebuilds the stored format array. The `useCameraFormats` hook re-fetches on device change, ensuring handles stay valid.

## Native Implementation (iOS)

### `getFormats(device: String) -> [[String: Any]]`

Enumerates `AVCaptureDevice.formats` for the requested device position. For each `AVCaptureDevice.Format`:

| CameraFormat field | AVCaptureDevice.Format source |
|---|---|
| `width`, `height` | `formatDescription.dimensions` |
| `minFps`, `maxFps` | `videoSupportedFrameRateRanges`: `minFps = min(range.minFrameRate)`, `maxFps = max(range.maxFrameRate)` across all ranges |
| `pixelFormat` | Always `'bgra'` — the capture output forces `kCVPixelFormatType_32BGRA` regardless of sensor format. Exposed for future use if we support other output formats. |
| `supportedColorSpaces` | `supportedColorSpaces` → mapped enum values |
| `isHDR` | `.hlgBT2020` present in supported color spaces |
| `fieldOfView` | `videoFieldOfView` |
| `isBinned` | `isVideoBinned` |
| `stabilizationModes` | `isVideoStabilizationModeSupported()` checked per mode |
| `isMultiCamSupported` | `isMultiCamSupported` |
| `nativeHandle` | Array index of stored `AVCaptureDevice.Format` reference |

The native side stores the `AVCaptureDevice.Format` objects in an array. When `useCamera` receives a config with a `nativeHandle`, it sets `device.activeFormat` to the stored format directly.

### Format activation

```swift
device.lockForConfiguration()
device.activeFormat = storedFormats[nativeHandle]
device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: Int32(format.maxFps))
device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: Int32(format.maxFps))
device.activeColorSpace = mapColorSpace(config.colorSpace ?? "sRGB")
device.unlockForConfiguration()
```

Session preset remains `.inputPriority`.

## JS Implementation

### `useCameraFormats`

```ts
function useCameraFormats(device: 'back' | 'front'): CameraFormat[] {
  const [formats, setFormats] = useState<CameraFormat[]>([]);

  useEffect(() => {
    const raw = WebGPUCameraModule.getFormats(device);
    setFormats(raw.map(mapNativeFormat));
  }, [device]);

  return formats;
}
```

### `useCamera` changes

- Remove `width`, `height`, `fps` from `CameraConfig`
- Add optional `format?: CameraFormat` and `colorSpace?: ColorSpace`
- If no `format`, pass `nativeHandle: -1` to native — native runs the default selection logic (1080p/30/sRGB) against the device's own format list. This avoids a JS→native→JS→native round-trip.
- Pass `nativeHandle` + `colorSpace` string to native setup call
- Derive `camera.width` / `camera.height` from the resolved format (consumed by `useGPUFrameProcessor`)

### `CameraHandle` — no changes

```ts
interface CameraHandle {
  isReady: boolean;
  width: number;
  height: number;
  fps: number;   // derived from resolved format's maxFps
}
```

## Example Usage

```ts
const formats = useCameraFormats('back');

// Find a 4K 120fps HDR format
const format = formats.find(f =>
  f.width >= 3840 && f.maxFps >= 120 && f.supportedColorSpaces.includes('hlgBT2020')
);

const camera = useCamera({
  device: 'back',
  format,
  colorSpace: 'hlgBT2020',
});

const { currentFrame } = useGPUFrameProcessor(camera, (frame) => {
  'worklet';
  frame.runShader(COLOR_GRADE_WGSL);
});
```

## Files to Create/Modify

- `packages/react-native-webgpu-camera/src/types.ts` — add `CameraFormat`, `ColorSpace`, `StabilizationMode`, update `CameraConfig`
- `packages/react-native-webgpu-camera/src/useCameraFormats.ts` — new hook
- `packages/react-native-webgpu-camera/src/useCamera.ts` — update config handling, add default selection
- `packages/react-native-webgpu-camera/src/index.ts` — export new hook and types
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` — add `getFormats()`, update session setup
- `apps/example/src/app/index.tsx` — update to use new API
