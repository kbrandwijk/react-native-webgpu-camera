# react-native-webgpu-camera

GPU compute shaders on live camera frames in React Native. Zero pixel copies.

```
Camera → IOSurface → Dawn (WebGPU) → Compute Shader → Skia Graphite → Canvas
```

Everything stays on the GPU. JavaScript gets opaque handles, not pixels.

## What it looks like

```typescript
import { useCamera, useGPUFrameProcessor } from 'react-native-webgpu-camera';

function CameraPreview() {
  const camera = useCamera({ device: 'back', width: 3840, height: 2160, fps: 120 });

  const { currentFrame } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    frame.runShader(SOBEL_WGSL);         // edge detection
    frame.runShader(SOBEL_COLOR_WGSL);   // colorize edges
  });

  return (
    <Canvas style={StyleSheet.absoluteFill}>
      <Image image={currentFrame} x={0} y={0} width={width} height={height} fit="cover" />
    </Canvas>
  );
}
```

Chain multiple WGSL compute shaders. Get buffer data back from the GPU:

```typescript
const { currentFrame, buffers } = useGPUFrameProcessor(camera, {
  pipeline: (frame) => {
    "worklet";
    const hist = frame.runShader(HISTOGRAM_WGSL, {
      output: Uint32Array,
      count: 256,
    });
    return { hist };
  },
  onFrame: (frame, { hist }) => {
    "worklet";
    // Draw on the GPU texture — burns into the video
    drawHistogramBars(frame.canvas, hist);
  },
});

// Or use buffers in React for UI-only overlays (not recorded)
const histData = buffers.value.hist;
```

## Performance

iPhone 16 Pro, Apple GPU profiler:

| Metric            | Single-pass    | Multi-pass (2 shaders) |
| ----------------- | -------------- | ---------------------- |
| GPU frame time    | 2.88ms         | 4ms                    |
| Display FPS       | 120fps         | 120fps                 |
| Camera resolution | 3840x2160 (4K) | 3840x2160 (4K)         |
| Frame drops       | 0%             | 0%                     |
| Thermal           | nominal        | nominal                |

## Status: Phase 0 — Spike Validation

This is an early-stage project validating whether the architecture works. It does. What's proven:

- Zero-copy camera frame import via IOSurface → Dawn `SharedTextureMemory`
- Multi-pass WGSL compute shader chains on the shared Skia Graphite Dawn device
- GPU buffer readback to JavaScript (histograms, feature data)
- 4K @ 120fps sustained with massive GPU headroom
- `useGPUFrameProcessor` hook with worklet-based render loop (no JS thread, no React re-renders)

What's next:

- ONNX Runtime WebGPU execution provider — ML inference on the same Dawn device
- `frame.runModel()` API for on-device ML in the shader pipeline
- Android support (`AHardwareBuffer` → `SharedTextureMemory`)
- Recording pipeline (AVAssetWriter surface)

## Tech stack

- [Expo](https://expo.dev) 55 / React Native 0.83 (New Architecture)
- [@shopify/react-native-skia](https://github.com/Shopify/react-native-skia) with Skia Graphite (`SK_GRAPHITE=1`)
- Dawn (bundled with Skia Graphite) for WebGPU compute
- [react-native-reanimated](https://github.com/software-mansion/react-native-reanimated) for UI thread worklets
- WGSL compute shaders

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ JavaScript                                          │
│                                                     │
│  useGPUFrameProcessor(camera, shaderChain)          │
│    ├── capture proxy records shader chain at setup   │
│    ├── useFrameCallback (UI thread worklet)          │
│    │     └── stream.nextImage() → SkImage            │
│    └── buffers shared value → React UI overlays      │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Native C++ (camera thread)                          │
│                                                     │
│  AVCaptureSession → CVPixelBuffer                   │
│    → IOSurface (already on GPU)                     │
│    → Dawn SharedTextureMemory (zero-copy import)    │
│    → wgpu::Texture (input)                          │
│    → Compute Pass 0 → texA                          │
│    → Compute Pass 1 → texB                          │
│    → ...                                            │
│    → MakeImageFromTexture → SkImage                 │
│    → CameraStreamHostObject.nextImage()             │
│                                                     │
│  Buffer readback:                                   │
│    GPU buffer → staging (double-buffered)            │
│    → MapAsync → readBuffer() via JSI                │
│                                                     │
├─────────────────────────────────────────────────────┤
│ GPU                                                 │
│                                                     │
│  Dawn device (shared with Skia Graphite)            │
│  WGSL compute shaders + storage buffers             │
│  Skia Graphite rendering                            │
└─────────────────────────────────────────────────────┘
```

## Read more

Full write-up of the spike — what broke, what worked, and the 13+ build failures along the way:

**[Zero-Copy GPU Compute on Camera Frames in React Native — What Actually Worked](https://dev.to/kbrandwijk/zero-copy-gpu-compute-on-camera-frames-in-react-native-what-actually-worked-512j)**

## License

MIT
