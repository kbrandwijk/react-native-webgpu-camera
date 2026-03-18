# useGPUFrameProcessor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the validated zero-copy GPU compute pipeline as two composable React hooks: `useCamera` (camera lifecycle) and `useGPUFrameProcessor` (GPU compute + Skia drawing on camera frames).

**Architecture:** Thin React hooks wrapping the existing native pipeline. `useCamera` manages AVCaptureSession via `WebGPUCameraModule`. `useGPUFrameProcessor` sets up the Dawn compute pipeline, creates a `CameraStreamHostObject` (JSI), and runs a `useFrameCallback` worklet that calls `stream.nextImage()`, writing results to a `SharedValue<SkImage>`. No native code changes needed — hooks are pure TypeScript on top of proven native infrastructure.

**Tech Stack:** TypeScript, React hooks, Reanimated (`useSharedValue`, `useFrameCallback`), `@shopify/react-native-skia` (SkImage types), Expo modules (`WebGPUCameraModule`)

**Spec:** `docs/superpowers/specs/2026-03-13-use-gpu-frame-processor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/react-native-webgpu-camera/src/types.ts` | Create | Shared types: `CameraConfig`, `CameraHandle`, `ProcessorFrame`, `GPUFrameProcessorResult`, `CameraStream` |
| `packages/react-native-webgpu-camera/src/useCamera.ts` | Create | Camera lifecycle hook — start/stop/restart on config change |
| `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` | Create | GPU frame processor hook — compute pipeline setup, stream creation, useFrameCallback worklet loop |
| `packages/react-native-webgpu-camera/src/index.ts` | Modify | Re-export hooks and types |
| `apps/example/src/app/index.tsx` | Modify | Replace manual pipeline wiring with hook API |

---

## v1 Scope Note

In v1, `frame.runShader()` configures which shader the native pipeline uses — it does not dispatch compute per-frame from the worklet. The native camera thread runs the compute shader on every frame automatically. The worklet just grabs the latest result via `stream.nextImage()`.

The processor callback is called once at setup time to capture the WGSL string. The `'worklet'` directive is forward-compatible — when `frame.canvas` support lands (requiring native SkSurface changes), the processor will genuinely run per-frame in the worklet. For v1, the callback is a configuration function.

`frame.canvas` is not functional in v1 — it requires native changes to expose an SkSurface from the pipeline. The `canvas` property exists on the type for API completeness but is a no-op stub.

---

## Chunk 1: Types and useCamera

### Task 1: Create shared types

**Files:**
- Create: `packages/react-native-webgpu-camera/src/types.ts`

- [ ] **Step 1: Create types file**

```typescript
import type { SharedValue } from 'react-native-reanimated';
import type { SkImage, SkCanvas } from '@shopify/react-native-skia';

// useCamera

export interface CameraConfig {
  device: 'back' | 'front';
  width: number;
  height: number;
  fps: number;
}

export interface CameraHandle {
  /** True once camera is producing frames */
  isReady: boolean;
  /** Camera frame dimensions (passed through from config) */
  width: number;
  height: number;
  fps: number;
}

// useGPUFrameProcessor

/** JSI host object shared across Reanimated runtimes */
export interface CameraStream {
  nextImage(): SkImage | null;
  dispose(): void;
}

export interface ProcessorFrame {
  /** Run a WGSL compute shader on the current frame */
  runShader(wgslCode: string): void;
  /** Skia canvas targeting the output texture — draws are recorded.
   *  NOTE: stub in v1, functional when native SkSurface support lands. */
  canvas: SkCanvas;
  /** Current frame dimensions */
  width: number;
  height: number;
}

export type FrameProcessor = (frame: ProcessorFrame) => void;

export interface GPUFrameProcessorResult {
  /** Latest processed frame as SkImage — drive a Skia Canvas with this.
   *  The hook owns disposal — do NOT call dispose() on this value. */
  currentFrame: SharedValue<SkImage | null>;
  /** Non-null if shader compilation or pipeline setup failed.
   *  This is React state (not a SharedValue) — changes trigger re-renders,
   *  which is appropriate since errors are rare. */
  error: string | null;
}

export type { SharedValue };

// Global JSI bindings installed by the native pipeline
declare global {
  function __webgpuCamera_nextImage(): SkImage | null;
  function __webgpuCamera_createStream(): CameraStream;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/types.ts
git commit -m "feat: add shared types for useCamera and useGPUFrameProcessor"
```

---

### Task 2: Create useCamera hook

**Files:**
- Create: `packages/react-native-webgpu-camera/src/useCamera.ts`

- [ ] **Step 1: Write useCamera hook**

Note: `useCamera` does NOT start the camera on its own. It only starts the camera after `useGPUFrameProcessor` has set up the compute pipeline. This is achieved by exposing the config and a `start()` function that `useGPUFrameProcessor` calls when the pipeline is ready. This matches the proven startup order: setup compute → create stream → start camera.

Wait — that couples the hooks. The simpler approach (matching the spec): `useCamera` starts the camera immediately, and `useGPUFrameProcessor` sets up compute when `camera.isReady`. Camera frames before compute is ready are simply dropped by the native `FrameDelegate` (it calls `dawnBridge?.processFrame()` which is nil before setup). This is the existing behavior.

```typescript
import { useState, useEffect, useRef } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraConfig, CameraHandle } from './types';

export function useCamera(config: CameraConfig): CameraHandle {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    setIsReady(false);

    WebGPUCameraModule.startCameraPreview(
      config.device,
      config.width,
      config.height,
      config.fps,
    );

    // Camera needs time to produce first frame
    const timer = setTimeout(() => setIsReady(true), 500);

    return () => {
      clearTimeout(timer);
      WebGPUCameraModule.stopCameraPreview();
      setIsReady(false);
    };
  }, [config.device, config.width, config.height, config.fps]);

  return {
    isReady,
    width: config.width,
    height: config.height,
    fps: config.fps,
  };
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useCamera.ts
git commit -m "feat: useCamera hook — camera lifecycle management"
```

---

## Chunk 2: useGPUFrameProcessor

### Task 3: Create useGPUFrameProcessor hook

**Files:**
- Create: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`

This is the core hook. It:
1. Extracts the WGSL shader from the processor callback (called once at setup)
2. Sets up the Dawn compute pipeline when the camera is ready
3. Creates a `CameraStreamHostObject` via JSI
4. Runs a `useFrameCallback` worklet that calls `stream.nextImage()` and updates `currentFrame`
5. Manages SkImage disposal (including final frame on unmount)

- [ ] **Step 1: Write useGPUFrameProcessor hook**

```typescript
import { useEffect, useState, useRef } from 'react';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import type { SkImage } from '@shopify/react-native-skia';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type {
  CameraHandle,
  CameraStream,
  FrameProcessor,
  GPUFrameProcessorResult,
} from './types';

export function useGPUFrameProcessor(
  camera: CameraHandle,
  processor: FrameProcessor,
): GPUFrameProcessorResult {
  const [error, setError] = useState<string | null>(null);
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImage | null>(null);
  const processorRef = useRef(processor);
  processorRef.current = processor;

  // Setup compute pipeline when camera is ready
  useEffect(() => {
    if (!camera.isReady) return;

    // v1: Extract WGSL by calling the processor once with a capture proxy.
    // The processor runs as a configuration function at setup time.
    // When frame.canvas support lands, the processor will run per-frame
    // in the worklet instead.
    let capturedWgsl: string | null = null;
    const captureFrame = {
      runShader(wgslCode: string) {
        if (!capturedWgsl) capturedWgsl = wgslCode;
      },
      canvas: new Proxy({} as any, {
        get() { return () => {}; }, // no-op stub for v1
      }),
      width: camera.width,
      height: camera.height,
    };
    try {
      processorRef.current(captureFrame);
    } catch {
      // Processor may reference worklet-only APIs during capture — safe to ignore
    }

    if (!capturedWgsl) {
      setError('No shader provided — call frame.runShader(wgslCode) in your processor');
      return;
    }

    // Setup native compute pipeline (compiles shader, creates GPU resources, installs JSI)
    const ok = WebGPUCameraModule.setupComputePipeline(
      capturedWgsl,
      camera.width,
      camera.height,
    );
    if (!ok) {
      setError('Compute pipeline setup failed');
      return;
    }
    setError(null);

    // Create stream host object — shared across Reanimated runtimes
    stream.value = globalThis.__webgpuCamera_createStream();

    return () => {
      // Dispose final frame to prevent GPU memory leak
      currentFrame.value?.dispose();
      currentFrame.value = null;
      stream.value = null;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, [camera.isReady, camera.width, camera.height, camera.fps]);

  // Frame callback — runs on UI thread every display frame
  useFrameCallback(() => {
    'worklet';
    const s = stream.value;
    if (!s) return;

    const img = s.nextImage();
    if (img) {
      currentFrame.value?.dispose();
      currentFrame.value = img;
    }
  });

  return { currentFrame, error };
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: useGPUFrameProcessor hook — worklet-based GPU frame processing"
```

---

## Chunk 3: Exports and Example App

### Task 4: Update package exports

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/index.ts`

- [ ] **Step 1: Update index.ts to re-export hooks and types**

Replace the current content with:

```typescript
// react-native-webgpu-camera

// Hooks
export { useCamera } from './useCamera';
export { useGPUFrameProcessor } from './useGPUFrameProcessor';

// Types
export type {
  CameraConfig,
  CameraHandle,
  CameraStream,
  ProcessorFrame,
  FrameProcessor,
  GPUFrameProcessorResult,
} from './types';

// Native module (advanced usage)
export { default as WebGPUCameraModule } from '../modules/webgpu-camera/src/WebGPUCameraModule';
export type { FrameDimensions } from '../modules/webgpu-camera/src/WebGPUCameraModule';
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/index.ts
git commit -m "feat: export useCamera, useGPUFrameProcessor hooks and types"
```

---

### Task 5: Update example app to use hooks

**Files:**
- Modify: `apps/example/src/app/index.tsx`

- [ ] **Step 1: Rewrite example app using hooks**

Replace the current manual pipeline wiring with the hook API. The app should:
- Use `useCamera` for camera lifecycle
- Use `useGPUFrameProcessor` for compute + rendering
- Pass `currentFrame` shared value to Skia Canvas
- Keep the start/stop button (controls whether hooks are active via mount/unmount)
- Remove all direct `WebGPUCameraModule` calls from the component

Note: This intentionally drops the spike metrics (`useSpikeMetrics`) and recording functionality. Those were spike validation tools — the hooks themselves are the deliverable now. The spike metrics proved the pipeline works at 120fps; this example demonstrates the clean public API.

```typescript
import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage } from '@shopify/react-native-skia';
import { useCamera, useGPUFrameProcessor } from 'react-native-webgpu-camera';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';

const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;

function CameraPreview() {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    frame.runShader(SOBEL_WGSL);
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <Group transform={[
          { translateX: screenW },
          { rotate: Math.PI / 2 },
        ]}>
          <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Pipeline running' : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);

  return (
    <View style={styles.container}>
      {isRunning && <CameraPreview />}

      <View style={styles.controls}>
        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={() => setIsRunning(!isRunning)}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  statusBar: {
    position: 'absolute', top: 44, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 4, padding: 8,
  },
  statusText: { color: '#aaa', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'center', gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 24,
    paddingHorizontal: 24, paddingVertical: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: { backgroundColor: 'rgba(255,80,80,0.4)', borderColor: 'rgba(255,80,80,0.6)' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
```

Key design choice: `CameraPreview` is a separate component that mounts/unmounts with `isRunning`. This means the hooks' `useEffect` cleanup naturally handles start/stop — no manual `startPipeline`/`stopPipeline` callbacks needed.

- [ ] **Step 2: Commit**

```bash
git add apps/example/src/app/index.tsx
git commit -m "feat: example app uses useCamera + useGPUFrameProcessor hooks"
```

---

## Chunk 4: Verify

### Task 6: Test on device

- [ ] **Step 1: Verify no native rebuild needed**

The hooks are pure TypeScript — no native code changes. Metro fast refresh should pick up the changes. The latest build (build-1773444320148.ipa) already has the `CameraStreamHostObject` and `__webgpuCamera_createStream` JSI bindings.

- [ ] **Step 2: Test on device**

Open the app on the iPhone 16 Pro. Tap "Start Pipeline". Expected:
- Status shows "Starting camera..." then "Pipeline running"
- Sobel edge detection visible on camera feed
- 120fps on GPU overlay (same performance as before hooks)
- Tap "Stop" unmounts `CameraPreview`, cleaning up camera + compute pipeline
- No memory leaks (final SkImage disposed on unmount)

- [ ] **Step 3: Final commit if any adjustments needed**

```bash
git add -A
git commit -m "fix: adjustments from device testing"
```
