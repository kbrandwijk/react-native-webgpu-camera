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
