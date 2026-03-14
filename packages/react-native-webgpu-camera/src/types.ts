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
  readBuffer(index: number): ArrayBuffer | null;
  getCanvas(): SkCanvas | null;
  flushCanvas(): void;
  dispose(): void;
}

/** Union of all typed array constructors — used for buffer output type inference */
export type TypedArrayConstructor =
  | typeof Float32Array | typeof Float64Array
  | typeof Int8Array | typeof Int16Array | typeof Int32Array
  | typeof Uint8Array | typeof Uint16Array | typeof Uint32Array
  | typeof Uint8ClampedArray;

/** Opaque handle returned by runShader in pipeline — resolved to live data in onFrame */
declare const __bufferBrand: unique symbol;
export type BufferHandle<T> = T & { readonly [__bufferBrand]: never };

/** Setup-time frame interface — used inside pipeline callback */
export interface ProcessorFrame {
  /** Run a compute shader — output feeds into next pass or becomes final frame */
  runShader(wgsl: string): void;
  /** Run a compute shader with buffer output — returns a handle resolved per-frame */
  runShader<T extends TypedArrayConstructor>(
    wgsl: string,
    options: { output: T; count: number },
  ): BufferHandle<InstanceType<T>>;

  /** Skia canvas targeting the current pass's output texture.
   *  Only valid inside `pipeline` for between-pass draws.
   *  Draws are recorded once at setup time and replayed every frame (static).
   *  For per-frame dynamic draws, use `onFrame`'s `RenderFrame.canvas`. */
  canvas: SkCanvas;

  /** Current frame dimensions */
  width: number;
  height: number;
}

/** Per-frame render interface — used inside onFrame callback */
export interface RenderFrame {
  /** Skia canvas targeting the final compute output texture */
  canvas: SkCanvas;
  /** Current frame dimensions */
  width: number;
  height: number;
}

/** Strips BufferHandle brand and adds | null for each value */
export type NullableBuffers<B> = {
  [K in keyof B]: B[K] extends BufferHandle<infer U> ? U | null : B[K] | null;
};

/** Configuration for the object form of useGPUFrameProcessor */
export interface ProcessorConfig<B extends Record<string, any>> {
  /** When true, onFrame blocks until current frame's compute + readback completes.
   *  Default false: onFrame receives most recent available data (may be 1 frame behind). */
  sync?: boolean;

  /** Runs once at setup. Declares shader chain and buffer outputs.
   *  Return value maps buffer names to handles for use in onFrame. */
  pipeline: (frame: ProcessorFrame) => B;

  /** Runs every display frame on UI thread.
   *  Receives resolved buffer data and a canvas for Skia draws. */
  onFrame?: (
    frame: RenderFrame,
    buffers: NullableBuffers<B>,
  ) => void;
}

export type FrameProcessor = (frame: ProcessorFrame) => void;

export interface GPUFrameProcessorResult {
  /** Latest processed frame as SkImage — drive a Skia Canvas with this.
   *  The hook owns disposal — do NOT call dispose() on this value. */
  currentFrame: SharedValue<SkImage | null>;
  /** Non-null if shader compilation or pipeline setup failed. */
  error: string | null;
}

export type { SharedValue };

// Global JSI bindings installed by the native pipeline
declare global {
  function __webgpuCamera_nextImage(): SkImage | null;
  function __webgpuCamera_createStream(): CameraStream;
}
