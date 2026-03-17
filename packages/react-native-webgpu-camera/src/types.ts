import type { SharedValue } from 'react-native-reanimated';
import type { SkImage, SkCanvas } from '@shopify/react-native-skia';
import type { ResourceHandle } from './GPUResource';

// useCamera

export interface CameraConfig {
  device: 'back' | 'front';
  /** Selected format from useCameraFormats(). Default: best 1080p/30fps match. */
  format?: CameraFormat;
  /** Color space to activate. Default: 'sRGB'. Must be in format.supportedColorSpaces. */
  colorSpace?: ColorSpace;
  /** Enable depth data capture (requires LiDAR device). Default: false. */
  useDepth?: boolean;
}

export interface CameraHandle {
  /** True once camera is producing frames */
  isReady: boolean;
  /** Camera frame dimensions (passed through from config) */
  width: number;
  height: number;
  fps: number;
  colorSpace: ColorSpace;
}

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
  supportsDepth: boolean;
  /** Opaque index into native format array — pass back to useCamera */
  nativeHandle: number;
}

// useGPUFrameProcessor

/** Per-step timing from last processFrame (milliseconds) */
export interface PipelineMetrics {
  lockWait: number;
  import: number;
  bindGroup: number;
  compute: number;
  buffers: number;
  makeImage: number;
  total: number;
  wall: number;
}

/** Result from beginFrame() — everything in one mutex lock */
export interface FrameSnapshot {
  image?: SkImage;
  canvas?: SkCanvas;
  buffers: (ArrayBuffer | null)[];
  pipelineFps: number;
  generation: number;
  metrics: PipelineMetrics;
}

/** JSI host object shared across Reanimated runtimes */
export interface CameraStream {
  nextImage(): SkImage | null;
  readBuffer(index: number): ArrayBuffer | null;
  getCanvas(): SkCanvas | null;
  flushCanvas(): void;
  flushCanvasAndGetImage(): SkImage | null;
  beginFrame(): FrameSnapshot | null;
  pipelineFps(): number;
  generation(): number;
  metrics(): PipelineMetrics | null;
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
  runShader(wgsl: string, options?: { inputs?: Record<string, any> }): void;
  /** Run a compute shader with buffer output — returns a handle resolved per-frame */
  runShader<T extends TypedArrayConstructor>(
    wgsl: string,
    options: { output: T; count: number; inputs?: Record<string, any> },
  ): BufferHandle<InstanceType<T>>;
  /** Run a compute shader with texture output — returns a handle for use as input */
  runShader(
    wgsl: string,
    options: { output: { readonly __outputType: 'texture2d' }; inputs?: Record<string, any> },
  ): ResourceHandle<'texture2d'>;

  /** Skia canvas targeting the current pass's output texture */
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
export interface ProcessorConfig<
  B extends Record<string, any>,
  R extends Record<string, any> = {},
> {
  /** When true, onFrame blocks until current frame's compute + readback completes.
   *  Default false: onFrame receives most recent available data (may be 1 frame behind). */
  sync?: boolean;

  /** Static GPU resources uploaded once at setup time.
   *  Handles are passed as the second argument to pipeline. */
  resources?: R;

  /** Runs once at setup. Declares shader chain and buffer outputs.
   *  Return value maps buffer names to handles for use in onFrame. */
  pipeline: (frame: ProcessorFrame, resources: R) => B;

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
  /** Latest buffer readback data — keyed by the names returned from pipeline().
   *  Values are null until the first readback completes. Updated every display frame. */
  buffers: SharedValue<Record<string, ArrayBufferView | null>>;
  /** Pipeline FPS — how many frames the native compute pipeline processes per second. */
  fps: SharedValue<number>;
  /** Display FPS — how many truly new frames reach the display per second. */
  displayFps: SharedValue<number>;
  /** Per-step timing from last processFrame (ms). Updated once per second. */
  metrics: SharedValue<PipelineMetrics | null>;
  /** Non-null if shader compilation or pipeline setup failed. */
  error: string | null;
}

export type { SharedValue };
export type { ResourceHandle } from './GPUResource';

// Global JSI bindings installed by the native pipeline
declare global {
  function __webgpuCamera_nextImage(): SkImage | null;
  function __webgpuCamera_createStream(): CameraStream;
}
