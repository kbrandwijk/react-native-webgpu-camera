import { useEffect, useState } from 'react';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import type { SkImage } from '@shopify/react-native-skia';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type {
  CameraHandle,
  CameraStream,
  ProcessorFrame,
  ProcessorConfig,
  FrameProcessor,
  GPUFrameProcessorResult,
  TypedArrayConstructor,
} from './types';

/** Internal: collected shader/buffer info from the capture proxy */
interface CapturedPass {
  wgsl: string;
  buffer?: {
    output: TypedArrayConstructor;
    count: number;
  };
}

/** Buffer metadata for resolving handles in the worklet */
interface BufferMeta {
  name: string;
  ctor: TypedArrayConstructor;
}

/**
 * Runs the pipeline callback with a capture proxy to collect shader chain
 * and buffer declarations. Returns the captured config.
 */
function capturePipeline<B extends Record<string, any>>(
  pipelineFn: (frame: ProcessorFrame) => B,
  width: number,
  height: number,
): { passes: CapturedPass[]; bufferMetas: BufferMeta[]; hasCanvas: boolean } {
  const passes: CapturedPass[] = [];
  const bufferMetas: BufferMeta[] = [];
  let hasCanvas = false;

  const captureFrame: ProcessorFrame = {
    runShader(wgsl: string, options?: { output: TypedArrayConstructor; count: number }) {
      const pass: CapturedPass = { wgsl };
      if (options) {
        pass.buffer = { output: options.output, count: options.count };
        bufferMetas.push({ name: `__buf_${bufferMetas.length}`, ctor: options.output });
      }
      passes.push(pass);
      return {} as any;
    },
    canvas: new Proxy({} as any, {
      get(_, prop) {
        if (typeof prop === 'string' && prop.startsWith('draw')) {
          hasCanvas = true;
        }
        return () => {};
      },
    }),
    width,
    height,
  };

  let returnedHandles: B | undefined;
  try {
    returnedHandles = pipelineFn(captureFrame);
  } catch {
    // Processor may reference worklet-only APIs during capture — safe to ignore
  }

  // Map returned handle keys to buffer indices
  if (returnedHandles) {
    const keys = Object.keys(returnedHandles);
    for (let i = 0; i < Math.min(keys.length, bufferMetas.length); i++) {
      bufferMetas[i].name = keys[i];
    }
  }

  return { passes, bufferMetas, hasCanvas };
}

/**
 * Build the native pipeline config from captured passes.
 */
function buildNativeConfig(
  passes: CapturedPass[],
  width: number,
  height: number,
  useCanvas: boolean,
  sync: boolean,
) {
  const shaders = passes.map((p) => p.wgsl);
  const buffers: [number, number, number][] = [];

  passes.forEach((pass, passIndex) => {
    if (pass.buffer) {
      const elementSize = pass.buffer.output.BYTES_PER_ELEMENT ?? 4;
      buffers.push([passIndex, elementSize, pass.buffer.count]);
    }
  });

  return { shaders, width, height, buffers, useCanvas, sync };
}

/**
 * Typed array constructor lookup — available as globals in the worklet runtime.
 */
function wrapBuffer(arrayBuffer: ArrayBuffer, bytesPerElement: number): ArrayBufferView {
  'worklet';
  switch (bytesPerElement) {
    case 8: return new Float64Array(arrayBuffer);
    case 4: return new Float32Array(arrayBuffer);
    case 2: return new Uint16Array(arrayBuffer);
    case 1: return new Uint8Array(arrayBuffer);
    default: return new Float32Array(arrayBuffer);
  }
}

export function useGPUFrameProcessor(
  camera: CameraHandle,
  processorOrConfig: FrameProcessor | ProcessorConfig<any>,
): GPUFrameProcessorResult {
  const [error, setError] = useState<string | null>(null);
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImage | null>(null);

  // Determine form at module scope (not in worklet)
  const isObjectForm = typeof processorOrConfig !== 'function' && 'pipeline' in processorOrConfig;
  const onFrameFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).onFrame
    : undefined;
  const sync = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).sync ?? false
    : false;
  const pipelineFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).pipeline
    : (processorOrConfig as FrameProcessor);

  // Buffer metadata shared with the worklet via shared values
  const bufferCount = useSharedValue(0);
  const bufferNames = useSharedValue<string[]>([]);
  const bufferBytesPerElement = useSharedValue<number[]>([]);
  const hasOnFrame = useSharedValue(false);

  // Setup compute pipeline when camera is ready
  useEffect(() => {
    if (!camera.isReady) return;

    // Capture shader chain and buffer declarations
    const { passes, bufferMetas, hasCanvas } = capturePipeline(
      pipelineFn as (frame: ProcessorFrame) => any,
      camera.width,
      camera.height,
    );

    if (passes.length === 0) {
      setError('No shader provided — call frame.runShader(wgslCode) in your processor');
      return;
    }

    // Store buffer metadata in shared values for worklet access
    bufferCount.value = bufferMetas.length;
    bufferNames.value = bufferMetas.map((m) => m.name);
    bufferBytesPerElement.value = bufferMetas.map((m) => m.ctor.BYTES_PER_ELEMENT ?? 4);
    hasOnFrame.value = !!onFrameFn;

    // Determine if canvas is used (between passes OR in onFrame)
    const useCanvas = hasCanvas || (isObjectForm && !!onFrameFn);

    // Build and send native config
    const nativeConfig = buildNativeConfig(
      passes, camera.width, camera.height, useCanvas, sync,
    );

    const ok = WebGPUCameraModule.setupMultiPassPipeline(nativeConfig);
    if (!ok) {
      setError('Multi-pass pipeline setup failed');
      return;
    }
    setError(null);

    // Create stream host object — shared across Reanimated runtimes
    stream.value = globalThis.__webgpuCamera_createStream();

    return () => {
      currentFrame.value?.dispose();
      currentFrame.value = null;
      stream.value = null;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, [camera.isReady, camera.width, camera.height, camera.fps]);

  // Frame callback — runs on Reanimated UI thread every display frame
  useFrameCallback(() => {
    'worklet';
    const s = stream.value;
    if (!s) return;

    const img = s.nextImage();
    if (!img) return;

    if (hasOnFrame.value && onFrameFn) {
      // --- Object form with onFrame ---
      const count = bufferCount.value;
      const names = bufferNames.value;
      const bpe = bufferBytesPerElement.value;
      const buffers: Record<string, any> = {};

      for (let i = 0; i < count; i++) {
        const buf = s.readBuffer(i);
        buffers[names[i]] = buf !== null ? wrapBuffer(buf, bpe[i]) : null;
      }

      const canvas = s.getCanvas();

      const renderFrame = {
        canvas: canvas!,
        width: img.width(),
        height: img.height(),
      };

      onFrameFn(renderFrame, buffers as any);

      if (canvas) {
        s.flushCanvas();
        const composited = s.nextImage();
        if (composited) {
          currentFrame.value?.dispose();
          currentFrame.value = composited;
          img.dispose();
        } else {
          currentFrame.value?.dispose();
          currentFrame.value = img;
        }
      } else {
        currentFrame.value?.dispose();
        currentFrame.value = img;
      }
    } else {
      // --- Shorthand form (no onFrame) ---
      currentFrame.value?.dispose();
      currentFrame.value = img;
    }
  });

  return { currentFrame, error };
}
