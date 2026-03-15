import { useEffect, useState } from "react";
import { useSharedValue, useFrameCallback } from "react-native-reanimated";
import type { SkImage } from "@shopify/react-native-skia";
import WebGPUCameraModule from "../modules/webgpu-camera/src/WebGPUCameraModule";
import type {
  CameraHandle,
  CameraStream,
  ProcessorFrame,
  ProcessorConfig,
  FrameProcessor,
  GPUFrameProcessorResult,
  PipelineMetrics,
  TypedArrayConstructor,
} from "./types";

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
    runShader(
      wgsl: string,
      options?: { output: TypedArrayConstructor; count: number },
    ) {
      const pass: CapturedPass = { wgsl };
      if (options) {
        pass.buffer = { output: options.output, count: options.count };
        bufferMetas.push({
          name: `__buf_${bufferMetas.length}`,
          ctor: options.output,
        });
      }
      passes.push(pass);
      return {} as any;
    },
    canvas: new Proxy({} as any, {
      get(_, prop) {
        if (typeof prop === "string" && prop.startsWith("draw")) {
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
 * Typed array constructor index → constructor.
 * We can't pass constructors directly into worklets, so we use an index.
 * 0=Float32, 1=Float64, 2=Int8, 3=Int16, 4=Int32, 5=Uint8, 6=Uint16, 7=Uint32, 8=Uint8Clamped
 */
const CTOR_INDEX: TypedArrayConstructor[] = [
  Float32Array,
  Float64Array,
  Int8Array,
  Int16Array,
  Int32Array,
  Uint8Array,
  Uint16Array,
  Uint32Array,
  Uint8ClampedArray,
];

function ctorToIndex(ctor: TypedArrayConstructor): number {
  const idx = CTOR_INDEX.indexOf(ctor);
  return idx >= 0 ? idx : 0;
}

function wrapBuffer(
  arrayBuffer: ArrayBuffer,
  ctorIndex: number,
): ArrayBufferView {
  "worklet";
  switch (ctorIndex) {
    case 0:
      return new Float32Array(arrayBuffer);
    case 1:
      return new Float64Array(arrayBuffer);
    case 2:
      return new Int8Array(arrayBuffer);
    case 3:
      return new Int16Array(arrayBuffer);
    case 4:
      return new Int32Array(arrayBuffer);
    case 5:
      return new Uint8Array(arrayBuffer);
    case 6:
      return new Uint16Array(arrayBuffer);
    case 7:
      return new Uint32Array(arrayBuffer);
    case 8:
      return new Uint8ClampedArray(arrayBuffer);
    default:
      return new Float32Array(arrayBuffer);
  }
}

export function useGPUFrameProcessor(
  camera: CameraHandle,
  processorOrConfig: FrameProcessor | ProcessorConfig<any>,
): GPUFrameProcessorResult {
  const [error, setError] = useState<string | null>(null);
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImage | null>(null);
  const buffers = useSharedValue<Record<string, ArrayBufferView | null>>({});
  const fps = useSharedValue(0);
  const displayFps = useSharedValue(0);
  const metrics = useSharedValue<PipelineMetrics | null>(null);
  const lastGen = useSharedValue(0);
  const displayFrameCount = useSharedValue(0);
  const displayFpsTime = useSharedValue(0);

  // Determine form at module scope (not in worklet)
  const isObjectForm =
    typeof processorOrConfig !== "function" && "pipeline" in processorOrConfig;
  const onFrameFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).onFrame
    : undefined;
  const sync = isObjectForm
    ? ((processorOrConfig as ProcessorConfig<any>).sync ?? false)
    : false;
  const pipelineFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any>).pipeline
    : (processorOrConfig as FrameProcessor);

  // Buffer metadata shared with the worklet via shared values
  const bufferCount = useSharedValue(0);
  const bufferNames = useSharedValue<string[]>([]);
  const bufferCtorIndices = useSharedValue<number[]>([]);
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

    // Zero passes is valid — camera frames pass through without compute

    // Store buffer metadata in shared values for worklet access
    bufferCount.value = bufferMetas.length;
    bufferNames.value = bufferMetas.map((m) => m.name);
    bufferCtorIndices.value = bufferMetas.map((m) => ctorToIndex(m.ctor));
    hasOnFrame.value = !!onFrameFn;

    // Determine if canvas is used (between passes OR in onFrame)
    const useCanvas = hasCanvas || (isObjectForm && !!onFrameFn);

    // Build and send native config
    const nativeConfig = buildNativeConfig(
      passes,
      camera.width,
      camera.height,
      useCanvas,
      sync,
    );

    const ok = WebGPUCameraModule.setupMultiPassPipeline(nativeConfig);
    if (!ok) {
      setError("Multi-pass pipeline setup failed");
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
    "worklet";
    const s = stream.value;
    if (!s) return;

    // Single native call: image + buffers + canvas + fps + generation + metrics
    const frame = s.beginFrame();
    if (!frame) return;

    // All from the same snapshot — no extra JSI calls
    fps.value = frame.pipelineFps;

    // Track display FPS — count frames where generation changed (truly new frame)
    if (frame.generation !== lastGen.value) {
      lastGen.value = frame.generation;
      displayFrameCount.value++;
      const now = performance.now();
      if (displayFpsTime.value === 0) {
        displayFpsTime.value = now;
      } else if (now - displayFpsTime.value >= 1000) {
        displayFps.value = Math.round(displayFrameCount.value * 1000 / (now - displayFpsTime.value));
        displayFrameCount.value = 0;
        displayFpsTime.value = now;
        metrics.value = frame.metrics;
      }
    }

    // Resolve buffers from the frame snapshot
    const count = bufferCount.value;
    if (count > 0) {
      const names = bufferNames.value;
      const bpe = bufferCtorIndices.value;
      const readBuffers: Record<string, any> = {};
      for (let i = 0; i < count; i++) {
        const buf = frame.buffers[i];
        readBuffers[names[i]] = buf != null ? wrapBuffer(buf, bpe[i]) : null;
      }
      buffers.value = readBuffers;
    }

    // onFrame canvas path — always flush to produce the composited image
    if (hasOnFrame.value && onFrameFn && frame.canvas) {
      const renderFrame = {
        canvas: frame.canvas,
        width: camera.width,
        height: camera.height,
      };
      onFrameFn(renderFrame, (buffers.value ?? {}) as any);

      const composited = s.flushCanvasAndGetImage();
      if (composited) {
        currentFrame.value?.dispose();
        currentFrame.value = composited;
        frame.image?.dispose();
        return;
      }
    }

    // Non-canvas path or fallback
    if (frame.image) {
      currentFrame.value?.dispose();
      currentFrame.value = frame.image;
    }
  });

  return { currentFrame, buffers, fps, displayFps, metrics, error };
}
