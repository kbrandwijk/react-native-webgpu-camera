import { useEffect, useState } from "react";
import { useSharedValue, useFrameCallback } from "react-native-reanimated";
import { Skia } from "@shopify/react-native-skia";
import type { SkImage } from "@shopify/react-native-skia";
import WebGPUCameraModule from "../modules/webgpu-camera/src/WebGPUCameraModule";
import { isResourceHandle, isTextureOutputToken } from './GPUResource';
import type { ResourceHandle, ModelResourceHandle } from './GPUResource';
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
  textureOutput?: boolean;
  inputs?: CapturedInput[];
}

/** A captured custom input binding for a pass */
interface CapturedInput {
  name: string;
  bindingIndex: number;
  type: 'texture3d' | 'texture2d' | 'sampler' | 'storageBufferRead';
  resourceHandle?: number;
  sourcePass?: number;
  sourceBuffer?: number;
  modelOutput?: number;
}

/** A captured model inference pass */
interface CapturedModel {
  path: string;
  inputShape?: number[];
  normalization?: { mean: [number, number, number]; std: [number, number, number] };
  sync: boolean;
  pipelineIndex: number;
}

/** A resource spec to send to native for GPU upload */
interface CapturedResource {
  type: 'texture3d' | 'texture2d' | 'storageBuffer' | 'cameraDepth';
  data?: ArrayBuffer;
  fileUri?: string;
  width?: number;
  height?: number;
  depth?: number;
  format?: string;
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
function capturePipeline<B extends Record<string, any>, R extends Record<string, any>>(
  pipelineFn: (frame: ProcessorFrame, resources: R) => B,
  resources: R | undefined,
  width: number,
  height: number,
): {
  passes: CapturedPass[];
  bufferMetas: BufferMeta[];
  hasCanvas: boolean;
  capturedResources: CapturedResource[];
  capturedModels: CapturedModel[];
} {
  const passes: CapturedPass[] = [];
  const bufferMetas: BufferMeta[] = [];
  const capturedResources: CapturedResource[] = [];
  const capturedModels: CapturedModel[] = [];
  let hasCanvas = false;

  // Map resource handle identity → index into capturedResources
  const handleToIndex = new Map<any, number>();

  // Build resource handle map and collect resource specs for native
  const resourceHandles: Record<string, any> = {};
  if (resources) {
    for (const [name, handle] of Object.entries(resources)) {
      if (isResourceHandle(handle)) {
        const rh = handle as ResourceHandle<any>;
        // Skip model resources — they're handled by runModel(), not the resource upload path
        if (rh.__resourceType === 'model') {
          resourceHandles[name] = handle;
          continue;
        }
        const idx = capturedResources.length;
        // Store resource spec for native upload
        capturedResources.push({
          type: rh.__resourceType as CapturedResource['type'],
          data: rh.__data,
          fileUri: rh.__fileUri,
          width: rh.__dims?.width,
          height: rh.__dims?.height,
          depth: rh.__dims?.depth,
          format: rh.__dims?.format,
        });
        handleToIndex.set(handle, idx);
        resourceHandles[name] = handle;
      }
    }
  }

  // Track pass-output handles: map handle → { passIndex, bufferIndex }
  const outputHandleMap = new Map<any, { passIndex: number; bufferIndex: number; isTexture: boolean; modelIndex?: number }>();

  const captureFrame: ProcessorFrame = {
    runShader(
      wgsl: string,
      options?: {
        output?: TypedArrayConstructor | { __outputType: string };
        count?: number;
        inputs?: Record<string, any>;
      },
    ) {
      const passIndex = passes.length;
      const pass: CapturedPass = { wgsl };

      // Process inputs — assign binding indices from 3+
      if (options?.inputs) {
        let nextBinding = 3;
        pass.inputs = [];

        for (const [name, handle] of Object.entries(options.inputs)) {
          if (isResourceHandle(handle) && handleToIndex.has(handle)) {
            const rh = handle as ResourceHandle<any>;
            // Resource from resources block — look up by identity
            const resIndex = handleToIndex.get(handle)!;
            if (rh.__resourceType === 'texture3d' || rh.__resourceType === 'texture2d' || rh.__resourceType === 'cameraDepth') {
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: rh.__resourceType === 'texture3d' ? 'texture3d' : 'texture2d',
                resourceHandle: resIndex,
              });
              nextBinding++;
              // Auto-pair sampler
              pass.inputs.push({
                name: `${name}_sampler`,
                bindingIndex: nextBinding,
                type: 'sampler',
                resourceHandle: resIndex,
              });
              nextBinding++;
            } else if (rh.__resourceType === 'storageBuffer') {
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: 'storageBufferRead',
                resourceHandle: resIndex,
              });
              nextBinding++;
            }
          } else if (outputHandleMap.has(handle)) {
            // Buffer/texture output from a previous pass or model
            const src = outputHandleMap.get(handle)!;
            if (src.modelIndex !== undefined) {
              // Model output — bind as texture + sampler
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: 'texture2d',
                modelOutput: src.modelIndex,
              });
              nextBinding++;
              pass.inputs.push({
                name: `${name}_sampler`,
                bindingIndex: nextBinding,
                type: 'sampler',
                modelOutput: src.modelIndex,
              });
              nextBinding++;
            } else if (src.isTexture) {
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: 'texture2d',
                sourcePass: src.passIndex,
              });
              nextBinding++;
              pass.inputs.push({
                name: `${name}_sampler`,
                bindingIndex: nextBinding,
                type: 'sampler',
                sourcePass: src.passIndex,
              });
              nextBinding++;
            } else {
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: 'storageBufferRead',
                sourcePass: src.passIndex,
                sourceBuffer: src.bufferIndex,
              });
              nextBinding++;
            }
          }
        }
      }

      // Process output
      if (options?.output) {
        if (isTextureOutputToken(options.output)) {
          pass.textureOutput = true;
          const handle = { __resourceType: 'texture2d', __handle: -1 } as any;
          outputHandleMap.set(handle, { passIndex, bufferIndex: -1, isTexture: true });
          passes.push(pass);
          return handle as any;
        } else {
          // Buffer output (existing path)
          const ctor = options.output as TypedArrayConstructor;
          pass.buffer = { output: ctor, count: options.count! };
          const bufIdx = bufferMetas.length;
          bufferMetas.push({
            name: `__buf_${bufIdx}`,
            ctor,
          });
          const handle = {} as any;
          outputHandleMap.set(handle, { passIndex, bufferIndex: bufIdx, isTexture: false });
          passes.push(pass);
          return handle as any;
        }
      }

      passes.push(pass);
      return {} as any;
    },
    runModel(modelHandle: ModelResourceHandle) {
      const modelIndex = capturedModels.length;
      const mrh = modelHandle as ModelResourceHandle;
      capturedModels.push({
        path: mrh.__fileUri ?? '',
        inputShape: mrh.__modelOptions?.inputShape,
        normalization: mrh.__modelOptions?.normalization,
        sync: mrh.__modelOptions?.sync ?? false,
        // pipelineIndex is the model's position in the overall pipeline execution order
        // (same as passes.length — model occupies a "slot" like a shader pass)
        pipelineIndex: passes.length,
      });
      // Return a handle that downstream runShader calls can reference
      const handle = { __resourceType: 'texture2d', __handle: -1 } as any;
      outputHandleMap.set(handle, { passIndex: -1, bufferIndex: -1, isTexture: false, modelIndex });
      return handle as any;
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
    returnedHandles = pipelineFn(captureFrame, (resourceHandles as unknown) as R);
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

  return { passes, bufferMetas, hasCanvas, capturedResources, capturedModels };
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
  capturedResources: CapturedResource[],
  appleLog: boolean,
  capturedModels: CapturedModel[],
) {
  const shaders = passes.map((p) => p.wgsl);
  const buffers: [number, number, number][] = [];

  // The native side always prepends a pass 0 shader (YUV→RGB for Apple Log,
  // passthrough with rotation for sRGB), shifting all user shader indices by +1.
  const passOffset = 1;

  passes.forEach((pass, passIndex) => {
    if (pass.buffer) {
      const elementSize = pass.buffer.output.BYTES_PER_ELEMENT ?? 4;
      buffers.push([passIndex + passOffset, elementSize, pass.buffer.count]);
    }
  });

  // Collect pass indices that produce texture outputs
  const textureOutputPasses = passes
    .map((p, i) => p.textureOutput ? i + passOffset : -1)
    .filter((i) => i >= 0);

  // Build resources array for native
  const resources = capturedResources.map((r) => ({
    type: r.type,
    ...(r.fileUri ? { fileUri: r.fileUri } : r.data ? { data: r.data } : {}),
    width: r.width ?? 0,
    height: r.height ?? 0,
    depth: r.depth ?? 0,
    format: r.format ?? 'rgba8unorm',
  }));

  // Build per-pass input bindings for native
  const passInputs: {
    passIndex: number;
    bindings: {
      index: number;
      type: string;
      resourceHandle?: number;
      sourcePass?: number;
      sourceBuffer?: number;
      modelOutput?: number;
    }[];
  }[] = [];

  passes.forEach((pass, passIndex) => {
    if (pass.inputs && pass.inputs.length > 0) {
      passInputs.push({
        passIndex: passIndex + passOffset,
        bindings: pass.inputs.map((inp) => ({
          index: inp.bindingIndex,
          type: inp.type,
          resourceHandle: inp.resourceHandle,
          sourcePass: inp.sourcePass !== undefined ? inp.sourcePass + passOffset : undefined,
          sourceBuffer: inp.sourceBuffer,
          modelOutput: inp.modelOutput,
        })),
      });
    }
  });

  // Build models array for native — apply passOffset to pipelineIndex
  const models = capturedModels.map((m) => ({
    path: m.path,
    inputShape: m.inputShape,
    normalization: m.normalization,
    sync: m.sync,
    pipelineIndex: m.pipelineIndex + passOffset,
  }));

  // Log binding assignments at setup (format: name→3(texture3d)+4(sampler))
  passes.forEach((pass, i) => {
    if (pass.inputs && pass.inputs.length > 0) {
      const desc = pass.inputs
        .filter((inp) => inp.type !== 'sampler')
        .map((inp) => {
          const sampler = pass.inputs?.find((s) => s.name === `${inp.name}_sampler`);
          return sampler
            ? `${inp.name}→${inp.bindingIndex}(${inp.type})+${sampler.bindingIndex}(sampler)`
            : `${inp.name}→${inp.bindingIndex}(${inp.type})`;
        })
        .join(', ');
      console.log(`[WebGPUCamera] Pass ${i + passOffset} bindings: ${desc}`);
    }
  });

  const useDepth = capturedResources.some((r) => r.type === 'cameraDepth');

  return { shaders, width, height, buffers, useCanvas, sync, appleLog, resources, passInputs, textureOutputPasses, useDepth, models };
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
    ? (processorOrConfig as ProcessorConfig<any, any>).onFrame
    : undefined;
  const sync = isObjectForm
    ? ((processorOrConfig as ProcessorConfig<any, any>).sync ?? false)
    : false;
  const pipelineFn = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any, any>).pipeline
    : (processorOrConfig as FrameProcessor);
  const canvasRef = isObjectForm
    ? (processorOrConfig as ProcessorConfig<any, any>).canvasRef
    : undefined;

  // Buffer metadata shared with the worklet via shared values
  const bufferCount = useSharedValue(0);
  const bufferNames = useSharedValue<string[]>([]);
  const bufferCtorIndices = useSharedValue<number[]>([]);
  const hasOnFrame = useSharedValue(false);
  const useCanvasOutput = useSharedValue(false);

  // Setup compute pipeline when camera is ready
  useEffect(() => {
    if (!camera.isReady) return;

    // Capture shader chain and buffer declarations
    const resourcesConfig = isObjectForm
      ? (processorOrConfig as ProcessorConfig<any, any>).resources
      : undefined;

    const { passes, bufferMetas, hasCanvas, capturedResources, capturedModels } = capturePipeline(
      pipelineFn as (frame: ProcessorFrame, resources: any) => any,
      resourcesConfig,
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
    // Both appleLog and hlgBT2020 deliver 10-bit YUV — need YUV→RGB conversion shader
    const appleLog = camera.colorSpace === 'appleLog' || camera.colorSpace === 'hlgBT2020';
    const nativeConfig = buildNativeConfig(
      passes,
      camera.width,
      camera.height,
      useCanvas,
      sync,
      capturedResources,
      appleLog,
      capturedModels,
    );

    console.log(`[WebGPUCamera] setupMultiPassPipeline: appleLog=${appleLog}, ${nativeConfig.shaders.length} shaders, ${nativeConfig.resources.length} resources, ${nativeConfig.passInputs.length} passInputs`);
    for (const r of nativeConfig.resources) {
      const rAny = r as any;
      console.log(`[WebGPUCamera] resource: type=${r.type} format=${r.format} ${r.width}x${r.height}x${r.depth} dataBytes=${rAny.data?.byteLength ?? 'null'} fileUri=${rAny.fileUri ?? 'null'}`);
    }
    if (nativeConfig.passInputs.length > 0) {
      console.log(`[WebGPUCamera] passInputs:`, JSON.stringify(nativeConfig.passInputs.map(pi => ({ passIndex: pi.passIndex, bindings: pi.bindings.length }))));
    }
    if (nativeConfig.models.length > 0) {
      console.log(`[WebGPUCamera] models:`, JSON.stringify(nativeConfig.models.map(m => ({ path: m.path, pipelineIndex: m.pipelineIndex, sync: m.sync }))));
    }

    let ok: boolean;
    try {
      ok = WebGPUCameraModule.setupMultiPassPipeline(nativeConfig);
    } catch (e) {
      console.error(`[WebGPUCamera] setupMultiPassPipeline THREW:`, e);
      setError(`Pipeline setup exception: ${e}`);
      return;
    }
    if (!ok) {
      setError("Multi-pass pipeline setup failed");
      return;
    }
    setError(null);

    // Create stream host object — shared across Reanimated runtimes
    const s = globalThis.__webgpuCamera_createStream();
    stream.value = s;

    // Configure WebGPU canvas IMMEDIATELY after stream creation,
    // before the frame callback starts processing frames
    let canvasSetupAborted = false;
    if (canvasRef?.current) {
      const ref = canvasRef.current;
      const contextId = ref.getContextId();
      const ctx = ref.getContext("webgpu");

      if (ctx) {
        const device = Skia.getDevice();
        ctx.configure({
          device,
          format: 'rgba16float' as GPUTextureFormat,
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
          alphaMode: "opaque",
        });
        // setCanvasContextId will reconfigure the surface to match compute output dims
        s.setCanvasContextId(contextId);
        useCanvasOutput.value = true;
        console.log(`[WebGPUCamera] Canvas output configured: contextId=${contextId}, compute=${camera.height}x${camera.width}`);
      }
    }

    return () => {
      canvasSetupAborted = true;
      currentFrame.value?.dispose();
      currentFrame.value = null;
      stream.value = null;
      useCanvasOutput.value = false;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, [camera.isReady, camera.width, camera.height, camera.fps, camera.colorSpace]);

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
    const hasNewFrame = frame.generation !== lastGen.value;
    if (hasNewFrame) {
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

    // WebGPU canvas output: compute thread copies directly to surface + presents.
    // No JS-thread work needed — just skip the SkImage path.
    if (useCanvasOutput.value) {
      frame.image?.dispose();
      return;
    }

    // onFrame canvas path — always flush to produce the composited image
    if (hasOnFrame.value && onFrameFn && frame.canvas) {
      // Canvas is portrait-oriented (rotation done in GPU pass 0)
      const renderFrame = {
        canvas: frame.canvas,
        width: camera.height,
        height: camera.width,
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
