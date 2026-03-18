# Custom Bind Group Support — Design Spec

## Summary

Extend the WebGPU compute pipeline to support custom shader inputs beyond the fixed binding(0)=input texture, binding(1)=output texture, binding(2)=optional output buffer. Enables 3D textures (LUT), 2D textures (depth maps, masks), and storage buffer cross-references between passes — all flowing through a unified `{ inputs: { ... } }` API with no string references.

## Input Types

Three types of custom inputs, all passed via `{ inputs: { ... } }` on `runShader`:

| Input type | Source | GPU binding |
|---|---|---|
| 3D texture + sampler | `resources` block | `texture_3d<f32>` + `sampler` (2 binding slots) |
| 2D texture + sampler | `resources` block or shader output | `texture_2d<f32>` + `sampler` (2 binding slots) |
| Storage buffer (read) | Shader output from previous pass, or `resources` block | `storage<read>` buffer (1 binding slot) |

## API

### GPUResource constructors

```ts
const GPUResource = {
  // For resources block — takes data + dimensions, uploaded once at setup
  texture3D(data: ArrayBuffer, dims: { width: number; height: number; depth: number }): ResourceHandle<'texture3d'>;
  texture2D(data: ArrayBuffer, dims: { width: number; height: number }): ResourceHandle<'texture2d'>;
  storageBuffer(data: ArrayBuffer): ResourceHandle<'storageBuffer'>;

  // As output type tokens — no args, declares shader output type
  texture2D: OutputTypeToken<'texture2d'>;
};
```

`texture3D` is only used in `resources` (static upload). `texture2D` doubles as both a resource constructor (with args) and an output type token (no args). `storageBuffer` can be used in `resources` for static data (e.g., precomputed weights) that flows into shaders via `inputs`, or it appears implicitly as a pass output handle.

Example with a static storage buffer resource:
```ts
resources: {
  weights: GPUResource.storageBuffer(weightData),
},
pipeline: (frame, { weights }) => {
  frame.runShader(WEIGHTED_BLUR_WGSL, { inputs: { weights } });
},
```

### Full usage

```ts
useGPUFrameProcessor(camera, {
  resources: {
    lut: GPUResource.texture3D(cubeData, { width: 33, height: 33, depth: 33 }),
  },
  pipeline: (frame, { lut }) => {
    'worklet';
    frame.runShader(LUT_WGSL, { inputs: { lut } });
    const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
    return { hist };
  },
  overlay: (frame, { hist }) => {
    'worklet';
    frame.runShader(HISTOGRAM_OVERLAY_WGSL, { inputs: { hist } });
    frame.runShader(ZEBRA_WGSL);
  },
  onFrame: (frame, { hist }) => {
    'worklet';
    if (hist) {
      frame.canvas.drawText(`peak: ${Math.max(...hist)}`, 100, 100);
    }
  },
});
```

### Shader output declaration

Each `runShader` call produces at most one side output (in addition to the implicit output texture at binding 1):

```ts
// Buffer output — same as today
const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });

// Texture output — new
const depth = frame.runShader(DEPTH_WGSL, { output: GPUResource.texture2D });
```

Both return typed handles that flow into `inputs` on subsequent passes.

## Binding Index Assignment

Fixed bindings (unchanged):
- **0**: input texture (`texture_2d<f32>`)
- **1**: output storage texture (`texture_storage_2d<rgba8unorm, write>`)
- **2**: optional output buffer (`storage<read_write>`)

Custom inputs start at **binding 3**, assigned sequentially by the capture proxy based on `inputs` declaration order:

```ts
frame.runShader(WGSL, { inputs: { lut, hist } });
// lut (texture3d) → binding 3 (texture) + binding 4 (sampler)
// hist (buffer)   → binding 5
```

A 3D or 2D texture consumes two binding slots (texture + sampler). A storage buffer consumes one slot.

The capture proxy logs the mapping at setup time so the developer knows which indices to use in WGSL:
```
[WebGPUCamera] Pass 1 bindings: lut→3(texture3d)+4(sampler), hist→5(storageRead)
```

### Corresponding WGSL

LUT shader:
```wgsl
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var lutTex: texture_3d<f32>;
@group(0) @binding(4) var lutSampler: sampler;
```

Histogram overlay shader:
```wgsl
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> histogram: array<u32, 256>;
```

## Native Implementation

### Resource upload (setup time)

The `resources` block is processed during pipeline setup. For each resource:

- **texture3D**: Create `wgpu::Texture` with `TextureDimension::e3D`, upload data via `queue.WriteTexture()`, create trilinear sampler
- **texture2D**: Create `wgpu::Texture` with `TextureDimension::e2D`, upload data, create linear sampler
- **storageBuffer**: Create `wgpu::Buffer` with `BufferUsage::Storage | BufferUsage::CopyDst`, upload data via `queue.WriteBuffer()`

All created GPU objects stored in a `resources` vector on the native pipeline, indexed by handle.

### Per-pass input config (JS → native)

The capture proxy sends extra binding info per pass alongside the existing shader/buffer config:

```ts
// Added to setupMultiPassPipeline config
passInputs: [
  // pass 1: LUT texture
  { passIndex: 1, bindings: [
    { index: 3, type: 'texture3d', resourceHandle: 0 },
    { index: 4, type: 'sampler', pairedWith: 3 },
  ]},
  // pass 2: histogram buffer from pass 0
  { passIndex: 2, bindings: [
    { index: 3, type: 'storageBufferRead', sourcePass: 0, sourceBuffer: 0 },
  ]},
]
```

### Bind group creation changes (DawnComputePipeline.mm)

Currently each pass builds a bind group with entries for bindings 0, 1, and optionally 2. The extension:

1. After the standard entries, appends extra `wgpu::BindGroupEntry` items for each declared input
2. For resource textures: entry points to the pre-uploaded texture's view
3. For resource samplers: entry points to the pre-created sampler
4. For buffer cross-references: entry points to the same `wgpu::Buffer` from the source pass. The consuming shader declares it as `var<storage, read>`, which produces a `ReadOnlyStorage` layout entry via `GetBindGroupLayout(0)`. Dawn validates that the buffer was created with `BufferUsage::Storage` — this is already the case since output buffers are created with `Storage | CopySrc | CopyDst`

The bind group layout is still auto-derived from the compiled shader via `GetBindGroupLayout(0)` — no manual layout creation needed. The native side just fills entries to match.

### Resource config (JS → native)

```ts
// Added to setupMultiPassPipeline config
resources: [
  { type: 'texture3d', width: 33, height: 33, depth: 33, data: ArrayBuffer },
  { type: 'texture2d', width: 1920, height: 1080, data: ArrayBuffer },
  { type: 'storageBuffer', data: ArrayBuffer },
]
```

### Texture output handles

When a shader declares `{ output: GPUResource.texture2D }`, the native side:
1. Creates an additional output texture (same dimensions as the pipeline) with `TextureUsage::StorageBinding | TextureUsage::TextureBinding`
2. Adds it as a `texture_storage_2d<rgba8unorm, write>` binding at **binding 2** in the shader. This is the same slot as the optional buffer output — a pass can have either a buffer output or a texture output, not both. Each shader is compiled into its own pipeline object, so `GetBindGroupLayout(0)` reflects the correct type (storage texture vs storage buffer) per shader.
3. Stores it in the pass state so subsequent passes can reference it via `inputs` (bound as a sampled `texture_2d<f32>` + `sampler`)

## TypeScript Types

```ts
interface ResourceHandle<T extends string> {
  readonly __resourceType: T;
  readonly __handle: number;
}

interface OutputTypeToken<T extends string> {
  readonly __outputType: T;
}

// GPUResource namespace
declare const GPUResource: {
  texture3D(data: ArrayBuffer, dims: { width: number; height: number; depth: number }): ResourceHandle<'texture3d'>;
  texture2D: {
    (data: ArrayBuffer, dims: { width: number; height: number }): ResourceHandle<'texture2d'>;
    readonly __outputType: 'texture2d';  // doubles as output type token
  };
  storageBuffer(data: ArrayBuffer): ResourceHandle<'storageBuffer'>;
};

// Updated ProcessorConfig — R is the resources type, B is the buffer returns type
interface ProcessorConfig<B extends Record<string, any>, R extends Record<string, ResourceHandle<any>> = {}> {
  sync?: boolean;
  resources?: R;
  pipeline: (frame: ProcessorFrame, resources: R) => B;
  overlay?: (frame: ProcessorFrame, buffers: NullableBuffers<B>) => void;
  onFrame?: (frame: RenderFrame, buffers: NullableBuffers<B>) => void;
}

// Updated runShader overloads
interface ProcessorFrame {
  runShader(wgsl: string): void;
  runShader(wgsl: string, options: { inputs: Record<string, ResourceHandle<any>> }): void;
  runShader<T extends TypedArrayConstructor>(
    wgsl: string,
    options: { output: T; count: number; inputs?: Record<string, ResourceHandle<any>> },
  ): BufferHandle<InstanceType<T>>;
  runShader(
    wgsl: string,
    options: { output: typeof GPUResource.texture2D; inputs?: Record<string, ResourceHandle<any>> },
  ): ResourceHandle<'texture2d'>;
}
```

## Capture Proxy Changes

The capture proxy in `useGPUFrameProcessor.ts` currently records `CapturedPass` with `{ wgsl, buffer? }`. Extended to:

```ts
interface CapturedPass {
  wgsl: string;
  buffer?: { output: TypedArrayConstructor; count: number };
  textureOutput?: boolean;  // true if output is GPUResource.texture2D
  inputs?: CapturedInput[];
}

interface CapturedInput {
  name: string;
  bindingIndex: number;  // assigned sequentially from 3
  type: 'texture3d' | 'texture2d' | 'sampler' | 'storageBufferRead';
  resourceHandle?: number;  // for resources block items
  sourcePass?: number;      // for buffer/texture from previous pass
  sourceBuffer?: number;    // global buffer index (matching the flat buffers[] array in native config)
}
```

**Capture proxy flow:**

1. Before calling `pipeline(captureFrame, resourceHandles)`, the proxy constructs a resources argument object by mapping each key in `config.resources` to a `ResourceHandle` with an assigned index
2. During `runShader` interception, if `options.inputs` is present, the proxy inspects each handle's `__resourceType` to determine its type and assigns binding indices starting at 3
3. For pass-output handles (returned by previous `runShader` calls), the proxy records `sourcePass` + `sourceBuffer` references instead of `resourceHandle`
4. After capture completes, the proxy builds the `passInputs` and `resources` arrays for the native config

## Files to Create/Modify

- `packages/react-native-webgpu-camera/src/GPUResource.ts` — new: GPUResource constructors and type definitions
- `packages/react-native-webgpu-camera/src/types.ts` — update ProcessorFrame, add ResourceHandle types
- `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` — extend capture proxy to handle resources/inputs, extend native config builder
- `packages/react-native-webgpu-camera/src/index.ts` — export GPUResource
- `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts` — extend setupMultiPassPipeline config type
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` — resource upload, extended bind group creation, texture outputs
- `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` — resource storage, extended PassState

## Scope Boundaries

**In scope:**
- Resource upload (3D texture, 2D texture, storage buffer)
- Custom input bindings on `runShader`
- Buffer cross-references between passes
- Texture output handles
- Auto-assigned binding indices with logging

**Out of scope (future):**
- Pipeline/overlay split (separate spec, consumes this)
- LUT file parsing (.cube format) — separate utility
- Dynamic resource updates (resources are static after setup)
- Multiple bind groups (group 1+) — everything stays in group 0
