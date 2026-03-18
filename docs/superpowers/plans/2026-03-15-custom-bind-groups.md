# Custom Bind Group Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the WebGPU compute pipeline to support custom shader inputs (3D textures, 2D textures, storage buffers) via `{ inputs: { ... } }` on `runShader`, with resources uploaded once at setup time.

**Architecture:** JS-side `GPUResource` constructors create typed handles. The capture proxy assigns binding indices (3+) during pipeline capture and sends resource specs + per-pass input bindings to native. Native creates GPU objects at setup and appends extra bind group entries alongside the existing bindings 0-2.

**Tech Stack:** TypeScript, C++ (Dawn/WebGPU), Objective-C++ bridge, Swift (Expo Modules)

**Spec:** `docs/superpowers/specs/2026-03-15-custom-bind-groups-design.md`

**Scope note:** The spec's usage example shows an `overlay` callback receiving resource handles. The `overlay` callback is part of the "Pipeline/overlay split" feature listed in the spec's out-of-scope section and will be implemented separately. This plan implements the core resource/input infrastructure using the existing `onFrame` callback.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/react-native-webgpu-camera/src/GPUResource.ts` | Create | GPUResource constructors, ResourceHandle types |
| `packages/react-native-webgpu-camera/src/types.ts` | Modify | Update ProcessorFrame (inputs param), ProcessorConfig (resources + 2nd arg), ResourceHandle types |
| `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts` | Modify | Extend capture proxy for resources/inputs, extend buildNativeConfig |
| `packages/react-native-webgpu-camera/src/index.ts` | Modify | Export GPUResource |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts` | Modify | Extend setupMultiPassPipeline config type |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift` | Modify | Pass resources + passInputs through to bridge |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h` | Modify | Add resources + passInputs params to setup method |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm` | Modify | Convert resources/passInputs to C++ types |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h` | Modify | Add ResourceSpec, InputBinding structs, extend setup() |
| `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm` | Modify | Resource upload, extended bind group creation |

---

## Chunk 1: TypeScript Types and GPUResource

### Task 1: Create GPUResource module

**Files:**
- Create: `packages/react-native-webgpu-camera/src/GPUResource.ts`

- [ ] **Step 1: Create GPUResource.ts**

```ts
/** Type tag for resource handles */
export interface ResourceHandle<T extends string> {
  readonly __resourceType: T;
  readonly __handle: number;
  readonly __data?: ArrayBuffer;
  readonly __dims?: { width: number; height: number; depth?: number };
}

/** Sentinel for output type tokens */
export interface OutputTypeToken<T extends string> {
  readonly __outputType: T;
}

function texture3D(
  data: ArrayBuffer,
  dims: { width: number; height: number; depth: number },
): ResourceHandle<'texture3d'> {
  return {
    __resourceType: 'texture3d',
    __handle: -1, // assigned by capture proxy
    __data: data,
    __dims: dims,
  };
}

function texture2DResource(
  data: ArrayBuffer,
  dims: { width: number; height: number },
): ResourceHandle<'texture2d'> {
  return {
    __resourceType: 'texture2d',
    __handle: -1, // assigned by capture proxy
    __data: data,
    __dims: dims,
  };
}

function storageBuffer(data: ArrayBuffer): ResourceHandle<'storageBuffer'> {
  return {
    __resourceType: 'storageBuffer',
    __handle: -1, // assigned by capture proxy
    __data: data,
  };
}

/** Token used as output type: frame.runShader(WGSL, { output: GPUResource.texture2D }) */
const texture2DToken: OutputTypeToken<'texture2d'> = {
  __outputType: 'texture2d',
};

/**
 * GPUResource constructors for creating typed GPU handles.
 *
 * In `resources` block: GPUResource.texture3D(data, dims) — uploads once at setup
 * As output type: GPUResource.texture2D (no args) — declares shader output type
 */
export const GPUResource = {
  texture3D,
  /** Call with (data, dims) for resource upload, or use without args as output type token */
  texture2D: Object.assign(texture2DResource, texture2DToken) as {
    (data: ArrayBuffer, dims: { width: number; height: number }): ResourceHandle<'texture2d'>;
    readonly __outputType: 'texture2d';
  },
  storageBuffer,
};

/** Type guard: is this a ResourceHandle? */
export function isResourceHandle(v: any): v is ResourceHandle<any> {
  return v != null && typeof v === 'object' && '__resourceType' in v;
}

/** Type guard: is this a texture output token? */
export function isTextureOutputToken(v: any): v is OutputTypeToken<any> {
  return v != null && typeof v === 'object' && '__outputType' in v;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/GPUResource.ts
git commit -m "feat: add GPUResource constructors and handle types"
```

---

### Task 2: Update TypeScript types

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/types.ts`

- [ ] **Step 1: Update ProcessorFrame to accept inputs**

Read `types.ts` first. Update the `ProcessorFrame` interface to add `inputs` to runShader options. Replace the existing `ProcessorFrame` interface with:

```ts
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
```

Add the import for ResourceHandle at the top of the file (it will be exported from GPUResource.ts but re-exported from types):

```ts
import type { ResourceHandle } from './GPUResource';
```

And re-export it:

```ts
export type { ResourceHandle } from './GPUResource';
```

- [ ] **Step 2: Update ProcessorConfig to include resources**

Replace the existing `ProcessorConfig` interface with:

```ts
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
```

- [ ] **Step 3: Verify types compile**

Run: `cd /Users/kim/dev/rn-webgpu-camera && /Users/kim/.bun/bin/bunx tsc --noEmit 2>&1 | grep -E "types\.ts|GPUResource" | grep -v "Cannot find module"`

Expected: no new errors from our files.

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/src/types.ts
git commit -m "feat: add inputs/resources to ProcessorFrame and ProcessorConfig"
```

---

### Task 3: Update exports

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/index.ts`

- [ ] **Step 1: Export GPUResource**

Add to the exports:

```ts
export { GPUResource } from './GPUResource';
export type { ResourceHandle } from './GPUResource';
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/src/index.ts
git commit -m "feat: export GPUResource and ResourceHandle"
```

---

## Chunk 2: Capture Proxy Extension

### Task 4: Extend capture proxy for resources and inputs

**Files:**
- Modify: `packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts`

This is the most complex JS-side change. The capture proxy must:
1. Accept a `resources` block and pass handles to the pipeline callback
2. Track `inputs` on each `runShader` call and assign binding indices from 3+
3. Track texture outputs
4. Build extended native config with resource specs and per-pass input bindings

- [ ] **Step 1: Add new interfaces for captured inputs**

Add after the existing `CapturedPass` interface (after line 23):

```ts
/** A captured custom input binding for a pass */
interface CapturedInput {
  name: string;
  bindingIndex: number;
  type: 'texture3d' | 'texture2d' | 'sampler' | 'storageBufferRead';
  resourceHandle?: number;
  sourcePass?: number;
  sourceBuffer?: number;
}

/** A resource spec to send to native for GPU upload */
interface CapturedResource {
  type: 'texture3d' | 'texture2d' | 'storageBuffer';
  data: ArrayBuffer;
  width?: number;
  height?: number;
  depth?: number;
}
```

- [ ] **Step 2: Extend CapturedPass interface**

Update the existing `CapturedPass` interface to:

```ts
interface CapturedPass {
  wgsl: string;
  buffer?: {
    output: TypedArrayConstructor;
    count: number;
  };
  textureOutput?: boolean;
  inputs?: CapturedInput[];
}
```

- [ ] **Step 3: Add resource processing to capturePipeline**

The current `capturePipeline` function (lines 35-88) takes `pipelineFn: (frame: ProcessorFrame) => B`. It needs to:
1. Accept a `resources` record
2. Build ResourceHandle objects for each resource
3. Pass them as the second arg to pipelineFn
4. Track captured resources for native config

Replace the entire `capturePipeline` function with:

```ts
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
} {
  const passes: CapturedPass[] = [];
  const bufferMetas: BufferMeta[] = [];
  const capturedResources: CapturedResource[] = [];
  let hasCanvas = false;

  // Map resource handle identity → index into capturedResources
  const handleToIndex = new Map<any, number>();

  // Build resource handle map and collect resource specs for native
  const resourceHandles: Record<string, any> = {};
  if (resources) {
    for (const [name, handle] of Object.entries(resources)) {
      if (isResourceHandle(handle)) {
        const rh = handle as ResourceHandle<any>;
        const idx = capturedResources.length;
        // Store resource spec for native upload
        capturedResources.push({
          type: rh.__resourceType as CapturedResource['type'],
          data: rh.__data!,
          width: rh.__dims?.width,
          height: rh.__dims?.height,
          depth: rh.__dims?.depth,
        });
        handleToIndex.set(handle, idx);
        resourceHandles[name] = handle;
      }
    }
  }

  // Track pass-output handles: map handle → { passIndex, bufferIndex }
  const outputHandleMap = new Map<any, { passIndex: number; bufferIndex: number; isTexture: boolean }>();

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
            if (rh.__resourceType === 'texture3d' || rh.__resourceType === 'texture2d') {
              pass.inputs.push({
                name,
                bindingIndex: nextBinding,
                type: rh.__resourceType,
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
            // Buffer/texture output from a previous pass
            const src = outputHandleMap.get(handle)!;
            if (src.isTexture) {
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

  return { passes, bufferMetas, hasCanvas, capturedResources };
}
```

Add the import at the top of the file:

```ts
import { isResourceHandle, isTextureOutputToken } from './GPUResource';
import type { ResourceHandle } from './GPUResource';
```

- [ ] **Step 4: Extend buildNativeConfig**

Replace the existing `buildNativeConfig` function with:

```ts
function buildNativeConfig(
  passes: CapturedPass[],
  width: number,
  height: number,
  useCanvas: boolean,
  sync: boolean,
  capturedResources: CapturedResource[],
) {
  const shaders = passes.map((p) => p.wgsl);
  const buffers: [number, number, number][] = [];

  passes.forEach((pass, passIndex) => {
    if (pass.buffer) {
      const elementSize = pass.buffer.output.BYTES_PER_ELEMENT ?? 4;
      buffers.push([passIndex, elementSize, pass.buffer.count]);
    }
  });

  // Collect pass indices that produce texture outputs
  const textureOutputPasses = passes
    .map((p, i) => p.textureOutput ? i : -1)
    .filter((i) => i >= 0);

  // Build resources array for native
  const resources = capturedResources.map((r) => ({
    type: r.type,
    data: r.data,
    width: r.width ?? 0,
    height: r.height ?? 0,
    depth: r.depth ?? 0,
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
    }[];
  }[] = [];

  passes.forEach((pass, passIndex) => {
    if (pass.inputs && pass.inputs.length > 0) {
      passInputs.push({
        passIndex,
        bindings: pass.inputs.map((inp) => ({
          index: inp.bindingIndex,
          type: inp.type,
          resourceHandle: inp.resourceHandle,
          sourcePass: inp.sourcePass,
          sourceBuffer: inp.sourceBuffer,
        })),
      });
    }
  });

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
      console.log(`[WebGPUCamera] Pass ${i} bindings: ${desc}`);
    }
  });

  return { shaders, width, height, buffers, useCanvas, sync, resources, passInputs, textureOutputPasses };
}
```

- [ ] **Step 5: Update the hook's setup effect to pass resources**

In the `useEffect` that calls `capturePipeline` (around line 202), update to:
1. Extract `resources` from `processorOrConfig`
2. Pass resources to `capturePipeline`
3. Pass `capturedResources` to `buildNativeConfig`

Find this code block in the hook:

```ts
const { passes, bufferMetas, hasCanvas } = capturePipeline(
  pipelineFn as (frame: ProcessorFrame) => any,
  camera.width,
  camera.height,
);
```

Replace with:

```ts
const resourcesConfig = isObjectForm
  ? (processorOrConfig as ProcessorConfig<any, any>).resources
  : undefined;

const { passes, bufferMetas, hasCanvas, capturedResources } = capturePipeline(
  pipelineFn as (frame: ProcessorFrame, resources: any) => any,
  resourcesConfig,
  camera.width,
  camera.height,
);
```

And update the `buildNativeConfig` call:

```ts
const nativeConfig = buildNativeConfig(
  passes,
  camera.width,
  camera.height,
  useCanvas,
  sync,
  capturedResources,
);
```

- [ ] **Step 6: Verify types compile**

Run: `cd /Users/kim/dev/rn-webgpu-camera && /Users/kim/.bun/bin/bunx tsc --noEmit 2>&1 | grep -E "useGPUFrameProcessor|GPUResource|types\.ts" | grep -v "Cannot find module"`

Expected: no new errors from our files (there may be pre-existing react type warnings).

- [ ] **Step 7: Commit**

```bash
git add packages/react-native-webgpu-camera/src/useGPUFrameProcessor.ts
git commit -m "feat: extend capture proxy for resources and custom inputs"
```

---

## Chunk 3: Native Module Interface + Bridge

### Task 5: Extend native module TypeScript interface

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`

- [ ] **Step 1: Update setupMultiPassPipeline config type**

Read the file first. Update the `setupMultiPassPipeline` config parameter to include `resources` and `passInputs`:

```ts
setupMultiPassPipeline(config: {
  shaders: string[];
  width: number;
  height: number;
  buffers: [number, number, number][];
  useCanvas: boolean;
  sync: boolean;
  resources: {
    type: string;
    data: ArrayBuffer;
    width: number;
    height: number;
    depth: number;
  }[];
  passInputs: {
    passIndex: number;
    bindings: {
      index: number;
      type: string;
      resourceHandle?: number;
      sourcePass?: number;
      sourceBuffer?: number;
    }[];
  }[];
  textureOutputPasses: number[];
}): boolean;
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts
git commit -m "feat: extend native module interface for resources and passInputs"
```

---

### Task 6: Update Swift module to pass resources/passInputs

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Extract resources and passInputs from config**

Read the file first. Find the `setupMultiPassPipeline` function. It currently extracts `shaders`, `width`, `height`, `buffers`, `useCanvas`, `sync` from the config dictionary.

After the existing extraction code, add extraction for `resources` and `passInputs`:

```swift
let resourcesRaw = config["resources"] as? [[String: Any]] ?? []
let passInputsRaw = config["passInputs"] as? [[String: Any]] ?? []
let textureOutputPasses = (config["textureOutputPasses"] as? [NSNumber] ?? []).map { $0.intValue }
```

- [ ] **Step 2: Update bridge call to pass resources and passInputs**

Update the bridge call from:

```swift
let ok = bridge.setupMultiPass(
    withShaders: shaders,
    width: Int32(width),
    height: Int32(height),
    bufferSpecs: bufferSpecs,
    useCanvas: useCanvas,
    sync: sync
)
```

To:

```swift
let ok = bridge.setupMultiPass(
    withShaders: shaders,
    width: Int32(width),
    height: Int32(height),
    bufferSpecs: bufferSpecs,
    useCanvas: useCanvas,
    sync: sync,
    resources: resourcesRaw,
    passInputs: passInputsRaw,
    textureOutputPasses: textureOutputPasses
)
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift
git commit -m "feat: pass resources and passInputs through Swift module"
```

---

### Task 7: Update Objective-C++ bridge

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h`
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm`

- [ ] **Step 1: Update bridge header**

Read `DawnPipelineBridge.h`. Add `resources` and `passInputs` parameters to the setup method:

```objc
- (BOOL)setupMultiPassWithShaders:(nonnull NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(nonnull NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                        resources:(nonnull NSArray<NSDictionary *> *)resources
                       passInputs:(nonnull NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(nonnull NSArray<NSNumber *> *)textureOutputPasses;
```

- [ ] **Step 2: Update bridge implementation**

Read `DawnPipelineBridge.mm`. Update the method to convert resources and passInputs to C++ structs and pass them to the C API.

Replace the setup method implementation:

```objc
- (BOOL)setupMultiPassWithShaders:(NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                        resources:(NSArray<NSDictionary *> *)resources
                       passInputs:(NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(NSArray<NSNumber *> *)textureOutputPasses {
  if (!_pipeline) return NO;

  // Convert shaders (unchanged)
  int shaderCount = (int)shaders.count;
  std::vector<const char*> cShaders(shaderCount);
  std::vector<std::string> shaderStorage(shaderCount);
  for (int i = 0; i < shaderCount; i++) {
    shaderStorage[i] = [shaders[i] UTF8String];
    cShaders[i] = shaderStorage[i].c_str();
  }

  // Convert buffer specs (unchanged)
  int bufferCount = (int)bufferSpecs.count;
  std::vector<int> flatSpecs(bufferCount * 3);
  for (int i = 0; i < bufferCount; i++) {
    NSArray<NSNumber *> *spec = bufferSpecs[i];
    flatSpecs[i * 3 + 0] = [spec[0] intValue];
    flatSpecs[i * 3 + 1] = [spec[1] intValue];
    flatSpecs[i * 3 + 2] = [spec[2] intValue];
  }

  // Convert resources to C++ structs
  std::vector<ResourceSpec> resourceSpecs;
  for (NSDictionary *res in resources) {
    ResourceSpec rs;
    NSString *type = res[@"type"];
    if ([type isEqualToString:@"texture3d"]) {
      rs.type = ResourceType::Texture3D;
    } else if ([type isEqualToString:@"texture2d"]) {
      rs.type = ResourceType::Texture2D;
    } else {
      rs.type = ResourceType::StorageBuffer;
    }
    rs.width = [res[@"width"] intValue];
    rs.height = [res[@"height"] intValue];
    rs.depth = [res[@"depth"] intValue];
    // Copy data to owned buffer (NSData may be released after this scope)
    NSData *data = res[@"data"];
    if (data) {
      const uint8_t *bytes = (const uint8_t *)data.bytes;
      rs.data.assign(bytes, bytes + data.length);
    }
    resourceSpecs.push_back(rs);
  }

  // Convert passInputs to C++ structs
  std::vector<PassInputSpec> passInputSpecs;
  for (NSDictionary *pi in passInputs) {
    PassInputSpec pis;
    pis.passIndex = [pi[@"passIndex"] intValue];
    NSArray<NSDictionary *> *bindings = pi[@"bindings"];
    for (NSDictionary *b in bindings) {
      InputBinding ib;
      ib.bindingIndex = [b[@"index"] intValue];
      NSString *btype = b[@"type"];
      if ([btype isEqualToString:@"texture3d"]) {
        ib.type = InputBindingType::Texture3D;
      } else if ([btype isEqualToString:@"texture2d"]) {
        ib.type = InputBindingType::Texture2D;
      } else if ([btype isEqualToString:@"sampler"]) {
        ib.type = InputBindingType::Sampler;
      } else {
        ib.type = InputBindingType::StorageBufferRead;
      }
      ib.resourceHandle = b[@"resourceHandle"] ? [b[@"resourceHandle"] intValue] : -1;
      ib.sourcePass = b[@"sourcePass"] ? [b[@"sourcePass"] intValue] : -1;
      ib.sourceBuffer = b[@"sourceBuffer"] ? [b[@"sourceBuffer"] intValue] : -1;
      pis.bindings.push_back(ib);
    }
    passInputSpecs.push_back(pis);
  }

  // Convert textureOutputPasses
  std::vector<int> texOutPasses;
  for (NSNumber *n in textureOutputPasses) {
    texOutPasses.push_back([n intValue]);
  }

  return dawn_pipeline_setup_multipass(
    _pipeline,
    cShaders.data(), shaderCount,
    width, height,
    flatSpecs.data(), bufferCount,
    useCanvas, sync,
    resourceSpecs.data(), (int)resourceSpecs.size(),
    passInputSpecs.data(), (int)passInputSpecs.size(),
    texOutPasses.data(), (int)texOutPasses.size()
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.h \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnPipelineBridge.mm
git commit -m "feat: extend bridge to pass resources and passInputs to C++"
```

---

## Chunk 4: C++ Pipeline Extension

### Task 8: Add resource and input binding structs to header

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h`

- [ ] **Step 1: Add new structs and enums**

Read the header file. Add these definitions before the `BufferSpec` struct:

```cpp
enum class ResourceType { Texture3D, Texture2D, StorageBuffer };

struct ResourceSpec {
  ResourceType type;
  std::vector<uint8_t> data;  // owns a copy of the upload data
  int width = 0;
  int height = 0;
  int depth = 0;
};

enum class InputBindingType { Texture3D, Texture2D, Sampler, StorageBufferRead };

struct InputBinding {
  int bindingIndex = 0;
  InputBindingType type;
  int resourceHandle = -1;  // index into resources array
  int sourcePass = -1;      // pass that produced this buffer/texture
  int sourceBuffer = -1;    // global buffer index
};

struct PassInputSpec {
  int passIndex = 0;
  std::vector<InputBinding> bindings;
};
```

- [ ] **Step 1b: Add hasTextureOutput flag to PassState**

In the existing `PassState` struct (alongside the existing `hasOutputBuffer` field), add:

```cpp
bool hasTextureOutput = false;  // pass outputs a texture at binding 2
```

- [ ] **Step 2: Update setup() signature**

Update the `setup` method declaration to accept resources and passInputs:

```cpp
bool setup(const std::vector<std::string>& wgslShaders,
           int width, int height,
           const std::vector<BufferSpec>& bufferSpecs,
           bool useCanvas, bool sync,
           const std::vector<ResourceSpec>& resources = {},
           const std::vector<PassInputSpec>& passInputs = {},
           const std::vector<int>& textureOutputPasses = {});
```

- [ ] **Step 3: Update C API function signature**

Update the `dawn_pipeline_setup_multipass` C function:

```c
bool dawn_pipeline_setup_multipass(
  DawnComputePipelineRef ref,
  const char** shaders, int shaderCount,
  int width, int height,
  const int* bufferSpecsFlat, int bufferCount,
  bool useCanvas, bool sync,
  const ResourceSpec* resources, int resourceCount,
  const PassInputSpec* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount);
```

- [ ] **Step 4: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h
git commit -m "feat: add ResourceSpec, InputBinding structs to pipeline header"
```

---

### Task 9: Implement resource upload and extended bind groups

**Files:**
- Modify: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm`

This is the core native change. We need to:
1. Add resource storage to the Impl struct
2. Upload resources at setup time
3. Append custom input entries to bind groups

- [ ] **Step 1: Add resource storage to Impl struct**

Add to the `Impl` struct (after the existing fields, around line 74):

```cpp
// Custom resources
struct UploadedResource {
  ResourceType type;
  wgpu::Texture texture;
  wgpu::TextureView textureView;
  wgpu::Sampler sampler;
  wgpu::Buffer buffer;
};
std::vector<UploadedResource> uploadedResources;

// Default linear sampler for texture outputs (always created if any custom inputs exist)
wgpu::Sampler defaultSampler;

// Per-pass custom input bindings (indexed by pass index)
std::vector<std::vector<InputBinding>> passInputBindings;

// Texture outputs from passes (indexed by pass index)
std::vector<wgpu::Texture> passTextureOutputs;
```

- [ ] **Step 2: Add resource upload in setup()**

After the ping-pong texture creation (around line 128) and before shader compilation, add resource upload:

```cpp
// Create default linear sampler (used for texture outputs from previous passes)
if (!passInputs.empty()) {
  wgpu::SamplerDescriptor samplerDesc{};
  samplerDesc.magFilter = wgpu::FilterMode::Linear;
  samplerDesc.minFilter = wgpu::FilterMode::Linear;
  samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
  samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
  _impl->defaultSampler = _impl->device.CreateSampler(&samplerDesc);
}

// Upload custom resources
for (int i = 0; i < (int)resources.size(); i++) {
  const auto& spec = resources[i];
  Impl::UploadedResource ur;
  ur.type = spec.type;

  if (spec.type == ResourceType::Texture3D) {
    wgpu::TextureDescriptor texDesc{};
    texDesc.dimension = wgpu::TextureDimension::e3D;
    texDesc.size = {(uint32_t)spec.width, (uint32_t)spec.height, (uint32_t)spec.depth};
    texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    texDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    texDesc.mipLevelCount = 1;
    ur.texture = _impl->device.CreateTexture(&texDesc);
    ur.textureView = ur.texture.CreateView();

    // Upload data
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = ur.texture;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = spec.width * 4; // RGBA8
    layout.rowsPerImage = spec.height;
    wgpu::Extent3D extent = {(uint32_t)spec.width, (uint32_t)spec.height, (uint32_t)spec.depth};
    _impl->device.GetQueue().WriteTexture(&dst, spec.data.data(), spec.data.size(), &layout, &extent);

    // Trilinear sampler
    wgpu::SamplerDescriptor samplerDesc{};
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    samplerDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeW = wgpu::AddressMode::ClampToEdge;
    ur.sampler = _impl->device.CreateSampler(&samplerDesc);

  } else if (spec.type == ResourceType::Texture2D) {
    wgpu::TextureDescriptor texDesc{};
    texDesc.dimension = wgpu::TextureDimension::e2D;
    texDesc.size = {(uint32_t)spec.width, (uint32_t)spec.height, 1};
    texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    texDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    texDesc.mipLevelCount = 1;
    ur.texture = _impl->device.CreateTexture(&texDesc);
    ur.textureView = ur.texture.CreateView();

    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = ur.texture;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = spec.width * 4;
    layout.rowsPerImage = spec.height;
    wgpu::Extent3D extent = {(uint32_t)spec.width, (uint32_t)spec.height, 1};
    _impl->device.GetQueue().WriteTexture(&dst, spec.data.data(), spec.data.size(), &layout, &extent);

    // Linear sampler
    wgpu::SamplerDescriptor samplerDesc{};
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    ur.sampler = _impl->device.CreateSampler(&samplerDesc);

  } else if (spec.type == ResourceType::StorageBuffer) {
    wgpu::BufferDescriptor bufDesc{};
    bufDesc.size = spec.data.size();
    bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    ur.buffer = _impl->device.CreateBuffer(&bufDesc);
    _impl->device.GetQueue().WriteBuffer(ur.buffer, 0, spec.data.data(), spec.data.size());
  }

  _impl->uploadedResources.push_back(std::move(ur));
}

// Store per-pass input bindings
_impl->passInputBindings.resize(_impl->passes.size());
for (const auto& pi : passInputs) {
  if (pi.passIndex < (int)_impl->passInputBindings.size()) {
    _impl->passInputBindings[pi.passIndex] = pi.bindings;
  }
}

// Create texture outputs for passes that declare { output: GPUResource.texture2D }
_impl->passTextureOutputs.resize(_impl->passes.size());
for (int passIdx : textureOutputPasses) {
  if (passIdx < 0 || passIdx >= (int)_impl->passes.size()) continue;
  wgpu::TextureDescriptor texDesc{};
  texDesc.dimension = wgpu::TextureDimension::e2D;
  texDesc.size = {(uint32_t)width, (uint32_t)height, 1};
  texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
  texDesc.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding;
  texDesc.mipLevelCount = 1;
  _impl->passTextureOutputs[passIdx] = _impl->device.CreateTexture(&texDesc);
  _impl->passes[passIdx].hasTextureOutput = true;
}
```

- [ ] **Step 3: Add helper to append custom input entries to bind groups**

Add this helper method to the class (before `processFrame`):

```cpp
void DawnComputePipeline::appendCustomInputEntries(
  int passIndex,
  std::vector<wgpu::BindGroupEntry>& entries
) {
  if (passIndex >= (int)_impl->passInputBindings.size()) return;
  const auto& bindings = _impl->passInputBindings[passIndex];

  for (const auto& ib : bindings) {
    wgpu::BindGroupEntry entry{};
    entry.binding = ib.bindingIndex;

    if (ib.type == InputBindingType::Texture3D || ib.type == InputBindingType::Texture2D) {
      if (ib.resourceHandle >= 0 && ib.resourceHandle < (int)_impl->uploadedResources.size()) {
        entry.textureView = _impl->uploadedResources[ib.resourceHandle].textureView;
      } else if (ib.sourcePass >= 0 && ib.sourcePass < (int)_impl->passTextureOutputs.size()) {
        // Texture output from a previous pass
        entry.textureView = _impl->passTextureOutputs[ib.sourcePass].CreateView();
      }
    } else if (ib.type == InputBindingType::Sampler) {
      if (ib.resourceHandle >= 0 && ib.resourceHandle < (int)_impl->uploadedResources.size()) {
        // Sampler paired with a resource texture
        entry.sampler = _impl->uploadedResources[ib.resourceHandle].sampler;
      } else if (ib.sourcePass >= 0) {
        // Sampler for a texture output from a previous pass — use default
        entry.sampler = _impl->defaultSampler;
      }
    } else if (ib.type == InputBindingType::StorageBufferRead) {
      if (ib.resourceHandle >= 0 && ib.resourceHandle < (int)_impl->uploadedResources.size()) {
        // Static resource buffer
        auto& ur = _impl->uploadedResources[ib.resourceHandle];
        entry.buffer = ur.buffer;
        entry.size = ur.buffer.GetSize();
      } else if (ib.sourceBuffer >= 0 && ib.sourceBuffer < (int)_impl->buffers.size()) {
        // Cross-pass buffer reference
        auto& sb = _impl->buffers[ib.sourceBuffer];
        entry.buffer = sb.gpuBuffer;
        entry.size = sb.byteSize;
      }
    }

    entries.push_back(entry);
  }
}
```

Add the declaration to the header too:

```cpp
void appendCustomInputEntries(int passIndex, std::vector<wgpu::BindGroupEntry>& entries);
```

- [ ] **Step 4: Update cached bind group creation in setup()**

In the existing bind group caching loop (passes 1+, around lines 215-250), after the standard entries are built, call the helper. The existing code likely uses a fixed-size array — convert to `std::vector<wgpu::BindGroupEntry>`.

Before the `CreateBindGroup` call, add:

```cpp
// Bind texture output at binding 2 (mutually exclusive with buffer output)
if (_impl->passes[i].hasTextureOutput
    && i < (int)_impl->passTextureOutputs.size()
    && _impl->passTextureOutputs[i]) {
  wgpu::BindGroupEntry texOutEntry{};
  texOutEntry.binding = 2;
  texOutEntry.textureView = _impl->passTextureOutputs[i].CreateView();
  entries.push_back(texOutEntry);
}

// Append custom input entries
appendCustomInputEntries(i, entries);
```

- [ ] **Step 5: Update per-frame bind group creation in processFrame()**

In `processFrame()`, pass 0's bind group is built per-frame (around lines 371-400). After the standard entries, add:

```cpp
// Bind texture output at binding 2 for pass 0 (if applicable)
if (_impl->passes[0].hasTextureOutput
    && !_impl->passTextureOutputs.empty()
    && _impl->passTextureOutputs[0]) {
  wgpu::BindGroupEntry texOutEntry{};
  texOutEntry.binding = 2;
  texOutEntry.textureView = _impl->passTextureOutputs[0].CreateView();
  entries.push_back(texOutEntry);
}

// Append custom input entries for pass 0
appendCustomInputEntries(0, entries);
```

Same conversion to `std::vector` if the existing code uses a fixed array.

- [ ] **Step 6: Update the C API function**

Update `dawn_pipeline_setup_multipass()` at the bottom of the file to accept and forward the new parameters:

```cpp
bool dawn_pipeline_setup_multipass(
  DawnComputePipelineRef ref,
  const char** shaders, int shaderCount,
  int width, int height,
  const int* bufferSpecsFlat, int bufferCount,
  bool useCanvas, bool sync,
  const ResourceSpec* resources, int resourceCount,
  const PassInputSpec* passInputs, int passInputCount,
  const int* textureOutputPasses, int textureOutputPassCount
) {
  // ... existing conversion code for shaders and bufferSpecs ...

  std::vector<ResourceSpec> resourceVec(resources, resources + resourceCount);
  std::vector<PassInputSpec> passInputVec(passInputs, passInputs + passInputCount);
  std::vector<int> texOutVec(textureOutputPasses, textureOutputPasses + textureOutputPassCount);

  return ref->pipeline->setup(shaderVec, width, height, bufferSpecVec,
                               useCanvas, sync, resourceVec, passInputVec, texOutVec);
}
```

- [ ] **Step 7: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.h \
  packages/react-native-webgpu-camera/modules/webgpu-camera/ios/DawnComputePipeline.mm
git commit -m "feat: implement resource upload and custom bind group entries"
```

---

## Chunk 5: Verification

### Task 10: Final verification

- [ ] **Step 1: Type check**

Run: `cd /Users/kim/dev/rn-webgpu-camera && /Users/kim/.bun/bin/bunx tsc --noEmit 2>&1 | grep -E "(GPUResource|useGPUFrameProcessor|types\.ts|WebGPUCameraModule)" | grep -v "Cannot find module"`

Expected: no new errors.

- [ ] **Step 2: Review all changes**

Run: `git diff main --stat`

Verify all files in the File Structure table are present.

- [ ] **Step 3: Build on device**

Run: `cd apps/example && eas build --platform ios --profile development --local`

Required to validate the C++/Swift changes compile.

- [ ] **Step 4: Smoke test**

After build, the existing shaders (passthrough, sobel, histogram) should work exactly as before — the custom bind group extension is additive and doesn't change the default bindings 0-2 path.
