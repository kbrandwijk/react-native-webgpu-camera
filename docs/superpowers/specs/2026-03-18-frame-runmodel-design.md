# frame.runModel() — GPU-Native ML Inference in the Camera Pipeline

## Goal

Run ONNX models on camera frames inside `useGPUFrameProcessor`, using the same Dawn GPU device as compute shaders. Zero CPU pixel copies.

```typescript
const { currentFrame } = useGPUFrameProcessor(camera, {
  resources: {
    depth: GPUResource.model('onnx-community/depth-anything-v2-small'),
  },
  pipeline: (frame, { depth }) => {
    'worklet';
    const depthMap = frame.runModel(depth);  // returns texture handle
    frame.runShader(DEPTH_OVERLAY_WGSL, { inputs: { depth: depthMap } });
  },
});
```

## Architecture

### Pipeline Integration

`frame.runModel()` is a pipeline pass, just like `frame.runShader()`. It:
- Reads from the current ping-pong texture (output of previous pass)
- Runs ONNX inference on the GPU (WebGPU EP, same Dawn device)
- Writes output to a texture or buffer available to subsequent passes

### Data Flow (Zero-Copy)

```
Camera IOSurface
  → Pass 0: YUV→RGB (compute shader) → texA (RGBA16Float, 1080x1920)
  → Pass 1: frame.runModel(depth)
      → Resize texA to model input (518x518) via compute shader
      → Bind resized texture as ONNX input tensor (GPU buffer)
      → ONNX Runtime inference (WebGPU EP, same Dawn device)
      → Output tensor → GPU buffer → texture view
      → Write depth texture to texB
  → Pass 2: frame.runShader(OVERLAY_WGSL, { inputs: { depth: depthMap } })
      → Blend camera + depth colormap → texA
  → Skia Canvas display
```

### Key Design Decisions

**1. Model Resource Lifecycle**

Models are declared in `resources`, like shaders and textures:
```typescript
resources: {
  depth: GPUResource.model(modelPathOrUrl, {
    inputShape: [1, 3, 518, 518],  // optional, inferred from model
    executionProvider: 'webgpu',     // default: 'webgpu', fallback: 'cpu'
  }),
}
```

The native side:
- Downloads the model (if URL) and caches it
- Creates an `InferenceSession` with WebGPU EP on the shared Dawn device
- Pre-allocates input/output GPU buffers
- Creates resize compute shader (camera resolution → model input size)

This happens once at setup, not per-frame.

**2. Input Preprocessing (GPU)**

The camera frame (1080x1920 RGBA16Float) needs to be resized and normalized for the model (e.g., 518x518 float32, ImageNet normalization). This is a compute shader:

```wgsl
// Auto-generated resize + normalize shader
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = vec2f(id.xy) / vec2f(modelW, modelH);
  let pixel = textureSampleLevel(cameraTex, sampler, uv, 0.0);

  // ImageNet normalization (or model-specific)
  let r = (pixel.r - 0.485) / 0.229;
  let g = (pixel.g - 0.456) / 0.224;
  let b = (pixel.b - 0.406) / 0.225;

  // Write to GPU buffer in NCHW layout
  let idx = id.y * modelW + id.x;
  outputBuffer[idx] = r;                          // channel 0
  outputBuffer[modelW * modelH + idx] = g;        // channel 1
  outputBuffer[2 * modelW * modelH + idx] = b;    // channel 2
}
```

This runs as a compute dispatch before the ONNX session.run().

**3. ONNX Runtime IO Binding**

The WebGPU EP supports IO binding — binding GPU buffers directly as input/output tensors:

```cpp
// C++ native side
Ort::IoBinding binding(session);

// Input: GPU buffer from resize compute shader
Ort::MemoryInfo gpuMemInfo("WebGPU_Buffer", OrtDeviceAllocator, 0, OrtMemTypeDefault);
Ort::Value inputTensor = Ort::Value::CreateTensor(
    gpuMemInfo, gpuInputBuffer, inputSize, inputShape, inputShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
binding.BindInput("input", inputTensor);

// Output: pre-allocated GPU buffer
binding.BindOutput("output", gpuMemInfo);

// Run — all GPU, no CPU data movement
session.Run(Ort::RunOptions{nullptr}, binding);

// Output is now in a GPU buffer, create texture view for next shader pass
```

**4. Output as Pipeline Texture**

The model output (e.g., 518x518 depth map) becomes available as a texture that subsequent shader passes can bind:

```typescript
const depthMap = frame.runModel(depth);  // returns ResourceHandle<'texture2d'>
frame.runShader(OVERLAY_WGSL, { inputs: { depth: depthMap } });
```

On the native side, the output GPU buffer is wrapped as a texture view. If the output is a different size than the camera frame, the shader handles upsampling (same as our LiDAR depth shader).

**5. Async by Default**

Most models can't run at camera FPS. `runModel` is **async by default** — inference runs on a background thread and the pipeline always gets the latest available result. The camera pipeline never blocks.

```typescript
resources: {
  depth: GPUResource.model(url),  // async by default
},
pipeline: (frame, { depth }) => {
  'worklet';
  const depthMap = frame.runModel(depth);  // latest result, may be stale or null
  if (depthMap) {
    frame.runShader(OVERLAY_WGSL, { inputs: { depth: depthMap } });
  }
},
```

This runs at "model FPS" (e.g., 15fps for a larger model) while the camera pipeline runs at full frame rate (60fps). The depth overlay updates at model speed but the camera stays smooth.

For small models that can keep up, `sync: true` makes it blocking:
```typescript
resources: {
  classifier: GPUResource.model(url, { sync: true }),  // blocks pipeline until done
},
```

## Implementation Steps

### Phase 1: GPU IO Binding + Async
1. Add `GPUResource.model(path)` resource type
2. Native: load model, create `InferenceSession` with WebGPU EP on shared Dawn device
3. Native: compile resize + normalize compute shader (camera res → model input)
4. Native: use ONNX Runtime IO binding to bind GPU buffers directly as input/output
5. Native: run inference on background thread, cache latest output texture
6. Add `frame.runModel(handle)` — returns latest cached output texture (null if not ready yet)
7. Output usable as input to subsequent `frame.runShader()` passes

### Phase 2: Polish
8. Model download + caching (URL support in `GPUResource.model()`)
9. Configurable normalization params per model
10. `sync: true` option for small models
11. Multiple models in one pipeline

## Native API Surface

### C++ (DawnComputePipeline)
```cpp
// Setup: called once when resources are parsed
int setupModel(const std::string& modelPath,
               const std::vector<int64_t>& inputShape,
               bool useWebGPU);

// Per-frame: called during processFrame when runModel is encountered
bool runModel(int modelIndex,
              wgpu::Texture inputTexture,
              int inputWidth, int inputHeight);

// Access output texture for subsequent passes
wgpu::TextureView getModelOutputView(int modelIndex);
```

### JS Bridge (useGPUFrameProcessor)
```typescript
// In capture proxy
runModel(handle: ResourceHandle<'model'>): ResourceHandle<'texture2d'>;

// Generates native instructions:
// 1. Resize current ping-pong texture → model input buffer (compute)
// 2. session.Run() with IO binding (WebGPU EP)
// 3. Output buffer → texture view
// 4. Return handle for use in subsequent runShader inputs
```

## Open Questions

1. **Model format**: ONNX only, or also CoreML/TFLite? ONNX is the most portable and has WebGPU EP.
2. **Input preprocessing**: Should normalization params be inferred from the model, or user-specified? Different models use different normalization.
3. **Output format**: Depth models output float32 1-channel. Classification outputs float32 1000-class. Object detection outputs bounding boxes. The API needs to handle different output shapes.
4. **Multiple models**: Can you run multiple models in one pipeline? (e.g., depth + segmentation)
5. **Model download**: Should `GPUResource.model(url)` handle download + caching, or require a local path?
