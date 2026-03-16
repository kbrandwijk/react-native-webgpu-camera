# Apple Log + LUT Support Design

## Goal

Support native Apple Log camera input without BGRA conversion, preserving full 10-bit dynamic range through the shader pipeline. Add .cube LUT file support as a 3D texture resource for display preview tone mapping.

## Architecture

The pipeline gains two modes, selected automatically based on the active camera color space:

- **SDR** (sRGB/P3): BGRA8 input → RGBA8Unorm ping-pong. Current behavior, unchanged.
- **Apple Log**: YUV 10-bit bi-planar input → automatic internal YUV→RGB pass → RGBA16Float ping-pong.

User shaders never see YUV. They receive a single RGBA16Float texture containing Apple Log encoded RGB values (BT.2020 primaries, Apple Log transfer curve). The LUT is applied via the existing `inputs` API from custom bind groups — no new shader contract.

## Input Handling

### SDR Path (unchanged)

- `AVCaptureVideoDataOutput` requests `kCVPixelFormatType_32BGRA`
- Single-plane IOSurface imported as `BGRA8Unorm` via Dawn `SharedTextureMemory`

### Apple Log Path

- `AVCaptureVideoDataOutput` requests `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`
- IOSurface has 2 planes:
  - Plane 0 (Y): full resolution, 10-bit values in 16-bit words → import as `R16Unorm` (normalized float, not integer)
  - Plane 1 (UV): half resolution, interleaved CbCr → import as `RG16Unorm` (normalized float)
- Two separate `ImportSharedTextureMemory` calls on the same IOSurface, using the `plane` field on `wgpu::SharedTextureMemoryIOSurfaceDescriptor` (plane 0 for Y, plane 1 for UV)
- Each produces a `wgpu::Texture` — one `texture_2d<f32>` for Y, one `texture_2d<f32>` for UV
- An automatic internal compute pass runs a built-in YUV→RGB shader and writes to the first RGBA16Float ping-pong texture
- This automatic pass is invisible to the user — it runs before their first shader pass
- If the user has zero shader passes, the YUV→RGB pass still runs so the display preview works

### Mode Detection

The pipeline mode is determined by a `pixelFormat` string passed through the pipeline config:
- `"appleLog"` → Apple Log path (YUV bi-planar, RGBA16Float ping-pong)
- `"bgra"` (default) → SDR path (unchanged)

The flow: `useCamera` exposes the color space → `useGPUFrameProcessor` includes it in the native config → `setupMultiPassPipeline` in Swift reads it and forwards to the Dawn bridge → `DawnComputePipeline::setup()` receives a new `bool appleLog` parameter.

`WebGPUCameraModule.swift` also uses this to select the appropriate pixel format for `AVCaptureVideoDataOutput`:
- Apple Log → `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`
- SDR → `kCVPixelFormatType_32BGRA`

## Ping-Pong Texture Format

| Mode | Format | Precision | WGSL storage qualifier |
|------|--------|-----------|----------------------|
| SDR | RGBA8Unorm | 8-bit per channel | `texture_storage_2d<rgba8unorm, write>` |
| Apple Log | RGBA16Float | 16-bit float (~11 bits mantissa) | `texture_storage_2d<rgba16float, write>` |

Note: `rgba16float` as a writable storage texture requires Dawn feature support. Dawn on Apple GPUs (Metal backend) supports this. The setup code should request/verify this feature at device creation.

The `SkImage` output creation also handles `RGBA16Float`. Skia Graphite supports this format for display.

### Auto YUV→RGB Pass and Ping-Pong Parity

The automatic YUV→RGB pass counts as an additional pass in the pipeline. This affects the ping-pong parity logic (`finalTex = passes.size() % 2`). The implementation must use the *effective* pass count (user passes + 1 for the auto pass) when determining which texture holds the final output.

The auto pass writes to texA. User pass 0 reads texA, writes texB. Etc. The cached bind group logic for pass 1+ still works because it's based on alternating read/write textures.

## YUV→RGB Conversion Details

The built-in `yuv_to_rgb.wgsl` shader performs:

1. **Video range expansion**: `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` uses video range (Y: 64–940, CbCr: 64–960 for 10-bit). Since we import as `R16Unorm`/`RG16Unorm`, values are normalized to [0,1]. The shader expands to full range:
   - Y' = (Y - 64/1023) / (940/1023 - 64/1023)
   - Cb' = (Cb - 64/1023) / (960/1023 - 64/1023) - 0.5
   - Cr' = (Cr - 64/1023) / (960/1023 - 64/1023) - 0.5

2. **BT.2020 YCbCr→RGB matrix** (non-constant-luminance):
   - R = Y' + 1.4746 * Cr'
   - G = Y' - 0.1646 * Cb' - 0.5714 * Cr'
   - B = Y' + 1.8814 * Cb'

3. **UV upsampling**: The UV plane is half resolution. The shader uses `textureSample` with bilinear filtering on the UV texture (requires a sampler), or calculates the half-res coordinate manually.

The output is Apple Log encoded RGB in BT.2020 primaries — the log transfer curve is preserved (it's baked into the Y'CbCr values).

## .cube LUT Support

### Parser (`parseCubeFile.ts`)

- Parses standard .cube text format
- Reads `LUT_3D_SIZE N` header, then N³ lines of `R G B` float triplets
- Produces `{ data: Float32Array, size: number }` where data is N³×4 (RGBA, A=1.0)
- Suitable for upload as a 3D texture via `GPUResource.texture3D()`

### Native Resource Upload Format

The existing `DawnComputePipeline.mm` resource upload path hardcodes `RGBA8Unorm` for all textures. This must be extended:

- `GPUResource.texture3D()` gains an optional `format` field (default: `'rgba8unorm'`)
- When `format: 'rgba32float'` is specified, the native upload uses `wgpu::TextureFormat::RGBA32Float` with `bytesPerRow = width * 16` (4 floats × 4 bytes)
- The `ResourceSpec` struct gains a `format` field to carry this through the bridge
- For the .cube LUT use case: `GPUResource.texture3D(data, { width: N, height: N, depth: N, format: 'rgba32float' })`

### Loading Flow

1. User picks a .cube file from device filesystem (document picker)
2. App reads file text, calls `parseCubeFile()`
3. Creates `GPUResource.texture3D(data, { width: N, height: N, depth: N, format: 'rgba32float' })`
4. Passes as a resource to `useGPUFrameProcessor`

### Shader Usage

Uses the existing custom bind groups `inputs` API — no new abstractions:

```typescript
const lut = GPUResource.texture3D(parsedLut.data, {
  width: parsedLut.size, height: parsedLut.size, depth: parsedLut.size,
  format: 'rgba32float',
});

const { currentFrame } = useGPUFrameProcessor(camera, {
  resources: { lut },
  pipeline: (frame, { lut }) => {
    'worklet';
    frame.runShader(LUT_WGSL, { inputs: { lut } });
  },
});
```

The LUT shader samples the 3D texture using Apple Log RGB values as texture coordinates.

## Recording Consideration

Not in scope for this design, but the architecture supports it:

- **No shader recording + no onFrame**: recording grabs the original CVPixelBuffer directly (bit-perfect Apple Log YUV)
- **Shader recording**: recording taps the RGBA16Float output, converts RGB→YUV on GPU for writing

The unprocessed path is a recording optimization that applies to both SDR and Apple Log modes.

## Files Changed

| File | Change |
|------|--------|
| `WebGPUCameraModule.swift` | Detect Apple Log color space, request YUV pixel format instead of BGRA, pass `appleLog` bool to Dawn bridge via `setupMultiPassPipeline` |
| `DawnPipelineBridge.h/mm` | Forward `appleLog` bool to pipeline setup |
| `DawnComputePipeline.h/mm` | New `bool appleLog` param on `setup()`, 2-plane IOSurface import, auto YUV→RGB pass, RGBA16Float ping-pong, RGBA32Float resource upload support |
| `useCamera.ts` | Expose `colorSpace` on `CameraHandle` so `useGPUFrameProcessor` can read it |
| `WebGPUCameraModule.ts` | Add `appleLog` to `setupMultiPassPipeline` config interface |
| `useGPUFrameProcessor.ts` | Read `camera.colorSpace`, include `appleLog` in native config, pass to `buildNativeConfig` |
| `GPUResource.ts` | Add optional `format` field to `texture3D()` dims parameter |
| `types.ts` | Add `colorSpace` to `CameraHandle`, add `format` to resource dims type |

## New Files

| File | Purpose |
|------|---------|
| `parseCubeFile.ts` | .cube LUT file parser utility |
| `yuv_to_rgb.wgsl` | Built-in YUV→RGB conversion shader (BT.2020 matrix, video range) |
| `lut.wgsl` | Example LUT application shader |
| Example app additions | File picker UI for .cube files, Apple Log format selection in format picker |

## What Does NOT Change

- Shader contract (binding 0 = input, binding 1 = output, binding 2 = optional buffer)
- Custom `inputs` API for resources
- Multi-pass pipeline model
- Buffer readback
- Canvas / onFrame compositing
- SDR pipeline behavior
