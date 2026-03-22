// Cinematic pipeline — composed: deband → tonemap (hable) → vignette → grain → dither
// Uses gpu-video-shaders Tier 1 composable functions in a single dispatch.
import { createDeband, createToneMap, createGrain, createDither } from 'webgpu-video-shaders/libplacebo';
import { createVignette } from 'webgpu-video-shaders/original';

const deband = createDeband({ iterations: 1, threshold: 2 });
const tonemap = createToneMap({ method: 'hable', srcPeakNits: 1000, dstPeakNits: 203 });
const vignette = createVignette({ strength: 0.35, innerRadius: 0.5, outerRadius: 1.2 });
const grain = createGrain({ amount: 3 });
const dither = createDither({ method: 'ordered', targetDepth: 8 });

export const CINEMA_WGSL = /* wgsl */ `
// Cinematic pipeline — deband → tonemap → vignette → grain → dither

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${deband.fn}
${tonemap.fn}
${vignette.fn}
${grain.fn}
${dither.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);

  var color = ${deband.fnName}(inputTex, coord, vec2i(dims));
  color = ${tonemap.fnName}(color);
  color = ${vignette.fnName}(color, coord, vec2i(dims));
  color = ${grain.fnName}(color, coord);
  color = ${dither.fnName}(color, coord);

  textureStore(outputTex, coord, color);
}
`;
