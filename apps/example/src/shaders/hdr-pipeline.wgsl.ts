// Composed HDR pipeline — deband + tone map + grain in a single compute dispatch.
// Uses gpu-video-shaders Tier 1 composable functions.

import { createDeband, createToneMap, createGrain } from 'webgpu-video-shaders/libplacebo';

const deband = createDeband({ iterations: 1, threshold: 3, radius: 16 });
const tonemap = createToneMap({ method: 'hable', srcPeakNits: 1000, dstPeakNits: 203 });
const grain = createGrain({ amount: 4 });

export const HDR_PIPELINE_WGSL = /* wgsl */ `
// Composed HDR pipeline — deband → tone map → grain
// Ported from libplacebo (LGPL-2.1+)

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${deband.fn}
${tonemap.fn}
${grain.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);

  var color = ${deband.fnName}(inputTex, coord, vec2i(dims));
  color = ${tonemap.fnName}(color);
  color = ${grain.fnName}(color, coord);

  textureStore(outputTex, coord, color);
}
`;
