// Color blindness simulation — from gpu-video-shaders
// Composed: cone distort (deuteranopia) as a Tier 1 function in a complete shader
import { createConeDistort } from 'webgpu-video-shaders/libplacebo';

const cone = createConeDistort({ type: 'deuteranopia', severity: 1.0 });

export const COLORBLIND_WGSL = /* wgsl */ `
// Color blindness simulation — deuteranopia (M-cone deficiency)

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${cone.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  let color = textureLoad(inputTex, coord, 0);
  let result = ${cone.fnName}(color);
  textureStore(outputTex, coord, result);
}
`;
