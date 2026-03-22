// Barrel distortion — from gpu-video-shaders
import { createDistort } from 'webgpu-video-shaders/original';

const dist = createDistort({ method: 'barrel', strength: 0.5 });

export const BARREL_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${dist.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  textureStore(outputTex, coord, ${dist.fnName}(inputTex, coord, vec2i(dims)));
}
`;
