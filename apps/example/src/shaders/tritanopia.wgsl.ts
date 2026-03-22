// Tritanopia (blue-blind) simulation — from gpu-video-shaders
import { createConeDistort } from 'webgpu-video-shaders/libplacebo';

const cone = createConeDistort({ type: 'tritanopia', severity: 1.0 });

export const TRITANOPIA_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${cone.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let color = textureLoad(inputTex, vec2i(id.xy), 0);
  textureStore(outputTex, vec2i(id.xy), ${cone.fnName}(color));
}
`;
