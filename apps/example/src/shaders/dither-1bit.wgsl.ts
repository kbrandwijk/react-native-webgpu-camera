// 1-bit ordered dithering — from gpu-video-shaders
// Creates a dramatic black & white posterized effect
import { createDither } from 'webgpu-video-shaders/libplacebo';

const dith = createDither({ method: 'ordered', targetDepth: 1 });

export const DITHER_1BIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${dith.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let color = textureLoad(inputTex, vec2i(id.xy), 0);
  textureStore(outputTex, vec2i(id.xy), ${dith.fnName}(color, vec2i(id.xy)));
}
`;
