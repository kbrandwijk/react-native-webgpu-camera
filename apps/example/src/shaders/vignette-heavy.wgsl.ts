// Heavy vignette effect — from gpu-video-shaders
import { createVignette } from 'webgpu-video-shaders/original';

const vig = createVignette({ strength: 0.7, innerRadius: 0.3, outerRadius: 0.9 });

export const VIGNETTE_HEAVY_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${vig.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  let color = textureLoad(inputTex, coord, 0);
  textureStore(outputTex, coord, ${vig.fnName}(color, coord, vec2i(dims)));
}
`;
