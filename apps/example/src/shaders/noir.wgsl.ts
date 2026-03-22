// Film noir — composed: achromatopsia → sigmoid contrast boost → vignette → grain → dither
// A dramatic black & white cinematic look
import {
  createConeDistort,
  createSigmoidize,
  createGrain,
  createDither,
} from 'webgpu-video-shaders/libplacebo';
import { createVignette } from 'webgpu-video-shaders/original';

const mono = createConeDistort({ type: 'achromatopsia' });
const sigmoid = createSigmoidize({ center: 0.5, slope: 10 });
const vig = createVignette({ strength: 0.6, innerRadius: 0.3, outerRadius: 1.0 });
const grain = createGrain({ amount: 8 });
const dith = createDither({ method: 'ordered', targetDepth: 6 });

export const NOIR_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${mono.fn}
${sigmoid.fn}
${vig.fn}
${grain.fn}
${dith.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);

  var color = textureLoad(inputTex, coord, 0);
  color = ${mono.fnName}(color);
  color = ${sigmoid.fnName}(color);
  color = ${vig.fnName}(color, coord, vec2i(dims));
  color = ${grain.fnName}(color, coord);
  color = ${dith.fnName}(color, coord);

  textureStore(outputTex, coord, color);
}
`;
