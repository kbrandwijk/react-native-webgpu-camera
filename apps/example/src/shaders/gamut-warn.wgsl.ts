// Gamut warning — converts P3 → BT.709 and highlights out-of-gamut pixels in magenta.
// Use with P3 D65 or Apple Log format for visible results.
// On sRGB feeds, shows original image unchanged (everything is in gamut).
import { createColorMap } from 'webgpu-video-shaders/libplacebo';

const p3ToBt709 = createColorMap({
  srcColorSpace: 'display-p3',
  dstColorSpace: 'bt709',
  srcTransfer: 'linear',
  dstTransfer: 'linear',
});

export const GAMUT_WARN_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

${p3ToBt709.fn}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let color = textureLoad(inputTex, vec2i(id.xy), 0);
  let bt709 = ${p3ToBt709.fnName}(color);

  let minC = min(min(bt709.r, bt709.g), bt709.b);
  let maxC = max(max(bt709.r, bt709.g), bt709.b);

  if (minC < -0.001 || maxC > 1.001) {
    textureStore(outputTex, vec2i(id.xy), vec4f(1.0, 0.0, 1.0, color.a));
  } else {
    textureStore(outputTex, vec2i(id.xy), color);
  }
}
`;
