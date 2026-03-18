// Depth colormap visualization shader — blends camera color with a
// depth-based colormap (blue=near, green=mid, yellow=far).
// Depth texture (320x180 R16Float, meters) is bilinearly upsampled to output resolution.

export const DEPTH_COLORMAP_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;

fn depthColormap(t: f32) -> vec3f {
  // Blue (near) -> Green (mid) -> Yellow (far)
  let r = clamp(2.0 * t - 0.5, 0.0, 1.0);
  let g = clamp(1.0 - 2.0 * abs(t - 0.5), 0.0, 1.0) + clamp(2.0 * t - 1.0, 0.0, 1.0);
  let b = clamp(1.0 - 2.0 * t, 0.0, 1.0);
  return vec3f(r, g, b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) { return; }

  // Camera color
  let color = textureLoad(inputTex, vec2i(id.xy), 0).rgb;

  // Sample depth — rotate 90° CW to match portrait output (depth texture is landscape)
  let depthDims = vec2f(textureDimensions(depthTex));
  let rotU = (f32(id.y) + 0.5) / f32(outDims.y);
  let rotV = 1.0 - (f32(id.x) + 0.5) / f32(outDims.x);
  let depth = textureSampleLevel(depthTex, depthSampler, vec2f(rotU, rotV), 0.0).r;

  // Normalize: 0-3m range mapped to 0-1
  let t = clamp(depth / 3.0, 0.0, 1.0);
  let mapColor = depthColormap(t);

  // Blend: 40% camera + 60% colormap
  let blended = mix(color, mapColor, 0.6);

  textureStore(outputTex, vec2i(id.xy), vec4f(blended, 1.0));
}
`;
