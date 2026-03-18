// Depth colormap visualization shader — blends camera color with a
// depth-based colormap (blue=near, green=mid, yellow=far).
// Depth texture (320x180 R16Float) is bilinearly upsampled to output resolution.
// Note: LiDAR delivers disparity (1/meters), not depth — invert for distance.

export const DEPTH_COLORMAP_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) { return; }

  // DEBUG: just passthrough camera — no depth at all
  let color = textureLoad(inputTex, vec2i(id.xy), 0);
  textureStore(outputTex, vec2i(id.xy), vec4f(color.rgb, 1.0));
}
`;
