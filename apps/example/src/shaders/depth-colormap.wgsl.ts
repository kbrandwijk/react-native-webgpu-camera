// Depth colormap visualization shader — blends camera color with a
// depth-based colormap (blue=near, green=mid, yellow=far).
// Depth texture (320x240 R16Float, meters) is bilinearly upsampled to output resolution.

export const DEPTH_COLORMAP_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;

fn depthColormap(depth: f32) -> vec3f {
  // Map 0-5m range: blue (0m) -> green (2.5m) -> yellow (5m+)
  let t = clamp(depth / 5.0, 0.0, 1.0);
  let blue  = vec3f(0.0, 0.0, 1.0);
  let green = vec3f(0.0, 1.0, 0.0);
  let yellow = vec3f(1.0, 1.0, 0.0);
  if (t < 0.5) {
    return mix(blue, green, t * 2.0);
  }
  return mix(green, yellow, (t - 0.5) * 2.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let outDims = textureDimensions(outputTex);
  if (id.x >= outDims.x || id.y >= outDims.y) {
    return;
  }

  // Load camera color
  let color = textureLoad(inputTex, vec2i(id.xy), 0);

  // Sample depth with bilinear interpolation
  let uv = (vec2f(id.xy) + 0.5) / vec2f(outDims);
  let depthSample = textureSampleLevel(depthTex, depthSampler, uv, 0.0);
  let depth = depthSample.r;

  // Map depth to colormap
  let mapped = depthColormap(depth);

  // Blend 60% camera + 40% depth colormap
  let blended = vec4f(
    color.rgb * 0.6 + mapped * 0.4,
    color.a
  );

  textureStore(outputTex, vec2i(id.xy), blended);
}
`;
