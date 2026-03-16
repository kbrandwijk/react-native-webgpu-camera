export const LUT_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
// Custom inputs: LUT 3D texture at binding 3, sampler at binding 4
@group(0) @binding(3) var lutTex: texture_3d<f32>;
@group(0) @binding(4) var lutSampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let color = textureLoad(inputTex, vec2i(id.xy), 0);

  // Use Apple Log RGB values as 3D texture coordinates
  // Clamp to [0,1] to stay within LUT bounds
  let lutCoord = clamp(color.rgb, vec3f(0.0), vec3f(1.0));
  // textureSampleLevel (not textureSample) — textureSample is unavailable in compute shaders
  let lutColor = textureSampleLevel(lutTex, lutSampler, lutCoord, 0.0);

  textureStore(outputTex, vec2i(id.xy), vec4f(lutColor.rgb, 1.0));
}
`;
