// Depth model overlay shader — blends camera feed with a colormap
// visualization of the depth estimation model output.
// The depth texture comes from frame.runModel() (ONNX depth-anything-v2).

export const DEPTH_MODEL_OVERLAY_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex: texture_2d<f32>;
@group(0) @binding(4) var depthSampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let camera = textureLoad(inputTex, vec2i(id.xy), 0);
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let raw = textureSampleLevel(depthTex, depthSampler, uv, 0.0).r;

  // Normalize raw disparity to 0-1 using a saturate — depth-anything outputs
  // large values so we use an adaptive scale: divide by a reasonable max.
  // A fixed divisor works because the model's output range is fairly stable.
  // Tweak this value if the colormap looks wrong.
  let d = saturate(raw / 50.0);

  // Blue (near) -> Green (mid) -> Yellow (far) colormap
  var color: vec3f;
  if (d < 0.5) {
    color = mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), d * 2.0);
  } else {
    color = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), (d - 0.5) * 2.0);
  }

  let blended = mix(camera.rgb, color, 0.4);
  textureStore(outputTex, vec2i(id.xy), vec4f(blended, 1.0));
}
`;
