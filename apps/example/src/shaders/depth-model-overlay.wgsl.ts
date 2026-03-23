// Depth model overlay shader — blends camera feed with a colormap
// visualization of the depth estimation model output.
// The depth buffer comes from frame.runModel() (ONNX depth-anything-v2).
// Zero-copy: reads directly from ORT's output storage buffer.

export const DEPTH_MODEL_OVERLAY_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> depthBuf: array<f32>;

const DEPTH_W: u32 = 518u;
const DEPTH_H: u32 = 518u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let camera = textureLoad(inputTex, vec2i(id.xy), 0);

  // Map pixel coordinate to depth buffer index (nearest neighbor)
  let uv = vec2f(id.xy) / vec2f(dims);
  let dx = u32(uv.x * f32(DEPTH_W));
  let dy = u32(uv.y * f32(DEPTH_H));
  let depthIdx = clamp(dy, 0u, DEPTH_H - 1u) * DEPTH_W + clamp(dx, 0u, DEPTH_W - 1u);
  let raw = depthBuf[depthIdx];

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
