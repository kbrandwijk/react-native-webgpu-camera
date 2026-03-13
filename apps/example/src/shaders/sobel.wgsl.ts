// Sobel edge detection compute shader.
// Reads from an input texture, writes edge-detected output to a storage texture.
// Used to validate Spike 2: compute dispatch from worklet thread.

export const SOBEL_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);

  // Sobel kernels
  // Gx:          Gy:
  // -1  0  1     -1 -2 -1
  // -2  0  2      0  0  0
  // -1  0  1      1  2  1

  var gx = vec3f(0.0);
  var gy = vec3f(0.0);

  // Sample 3x3 neighborhood
  let tl = textureLoad(inputTex, coord + vec2i(-1, -1), 0).rgb;
  let tc = textureLoad(inputTex, coord + vec2i( 0, -1), 0).rgb;
  let tr = textureLoad(inputTex, coord + vec2i( 1, -1), 0).rgb;
  let ml = textureLoad(inputTex, coord + vec2i(-1,  0), 0).rgb;
  let mr = textureLoad(inputTex, coord + vec2i( 1,  0), 0).rgb;
  let bl = textureLoad(inputTex, coord + vec2i(-1,  1), 0).rgb;
  let bc = textureLoad(inputTex, coord + vec2i( 0,  1), 0).rgb;
  let br = textureLoad(inputTex, coord + vec2i( 1,  1), 0).rgb;

  gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  let edge = sqrt(gx * gx + gy * gy);
  let intensity = clamp(length(edge) / 3.0, 0.0, 1.0);

  textureStore(outputTex, coord, vec4f(intensity, intensity, intensity, 1.0));
}
`;
