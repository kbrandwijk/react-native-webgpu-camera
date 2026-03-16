// Sobel edge detection with color preservation.
// Flat areas keep original camera color; edges glow with a neon cyan highlight.
// Blend strength controls how much the original image shows through.

export const SOBEL_COLOR_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);

  // Sample 3x3 neighborhood
  let tl = textureLoad(inputTex, coord + vec2i(-1, -1), 0).rgb;
  let tc = textureLoad(inputTex, coord + vec2i( 0, -1), 0).rgb;
  let tr = textureLoad(inputTex, coord + vec2i( 1, -1), 0).rgb;
  let ml = textureLoad(inputTex, coord + vec2i(-1,  0), 0).rgb;
  let mc = textureLoad(inputTex, coord + vec2i( 0,  0), 0).rgb;
  let mr = textureLoad(inputTex, coord + vec2i( 1,  0), 0).rgb;
  let bl = textureLoad(inputTex, coord + vec2i(-1,  1), 0).rgb;
  let bc = textureLoad(inputTex, coord + vec2i( 0,  1), 0).rgb;
  let br = textureLoad(inputTex, coord + vec2i( 1,  1), 0).rgb;

  // Sobel gradients
  let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  let edge = sqrt(gx * gx + gy * gy);
  let intensity = clamp(length(edge) / 3.0, 0.0, 1.0);

  // Edge glow color (cyan-to-white based on edge strength)
  let edgeColor = mix(vec3f(0.0, 0.8, 1.0), vec3f(1.0, 1.0, 1.0), intensity);

  // Blend: original color darkened slightly + bright edge overlay
  let base = mc * 0.7;
  let result = mix(base, edgeColor, intensity);

  textureStore(outputTex, coord, vec4f(result, 1.0));
}
`;
