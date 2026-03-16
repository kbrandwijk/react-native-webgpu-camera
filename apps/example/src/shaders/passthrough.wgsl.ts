// Passthrough compute shader — copies input to output unchanged.
// Used to measure pipeline overhead without any processing.
// Debug: samples 5 pixels and writes RGBA to a storage buffer.

export const PASSTHROUGH_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> debug: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let color = textureLoad(inputTex, vec2i(id.xy), 0);
  textureStore(outputTex, vec2i(id.xy), color);

  // Sample 5 pixels: center, TL, TR, BL, BR — write to debug buffer
  if (id.x == dims.x / 2u && id.y == dims.y / 2u) {
    let center = textureLoad(inputTex, vec2i(vec2u(dims.x / 2u, dims.y / 2u)), 0);
    debug[0] = center.r; debug[1] = center.g; debug[2] = center.b; debug[3] = center.a;

    let tl = textureLoad(inputTex, vec2i(100, 100), 0);
    debug[4] = tl.r; debug[5] = tl.g; debug[6] = tl.b; debug[7] = tl.a;

    let tr = textureLoad(inputTex, vec2i(vec2u(dims.x - 100u, 100u)), 0);
    debug[8] = tr.r; debug[9] = tr.g; debug[10] = tr.b; debug[11] = tr.a;

    let bl = textureLoad(inputTex, vec2i(vec2u(100u, dims.y - 100u)), 0);
    debug[12] = bl.r; debug[13] = bl.g; debug[14] = bl.b; debug[15] = bl.a;

    let br = textureLoad(inputTex, vec2i(vec2u(dims.x - 100u, dims.y - 100u)), 0);
    debug[16] = br.r; debug[17] = br.g; debug[18] = br.b; debug[19] = br.a;
  }
}
`;
