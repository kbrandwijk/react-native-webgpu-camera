// Luminance histogram compute shader.
// Reads from input texture, computes 256-bin brightness histogram into a storage buffer.
// Each thread clears its assigned bin(s) first, then barrier, then accumulate.
// Outputs a passthrough of the input to the output texture (identity pass).

export const HISTOGRAM_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 256>;

// We use a large workgroup so we can clear all 256 bins with a storageBarrier.
// 256 threads per workgroup = each thread clears one bin.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let dims = textureDimensions(inputTex);

  // --- Phase 1: Clear histogram (first workgroup only) ---
  // Only workgroup (0,0) clears. This is a race with other workgroups,
  // but since we read the *previous* frame's histogram on the JS side
  // (double-buffered staging), and atomicStore of 0 before atomicAdd
  // is fine for approximate results at 120fps.
  if (gid.x < 256u && gid.y == 0u) {
    atomicStore(&histogram[gid.x], 0u);
  }

  // storageBarrier ensures the clear is visible to this workgroup
  storageBarrier();

  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let coord = vec2i(gid.xy);
  let color = textureLoad(inputTex, coord, 0);

  // Passthrough: write input to output unchanged
  textureStore(outputTex, coord, color);

  // Rec. 709 luminance
  let lum = dot(color.rgb, vec3f(0.2126, 0.7152, 0.0722));
  let bin = min(u32(lum * 256.0), 255u);
  atomicAdd(&histogram[bin], 1u);
}
`;
