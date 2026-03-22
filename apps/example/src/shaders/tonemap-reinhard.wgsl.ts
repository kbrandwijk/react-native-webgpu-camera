// Reinhard tone mapping — from gpu-video-shaders (Tier 2)
import { createToneMapShader } from 'webgpu-video-shaders/libplacebo';

const { wgsl } = createToneMapShader({ method: 'reinhard', srcPeakNits: 1000, dstPeakNits: 203 });
export const TONEMAP_REINHARD_WGSL = wgsl;
