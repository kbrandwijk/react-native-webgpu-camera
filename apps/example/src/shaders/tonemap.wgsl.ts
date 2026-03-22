// Standalone tone map shader — from gpu-video-shaders (Tier 2)
import { createToneMapShader } from 'webgpu-video-shaders/libplacebo';

const { wgsl } = createToneMapShader({ method: 'hable', srcPeakNits: 1000, dstPeakNits: 203 });
export const TONEMAP_WGSL = wgsl;
