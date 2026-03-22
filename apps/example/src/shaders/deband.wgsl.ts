// Standalone deband shader — from gpu-video-shaders (Tier 2)
import { createDebandShader } from 'webgpu-video-shaders/libplacebo';

const { wgsl } = createDebandShader({ iterations: 2, threshold: 4, radius: 16, grain: 6 });
export const DEBAND_WGSL = wgsl;
