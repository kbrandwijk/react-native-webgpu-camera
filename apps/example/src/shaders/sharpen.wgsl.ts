// Adaptive sharpening shader — from gpu-video-shaders (Tier 2)
import { createSharpenShader } from 'webgpu-video-shaders/original';

const { wgsl } = createSharpenShader({ method: 'cas', strength: 0.6 });
export const SHARPEN_WGSL = wgsl;
