// Gaussian blur shader — from gpu-video-shaders (Tier 2)
import { createGaussianBlurShader } from 'webgpu-video-shaders/original';

const { wgsl } = createGaussianBlurShader({ radius: 4, sigma: 2 });
export const BLUR_WGSL = wgsl;
