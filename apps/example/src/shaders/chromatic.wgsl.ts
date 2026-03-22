// Chromatic aberration shader — from gpu-video-shaders (Tier 2)
import { createDistortShader } from 'webgpu-video-shaders/original';

const { wgsl } = createDistortShader({ method: 'chromatic-aberration', chromaticOffset: 0.003 });
export const CHROMATIC_WGSL = wgsl;
