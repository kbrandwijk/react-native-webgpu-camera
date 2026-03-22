// BT.2390 tone mapping — from gpu-video-shaders (Tier 2)
// The ITU standard for HDR→SDR conversion
import { createToneMapShader } from 'webgpu-video-shaders/libplacebo';

const { wgsl } = createToneMapShader({ method: 'bt2390', srcPeakNits: 1000, dstPeakNits: 203 });
export const TONEMAP_BT2390_WGSL = wgsl;
