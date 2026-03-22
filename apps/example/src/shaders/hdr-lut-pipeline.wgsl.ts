// HDR LUT Pipeline — 3-pass GPU pipeline matching libplacebo architecture
//
//   Pass 1: Peak detection — passthrough + measures luminance stats
//   Pass 2: LUT generation — reads peak stats, generates/caches 256-entry LUT
//   Pass 3: LUT application — linearize HLG → IPT → LUT tone map → IPT⁻¹ → sRGB BT.709
//
// The LUT apply shader handles the FULL color pipeline in one pass:
//   HLG OETF⁻¹ + OOTF → linear BT.2020 → IPT → tone map → IPT⁻¹ → BT.709 → sRGB
// This matches libplacebo's pl_shader_color_map_ex which combines linearize,
// tone mapping, and delinearize in one dispatch.

import {
  createPeakDetectShader,
  createToneMapLutGenPipelineShader,
  createToneMapLutApplyShader,
} from 'webgpu-video-shaders/libplacebo';

const BINS = 64;
const LUT_SIZE = 256;

// Pass 1: Peak detection (passthrough + writes luminance stats)
const peakDetect = createPeakDetectShader({ histogramBins: BINS });
export const PEAK_DETECT_WGSL = peakDetect.wgsl;

// Pass 2: LUT generation (reads peak stats, generates/caches LUT)
const lutGen = createToneMapLutGenPipelineShader({
  method: 'bt2390',
  lutSize: LUT_SIZE,
  dstPeakNits: 203,
  peakDetectBins: BINS,
});
export const LUT_GEN_WGSL = lutGen.wgsl;

// Pass 3: LUT application — full color pipeline in one pass
// HLG (Apple Log) → linear BT.2020 → IPT → tone map → IPT⁻¹ → sRGB BT.709
const lutApply = createToneMapLutApplyShader({
  lutSize: LUT_SIZE,
  srcPeakNits: 1000,
  srcTransfer: 'hlg',
  dstTransfer: 'srgb',
  srcPrimaries: 'bt2020',
  dstPrimaries: 'bt709',
  srcPeakNitsOOTF: 1000,
});
export const LUT_APPLY_WGSL = lutApply.wgsl;
