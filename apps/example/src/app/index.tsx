import { useState, useEffect, useMemo, useRef } from 'react';
import { useRouter } from 'expo-router';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
  Modal,
  FlatList,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage, Skia, Picture, createPicture, WebGPUCanvas } from '@shopify/react-native-skia';
import type { WebGPUCanvasRef } from '@shopify/react-native-skia';
import { useDerivedValue } from 'react-native-reanimated';
import { Asset } from 'expo-asset';
import { useCamera, useGPUFrameProcessor, useCameraFormats, GPUResource } from 'react-native-webgpu-camera';
import type { CameraFormat, ColorSpace } from 'react-native-webgpu-camera';
import { PASSTHROUGH_WGSL } from '@/shaders/passthrough.wgsl';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import { SOBEL_COLOR_WGSL } from '@/shaders/sobel-color.wgsl';
import { HISTOGRAM_WGSL } from '@/shaders/histogram.wgsl';
import { LUT_WGSL } from '@/shaders/lut.wgsl';
import { DEPTH_COLORMAP_WGSL } from '@/shaders/depth-colormap.wgsl';
import { DEPTH_MODEL_OVERLAY_WGSL } from '@/shaders/depth-model-overlay.wgsl';
import { DEBAND_WGSL } from '@/shaders/deband.wgsl';
import { TONEMAP_WGSL } from '@/shaders/tonemap.wgsl';
import { HDR_PIPELINE_WGSL } from '@/shaders/hdr-pipeline.wgsl';
import { SHARPEN_WGSL } from '@/shaders/sharpen.wgsl';
import { BLUR_WGSL } from '@/shaders/blur.wgsl';
import { CHROMATIC_WGSL } from '@/shaders/chromatic.wgsl';
import { COLORBLIND_WGSL } from '@/shaders/colorblind.wgsl';
import { CINEMA_WGSL } from '@/shaders/cinema.wgsl';
import { PEAK_DETECT_WGSL as HDR_PEAK_WGSL, LUT_GEN_WGSL, LUT_APPLY_WGSL } from '@/shaders/hdr-lut-pipeline.wgsl';
import { PROTANOPIA_WGSL } from '@/shaders/protanopia.wgsl';
import { TRITANOPIA_WGSL } from '@/shaders/tritanopia.wgsl';
import { MONOCHROME_WGSL } from '@/shaders/monochrome.wgsl';
import { DITHER_1BIT_WGSL } from '@/shaders/dither-1bit.wgsl';
import { DITHER_3BIT_WGSL } from '@/shaders/dither-3bit.wgsl';
import { TONEMAP_REINHARD_WGSL } from '@/shaders/tonemap-reinhard.wgsl';
import { TONEMAP_BT2390_WGSL } from '@/shaders/tonemap-bt2390.wgsl';
import { GAMUT_WARN_WGSL } from '@/shaders/gamut-warn.wgsl';
import { VIGNETTE_HEAVY_WGSL } from '@/shaders/vignette-heavy.wgsl';
import { BARREL_WGSL } from '@/shaders/barrel.wgsl';
import { NOIR_WGSL } from '@/shaders/noir.wgsl';
import DepthEstimation from '@/components/DepthEstimation';
import OrtTest from '@/components/OrtTest';
import { Paths, File as ExpoFile } from 'expo-file-system';

type ShaderMode =
  | { name: string; wgsl: readonly string[]; type: 'simple' }
  | { name: string; type: 'histogram' }
  | { name: string; type: 'histogram-onframe' }
  | { name: string; type: 'applelog' }
  | { name: string; type: 'depth' }
  | { name: string; type: 'depth-model' }
  | { name: string; type: 'hdr-lut' };

const DEPTH_MODEL_URL = 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx';
const DEPTH_MODEL_PATH = `${Paths.document.uri}/depth-anything-v2-small.onnx`;

const SHADERS: ShaderMode[] = [
  { name: 'Depth Model', type: 'depth-model' },
  { name: 'Depth', type: 'depth' },
    { name: 'None', wgsl: [], type: 'simple' },
  { name: 'Passthrough', wgsl: [PASSTHROUGH_WGSL], type: 'simple' },
  { name: 'Sobel', wgsl: [SOBEL_WGSL], type: 'simple' },
  { name: 'Sobel Color', wgsl: [SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Multi-pass', wgsl: [SOBEL_WGSL, SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Histogram', type: 'histogram' },
  { name: 'Hist (burn)', type: 'histogram-onframe' },
  { name: 'LUT', type: 'applelog' },
  { name: 'Deband', wgsl: [DEBAND_WGSL], type: 'simple' },
  { name: 'Tonemap', wgsl: [TONEMAP_WGSL], type: 'simple' },
  { name: 'HDR Pipeline', wgsl: [HDR_PIPELINE_WGSL], type: 'simple' },
  { name: 'Sharpen', wgsl: [SHARPEN_WGSL], type: 'simple' },
  { name: 'Blur', wgsl: [BLUR_WGSL], type: 'simple' },
  { name: 'Chromatic', wgsl: [CHROMATIC_WGSL], type: 'simple' },
  { name: 'Colorblind', wgsl: [COLORBLIND_WGSL], type: 'simple' },
  { name: 'Cinema', wgsl: [CINEMA_WGSL], type: 'simple' },
  { name: 'Noir', wgsl: [NOIR_WGSL], type: 'simple' },
  { name: 'HDR LUT', type: 'hdr-lut' },
  { name: 'TM Reinhard', wgsl: [TONEMAP_REINHARD_WGSL], type: 'simple' },
  { name: 'TM BT.2390', wgsl: [TONEMAP_BT2390_WGSL], type: 'simple' },
  { name: 'Gamut Warn', wgsl: [GAMUT_WARN_WGSL], type: 'simple' },
  { name: 'Protanopia', wgsl: [PROTANOPIA_WGSL], type: 'simple' },
  { name: 'Tritanopia', wgsl: [TRITANOPIA_WGSL], type: 'simple' },
  { name: 'Monochrome', wgsl: [MONOCHROME_WGSL], type: 'simple' },
  { name: 'Dither 1-bit', wgsl: [DITHER_1BIT_WGSL], type: 'simple' },
  { name: 'Dither 3-bit', wgsl: [DITHER_3BIT_WGSL], type: 'simple' },
  { name: 'Vignette', wgsl: [VIGNETTE_HEAVY_WGSL], type: 'simple' },
  { name: 'Barrel', wgsl: [BARREL_WGSL], type: 'simple' },
];

function CameraPreview({ shaderChain, format, colorSpace }: { shaderChain: readonly string[]; format?: CameraFormat; colorSpace?: ColorSpace }) {
  const canvasRef = useRef<WebGPUCanvasRef>(null);

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  const { fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    canvasRef,
    pipeline: (frame) => {
      'worklet';
      for (const wgsl of shaderChain) {
        // Only request debug buffer for passthrough shader (has @binding(2) debug array)
        if (wgsl === shaderChain[0] && shaderChain.length === 1 && wgsl.includes('debug')) {
          frame.runShader(wgsl, { output: Float32Array, count: 20 });
        } else {
          frame.runShader(wgsl);
        }
      }
    },
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[FPS] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms bind=${m.bindGroup.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms buf=${m.buffers.toFixed(2)}ms skImg=${m.makeImage.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <View style={{ flex: 1, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center' }}>
        <WebGPUCanvas ref={canvasRef} style={{
          width: '100%',
          aspectRatio: camera.height / camera.width, // portrait: w=cam.h, h=cam.w
          borderWidth: 1,
          borderColor: 'white',
        }} />
      </View>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramPreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  // Histogram with buffer readback (sync required until async readback is implemented)
  const { currentFrame, buffers, error } = useGPUFrameProcessor(camera, {
    sync: false,
    pipeline: (frame) => {
      'worklet';
      const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
      return { hist };
    },
  });

  const emptyPicture = createPicture(() => {});

  // Texture is now portrait (rotation done in GPU) — dimensions match directly
  const texW = camera.height;  // portrait width (was landscape height)
  const texH = camera.width;   // portrait height (was landscape width)

  // Scale factor: screen points → texture pixels
  const overlayScale = screenW / texW;

  // Create a Skia Picture from buffer data — draws in texture coordinates
  const histPicture = useDerivedValue(() => {
    const hist = buffers.value.hist as Uint32Array | null;
    // console.log('[Histogram oveblue is rlay] buffer data:', hist ? `${hist.length} elements, first=${hist[0]}` : 'null');
    if (!hist) return emptyPicture;

    // Same coordinates as onFrame burn-in
    const histW = 1200;
    const histH = 500;
    const x0 = texW - histW - 60;
    const y0 = texH - histH - 120;

    return createPicture((canvas) => {
      // Background
      const bgPaint = Skia.Paint();
      bgPaint.setColor(Skia.Color('black'));
      bgPaint.setAlphaf(0.7);
      canvas.drawRect(Skia.XYWHRect(x0 - 8, y0 - 8, histW + 16, histH + 16), bgPaint);

      // Find max for normalization
      let maxVal = 1;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > maxVal) maxVal = hist[i];
      }

      // Bars
      const barPaint = Skia.Paint();
      barPaint.setColor(Skia.Color('white'));
      const barW = histW / 256;
      for (let i = 0; i < 256; i++) {
        const barH = (hist[i] / maxVal) * histH;
        if (barH < 1) continue;
        canvas.drawRect(
          Skia.XYWHRect(x0 + i * barW, y0 + histH - barH, Math.max(barW, 1), barH),
          barPaint,
        );
      }
    });
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
        {/* Histogram overlay — drawn in texture coords, scaled to screen */}
        <Group transform={[{ scale: overlayScale }]}>
          <Picture picture={histPicture} />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Histogram ${camera.width}x${camera.height}` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramOnFramePreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, {
    sync: false,
    pipeline: (frame) => {
      'worklet';
      const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
      return { hist };
    },
    onFrame: (() => {
      const bgPaint = Skia.Paint();
      bgPaint.setColor(Skia.Color('black'));
      bgPaint.setAlphaf(0.6);
      const barPaint = Skia.Paint();
      barPaint.setColor(Skia.Color('white'));
      // Cache normalized bar heights so we can redraw even when hist is null
      const barHeights = new Float32Array(256);
      let hasData = false;

      return (frame: any, { hist }: any) => {
        'worklet';
        // Update cached heights when new data arrives
        if (hist) {
          let maxVal = 1;
          for (let i = 0; i < 256; i++) {
            if (hist[i] > maxVal) maxVal = hist[i];
          }
          for (let i = 0; i < 256; i++) {
            barHeights[i] = hist[i] / maxVal;
          }
          hasData = true;
        }
        if (!hasData) return;

        // Always draw — compute overwrites finalTex every frame
        const histW = 1200;
        const histH = 500;
        const x0 = (frame.width - histW) / 2;
        const y0 = (frame.height - histH) / 2;

        frame.canvas.drawRect(Skia.XYWHRect(x0 - 8, y0 - 8, histW + 16, histH + 16), bgPaint);

        const barW = histW / 256;
        for (let i = 0; i < 256; i++) {
          const barH = barHeights[i] * histH;
          if (barH < 1) continue;
          frame.canvas.drawRect(
            Skia.XYWHRect(x0 + i * barW, y0 + histH - barH, Math.max(barW, 1), barH),
            barPaint,
          );
        }
      };
    })(),
  });


  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Hist-burn ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function AppleLogPreview({ format, colorSpace, lutResource }: { format?: CameraFormat; colorSpace?: ColorSpace; lutResource: ReturnType<typeof GPUResource.texture3D> | null }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  const resources = lutResource ? { lut: lutResource } : undefined;

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    resources,
    pipeline: (frame, res: any) => {
      'worklet';
      if (res?.lut) {
        frame.runShader(LUT_WGSL, { inputs: { lut: res.lut } });
      }
    },
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[AppleLog] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms bind=${m.bindGroup.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms buf=${m.buffers.toFixed(2)}ms skImg=${m.makeImage.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Apple Log ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function DepthPreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
    useDepth: true,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, {
    resources: {
      depth: GPUResource.cameraDepth(),
    },
    pipeline: (frame, res: any) => {
      'worklet';
      if (res?.depth) {
        frame.runShader(DEPTH_COLORMAP_WGSL, { inputs: { depth: res.depth } });
      }
    },
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Depth ${camera.width}x${camera.height}` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function DepthModelPreview({ format, colorSpace, modelPath }: { format?: CameraFormat; colorSpace?: ColorSpace; modelPath: string }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  // Memoize resources — GPUResource.model() creates a new object each call,
  // which would trigger useEffect re-runs and recreate the pipeline every render.
  const depthResource = useMemo(
    () => GPUResource.model(modelPath, { inputShape: [1, 3, 518, 518] }),
    [modelPath],
  );

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    resources: { depth: depthResource },
    pipeline: (frame, res: any) => {
      'worklet';
      const depthMap = frame.runModel(res.depth);
      if (depthMap) {
        frame.runShader(DEPTH_MODEL_OVERLAY_WGSL, { inputs: { depth: depthMap } });
      }
    },
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[DepthModel] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `Depth Model ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HdrLutPreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  // 3-pass GPU pipeline (matching libplacebo architecture):
  //   Pass 1: Peak detection — measures scene luminance stats
  //   Pass 2: LUT generation — reads peak stats, generates/caches tone map LUT
  //   Pass 3: LUT application — reads LUT, applies tone mapping per-pixel in IPT space
  const { currentFrame, buffers, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    sync: false,
    pipeline: (frame) => {
      'worklet';
      // Pass 1: peak detection (passthrough + measures luminance)
      const peakStats = frame.runShader(HDR_PEAK_WGSL, { output: Uint32Array, count: 66 });
      // Pass 2: LUT gen (reads peak stats, generates/caches LUT)
      const lutBuffer = frame.runShader(LUT_GEN_WGSL, {
        output: Float32Array,
        count: 257,
        inputs: { peakData: peakStats },
      });
      // Pass 3: LUT apply (RGB→IPT→tone map→IPT⁻¹→RGB, all in one pass)
      frame.runShader(LUT_APPLY_WGSL, { inputs: { lut: lutBuffer } });
      return { peakStats, lutBuffer };
    },
  });

  const lastPeakRef = { current: 0 };
  useDerivedValue(() => {
    const m = metrics.value;
    const peak = buffers.value.peakStats as Uint32Array | null;
    const lut = buffers.value.lutBuffer as Float32Array | null;
    if (fps.value > 0 && m) {
      const peakMax = peak ? peak[0] : 0;
      const peakPQ = peakMax / 65535;
      const cachedPeak = lut ? lut[256] : 0;
      const regenerated = Math.abs(cachedPeak - lastPeakRef.current) > 0.0001;
      lastPeakRef.current = cachedPeak;
      console.log(`[HDR LUT] ${fps.value}fps | peak=${peakPQ.toFixed(4)} PQ (${peakMax}) cached=${cachedPeak.toFixed(4)} ${regenerated ? 'REGEN' : 'cached'} | compute=${m.compute.toFixed(2)}ms total=${m.total.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="cover" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? `HDR LUT (BT.2390) ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

export default function CameraSpikeScreen() {
  const router = useRouter();
  const [isRunning, setIsRunning] = useState(false);
  const [shaderIndex, setShaderIndex] = useState(0);
  const [showDepth, setShowDepth] = useState(false);
  const [showOrt, setShowOrt] = useState(false);
  const [lutResource, setLutResource] = useState<ReturnType<typeof GPUResource.texture3D> | null>(null);
  const [depthModelPath, setDepthModelPath] = useState<string | null>(null);
  const shader = SHADERS[shaderIndex];

  // Load bundled .cube LUT on mount — pass file URI, native side parses .cube
  useEffect(() => {
    (async () => {
      const [asset] = await Asset.loadAsync(require('../../assets/AppleLogToRec709-v1.0.cube'));
      if (!asset.localUri) return;
      setLutResource(GPUResource.texture3D(asset.localUri, {
        width: 0, height: 0, depth: 0,
        format: 'rgba32float',
      }));
    })();
  }, []);

  // Download depth model on mount
  useEffect(() => {
    (async () => {
      const file = new ExpoFile(DEPTH_MODEL_PATH);
      if (!file.exists) {
        console.log('[DepthModel] Downloading depth model...');
        const resp = await fetch(DEPTH_MODEL_URL);
        const bytes = new Uint8Array(await resp.arrayBuffer());
        file.write(bytes);
        console.log('[DepthModel] Download complete');
      }
      setDepthModelPath(DEPTH_MODEL_PATH);
    })();
  }, []);

  // Format enumeration — expand into one entry per (format, colorSpace) permutation
  const formats = useCameraFormats('back');
  type FormatEntry = { format: CameraFormat; colorSpace: ColorSpace };
  const formatEntries = formats.flatMap((f) =>
    (f.supportedColorSpaces ?? ['sRGB']).map((cs) => ({ format: f, colorSpace: cs as ColorSpace }))
  );
  const [selected, setSelected] = useState<FormatEntry | undefined>();

  // Auto-select best sRGB format (prefer depth-capable)
  useEffect(() => {
    if (formatEntries.length === 0) return;
    const best =
      formatEntries.find(e => e.colorSpace === 'sRGB' && e.format.width >= 1920 && e.format.maxFps >= 120) ??
      formatEntries.find(e => e.colorSpace === 'sRGB' && e.format.width >= 1920 && e.format.maxFps >= 60) ??
      formatEntries.find(e => e.colorSpace === 'sRGB') ??
      formatEntries[0];
    setSelected(best);
  }, [formats]);

  const selectedFormat = selected?.format;
  const selectedColorSpace = selected?.colorSpace ?? 'sRGB';

  const [showFormatPicker, setShowFormatPicker] = useState(false);
  const [showShaderPicker, setShowShaderPicker] = useState(false);

  if (showDepth) {
    return <DepthEstimation onBack={() => setShowDepth(false)} />;
  }

  if (showOrt) {
    return <OrtTest />;
  }

  const formatLabel = selected
    ? `${selected.format.width}x${selected.format.height} @${Math.round(selected.format.maxFps)} ${selected.colorSpace}`
    : 'Auto';

  return (
    <View style={styles.container}>
      {/* key forces re-mount because pipeline setup captures shader chain */}
      {isRunning && shader.type === 'histogram' && <HistogramPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'histogram-onframe' && <HistogramOnFramePreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'simple' && <CameraPreview key={`${shader.name}-${selectedColorSpace}`} shaderChain={shader.wgsl} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'applelog' && <AppleLogPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} lutResource={lutResource} />}
      {isRunning && shader.type === 'hdr-lut' && <HdrLutPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'depth' && <DepthPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'depth-model' && depthModelPath && <DepthModelPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} modelPath={depthModelPath} />}
      {isRunning && shader.type === 'depth-model' && !depthModelPath && (
        <View style={StyleSheet.absoluteFill}>
          <View style={styles.statusBar}>
            <Text style={styles.statusText}>Downloading depth model...</Text>
          </View>
        </View>
      )}

      <View style={styles.controls}>
        <Pressable
          style={styles.button}
          onPress={() => setShowShaderPicker(true)}
        >
          <Text style={styles.buttonText}>{shader.name}</Text>
        </Pressable>

        {!isRunning && (
          <Pressable
            style={styles.button}
            onPress={() => setShowFormatPicker(true)}
          >
            <Text style={styles.buttonText}>{formatLabel}</Text>
          </Pressable>
        )}

        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={() => setIsRunning(!isRunning)}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>

        {!isRunning && (
          <>
            <Pressable style={styles.button} onPress={() => setShowDepth(true)}>
              <Text style={styles.buttonText}>Depth AI</Text>
            </Pressable>
            <Pressable style={styles.button} onPress={() => setShowOrt(true)}>
              <Text style={styles.buttonText}>ORT Test</Text>
            </Pressable>
            <Pressable style={styles.button} onPress={() => router.push('/skia-video' as any)}>
              <Text style={styles.buttonText}>Skia Video</Text>
            </Pressable>
            <Pressable style={styles.button} onPress={() => router.push('/webgpu-test' as any)}>
              <Text style={styles.buttonText}>WebGPU Test</Text>
            </Pressable>
          </>
        )}
      </View>

      <Modal visible={showFormatPicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Camera Format</Text>
            <FlatList
              data={formatEntries}
              keyExtractor={(e) => `${e.format.nativeHandle}-${e.colorSpace}`}
              renderItem={({ item: e }) => {
                const isSelected = selected?.format.nativeHandle === e.format.nativeHandle && selected?.colorSpace === e.colorSpace;
                return (
                  <Pressable
                    style={[styles.formatRow, isSelected && styles.formatRowSelected]}
                    onPress={() => { setSelected(e); setShowFormatPicker(false); }}
                  >
                    <Text style={[styles.formatText, isSelected && styles.formatTextSelected]}>
                      {e.format.width}x{e.format.height}
                    </Text>
                    <Text style={[styles.formatDetail, isSelected && styles.formatTextSelected]}>
                      {Math.round(e.format.minFps)}-{Math.round(e.format.maxFps)}fps {e.colorSpace}{e.format.isBinned ? ' bin' : ''}{e.format.supportsDepth ? ' D' : ''}
                    </Text>
                  </Pressable>
                );
              }}
            />
            <Pressable style={styles.modalClose} onPress={() => setShowFormatPicker(false)}>
              <Text style={styles.buttonText}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      <Modal visible={showShaderPicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Shader</Text>
            <FlatList
              data={SHADERS}
              keyExtractor={(s, i) => `${i}-${s.name}`}
              renderItem={({ item: s, index: i }) => {
                const isSelected = i === shaderIndex;
                return (
                  <Pressable
                    style={[styles.formatRow, isSelected && styles.formatRowSelected]}
                    onPress={() => { setShaderIndex(i); setShowShaderPicker(false); }}
                  >
                    <Text style={[styles.formatText, isSelected && styles.formatTextSelected]}>
                      {s.name}
                    </Text>
                    <Text style={[styles.formatDetail, isSelected && styles.formatTextSelected]}>
                      {s.type}
                    </Text>
                  </Pressable>
                );
              }}
            />
            <Pressable style={styles.modalClose} onPress={() => setShowShaderPicker(false)}>
              <Text style={styles.buttonText}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  statusBar: {
    position: 'absolute', top: 44, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 4, padding: 8,
  },
  statusText: { color: '#aaa', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 20,
    paddingHorizontal: 16, paddingVertical: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: { backgroundColor: 'rgba(255,80,80,0.4)', borderColor: 'rgba(255,80,80,0.6)' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  modalOverlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a1a', borderTopLeftRadius: 16, borderTopRightRadius: 16,
    maxHeight: '60%', paddingBottom: 40,
  },
  modalTitle: {
    color: '#fff', fontSize: 18, fontWeight: '700',
    padding: 16, textAlign: 'center',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.15)',
  },
  formatRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.08)',
  },
  formatRowSelected: { backgroundColor: 'rgba(80,140,255,0.25)' },
  formatText: { color: '#fff', fontSize: 16, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  formatDetail: { color: '#888', fontSize: 13, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  formatTextSelected: { color: '#6af' },
  modalClose: {
    alignSelf: 'center', marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 20,
    paddingHorizontal: 32, paddingVertical: 12,
  },
});
