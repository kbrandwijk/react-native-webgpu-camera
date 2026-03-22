import { useState, useEffect, useRef } from 'react';
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
import type { WebGPUCanvasRef } from '@shopify/react-native-skia';
import { WebGPUCanvas, Skia } from '@shopify/react-native-skia';
import { useRouter } from 'expo-router';

// Shader presets
import { createDeband, createToneMap, createGrain, createDither, createConeDistort, createSigmoidize } from 'webgpu-video-shaders/libplacebo';
import { createVignette } from 'webgpu-video-shaders/original';

// ─── Shader presets (same composable functions as camera pipeline) ──────

interface ShaderPreset {
  name: string;
  wgsl: string;
}

function passthrough(): string {
  return /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  textureStore(outputTex, vec2i(id.xy), textureLoad(inputTex, vec2i(id.xy), 0));
}
`;
}

function wrapColor(fnCode: string, call: string): string {
  return /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
${fnCode}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  let color = textureLoad(inputTex, coord, 0);
  textureStore(outputTex, coord, ${call});
}
`;
}

function wrapTex(fnCode: string, fnName: string): string {
  return /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
${fnCode}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  textureStore(outputTex, coord, ${fnName}(inputTex, coord, vec2i(dims)));
}
`;
}

const PRESETS: ShaderPreset[] = [];

PRESETS.push({ name: 'None', wgsl: passthrough() });

const deband = createDeband({ iterations: 2, threshold: 4, grain: 6 });
PRESETS.push({ name: 'Deband', wgsl: wrapTex(deband.fn, deband.fnName) });

const tmHable = createToneMap({ method: 'hable', srcPeakNits: 1000, dstPeakNits: 203 });
PRESETS.push({ name: 'Tonemap', wgsl: wrapColor(tmHable.fn, `${tmHable.fnName}(color)`) });

const deuter = createConeDistort({ type: 'deuteranopia' });
PRESETS.push({ name: 'Deuteranopia', wgsl: wrapColor(deuter.fn, `${deuter.fnName}(color)`) });

const achrom = createConeDistort({ type: 'achromatopsia' });
PRESETS.push({ name: 'Monochrome', wgsl: wrapColor(achrom.fn, `${achrom.fnName}(color)`) });

const dith1 = createDither({ method: 'ordered', targetDepth: 1 });
PRESETS.push({ name: 'Dither 1-bit', wgsl: wrapColor(dith1.fn, `${dith1.fnName}(color, coord)`) });

const sigmoid = createSigmoidize({ center: 0.5, slope: 10 });
PRESETS.push({ name: 'Sigmoid', wgsl: wrapColor(sigmoid.fn, `${sigmoid.fnName}(color)`) });

const vig = createVignette({ strength: 0.6, innerRadius: 0.3, outerRadius: 1.0 });
PRESETS.push({ name: 'Vignette', wgsl: wrapColor(vig.fn, `${vig.fnName}(color, coord, vec2i(dims))`) });

// Noir composed
const noirMono = createConeDistort({ type: 'achromatopsia' });
const noirSig = createSigmoidize({ center: 0.5, slope: 10 });
const noirVig = createVignette({ strength: 0.6, innerRadius: 0.3, outerRadius: 1.0 });
const noirGrain = createGrain({ amount: 8 });
const noirDith = createDither({ method: 'ordered', targetDepth: 6 });
PRESETS.push({
  name: 'Film Noir',
  wgsl: /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
${noirMono.fn}
${noirSig.fn}
${noirVig.fn}
${noirGrain.fn}
${noirDith.fn}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let coord = vec2i(id.xy);
  var color = textureLoad(inputTex, coord, 0);
  color = ${noirMono.fnName}(color);
  color = ${noirSig.fnName}(color);
  color = ${noirVig.fnName}(color, coord, vec2i(dims));
  color = ${noirGrain.fnName}(color, coord);
  color = ${noirDith.fnName}(color, coord);
  textureStore(outputTex, coord, color);
}
`,
});

// ─── Video URL ──────────────────────────────────────────────────────────
const VIDEO_URL = 'https://bit.ly/skia-video';

// ─── Component ──────────────────────────────────────────────────────────

export default function SkiaVideoScreen() {
  const router = useRouter();
  const { width: screenW, height: screenH } = useWindowDimensions();
  const canvasRef = useRef<WebGPUCanvasRef>(null);
  const [presetIndex, setPresetIndex] = useState(0);
  const [showPicker, setShowPicker] = useState(false);
  const [status, setStatus] = useState('Initializing...');
  const presetIndexRef = useRef(0);
  const pipelinesRef = useRef<GPUComputePipeline[]>([]);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Keep ref in sync with state
  presetIndexRef.current = presetIndex;

  useEffect(() => {
    const timeoutId = setTimeout(async () => {
      if (!canvasRef.current) return;
      if (typeof RNWebGPU === 'undefined') {
        setStatus('WebGPU not available (SK_GRAPHITE required)');
        return;
      }

      const ctx = canvasRef.current.getContext('webgpu');
      if (!ctx) {
        setStatus('Failed to get WebGPU context');
        return;
      }

      const device = Skia.getDevice();
      const format = navigator.gpu.getPreferredCanvasFormat();

      ctx.configure({
        device,
        format,
        alphaMode: 'opaque',
      });

      // Create blit render pipeline (compute output → canvas surface)
      const blitShader = device.createShaderModule({ code: `
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VsOut;
  o.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(uv.x, 1.0 - uv.y);
  return o;
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  return textureSample(srcTex, srcSamp, in.uv);
}
` });

      const blitPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: blitShader, entryPoint: 'vs' },
        fragment: {
          module: blitShader,
          entryPoint: 'fs',
          targets: [{ format }],
        },
        primitive: { topology: 'triangle-list' },
      });

      const blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

      // Pre-compile all compute pipelines
      const pipelines: GPUComputePipeline[] = [];
      for (const preset of PRESETS) {
        try {
          pipelines.push(device.createComputePipeline({
            layout: 'auto',
            compute: {
              module: device.createShaderModule({ code: preset.wgsl }),
              entryPoint: 'main',
            },
          }));
        } catch (e) {
          console.warn(`Failed to compile shader ${preset.name}:`, e);
          pipelines.push(device.createComputePipeline({
            layout: 'auto',
            compute: {
              module: device.createShaderModule({ code: passthrough() }),
              entryPoint: 'main',
            },
          }));
        }
      }
      pipelinesRef.current = pipelines;

      // Start video
      const video = await Skia.Video(VIDEO_URL);
      video.setLooping(true);
      video.setVolume(0);
      video.play();

      let running = true;
      let frameCount = 0;
      let lastFpsTime = Date.now();

      const render = () => {
        if (!running) return;

        const frame = video.nextImage();
        if (!frame) {
          animId = requestAnimationFrame(render);
          return;
        }

        // Convert SkImage → GPUTexture (shared Dawn device, zero-copy)
        const inputTexture = Skia.Image.MakeTextureFromImage(frame);
        if (!inputTexture) {
          frame.dispose();
          animId = requestAnimationFrame(render);
          return;
        }

        const w = (inputTexture as GPUTexture).width;
        const h = (inputTexture as GPUTexture).height;

        // Create output storage texture (also needs TEXTURE_BINDING for blit sampling)
        const outputTexture = device.createTexture({
          size: { width: w, height: h },
          format: 'rgba8unorm',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        const computePipeline = pipelines[presetIndexRef.current] ?? pipelines[0];

        const encoder = device.createCommandEncoder();

        // Compute pass: apply shader effect
        const computeBindGroup = device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: (inputTexture as GPUTexture).createView() },
            { binding: 1, resource: outputTexture.createView() },
          ],
        });
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        computePass.end();

        // Blit render pass: compute output → canvas surface
        const canvasTex = ctx.getCurrentTexture();
        const blitBindGroup = device.createBindGroup({
          layout: blitPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: outputTexture.createView() },
            { binding: 1, resource: blitSampler },
          ],
        });
        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: canvasTex.createView(),
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        renderPass.setPipeline(blitPipeline);
        renderPass.setBindGroup(0, blitBindGroup);
        renderPass.draw(3);
        renderPass.end();

        device.queue.submit([encoder.finish()]);
        ctx.present();

        outputTexture.destroy();
        frame.dispose();

        // FPS
        frameCount++;
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
          setStatus(`${w}x${h} @ ${frameCount}fps — ${PRESETS[presetIndexRef.current].name}`);
          frameCount = 0;
          lastFpsTime = now;
        }

        animId = requestAnimationFrame(render);
      };

      let animId = requestAnimationFrame(render);

      cleanupRef.current = () => {
        running = false;
        cancelAnimationFrame(animId);
        video.dispose();
      };

      setStatus('Playing...');
    }, 200);

    return () => {
      clearTimeout(timeoutId);
      cleanupRef.current?.();
    };
  }, []);

  const preset = PRESETS[presetIndex];

  if (typeof RNWebGPU === 'undefined') {
    return (
      <View style={styles.container}>
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>WebGPU Canvas requires SK_GRAPHITE</Text>
        </View>
        <View style={styles.controls}>
          <Pressable style={styles.button} onPress={() => router.back()}>
            <Text style={styles.buttonText}>Back</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <WebGPUCanvas ref={canvasRef} style={StyleSheet.absoluteFill} />

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      <View style={styles.controls}>
        <Pressable style={styles.button} onPress={() => router.back()}>
          <Text style={styles.buttonText}>Back</Text>
        </Pressable>
        <Pressable style={styles.button} onPress={() => setShowPicker(true)}>
          <Text style={styles.buttonText}>{preset.name}</Text>
        </Pressable>
      </View>

      <Modal visible={showPicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Shader</Text>
            <FlatList
              data={PRESETS}
              keyExtractor={(_, i) => String(i)}
              renderItem={({ item, index }) => (
                <Pressable
                  style={[styles.row, index === presetIndex && styles.rowSelected]}
                  onPress={() => { setPresetIndex(index); setShowPicker(false); }}
                >
                  <Text style={[styles.rowText, index === presetIndex && styles.rowTextSelected]}>
                    {item.name}
                  </Text>
                </Pressable>
              )}
            />
            <Pressable style={styles.modalClose} onPress={() => setShowPicker(false)}>
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
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'flex-end' },
  modalContent: {
    backgroundColor: '#1a1a1a', borderTopLeftRadius: 16, borderTopRightRadius: 16,
    maxHeight: '60%', paddingBottom: 40,
  },
  modalTitle: {
    color: '#fff', fontSize: 18, fontWeight: '700', padding: 16, textAlign: 'center',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.15)',
  },
  row: {
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.08)',
  },
  rowSelected: { backgroundColor: 'rgba(80,140,255,0.25)' },
  rowText: { color: '#fff', fontSize: 16, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  rowTextSelected: { color: '#6af' },
  modalClose: {
    alignSelf: 'center', marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 20,
    paddingHorizontal: 32, paddingVertical: 12,
  },
});
