import { useRef, useState, useCallback } from 'react';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';

export interface GPUPipelineState {
  status: 'idle' | 'initializing' | 'ready' | 'error';
  error?: string;
  deviceSource: 'graphite' | 'rn-wgpu' | 'unknown';
  computeSupported: boolean;
}

interface GPUResources {
  device: GPUDevice;
  computePipeline: GPUComputePipeline;
  inputTexture: GPUTexture;
  outputTexture: GPUTexture;
  bindGroup: GPUBindGroup;
  width: number;
  height: number;
}

export function useGPUPipeline(width = 1920, height = 1080) {
  const [state, setState] = useState<GPUPipelineState>({
    status: 'idle',
    deviceSource: 'unknown',
    computeSupported: false,
  });
  const resources = useRef<GPUResources | null>(null);

  const initialize = useCallback(async () => {
    setState(s => ({ ...s, status: 'initializing' }));

    try {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('navigator.gpu not available — Skia Graphite may not be active');
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No GPU adapter available');
      }

      const device = await adapter.requestDevice();
      console.log('[GPUPipeline] Device acquired from navigator.gpu');

      // Detect device source
      let deviceSource: 'graphite' | 'rn-wgpu' | 'unknown' = 'unknown';
      try {
        const g = globalThis as any;
        if (g.__SKIA_GRAPHITE_ACTIVE__ === true) {
          deviceSource = 'graphite';
        } else {
          const adapterInfo = await adapter.requestAdapterInfo?.();
          if (adapterInfo?.description?.toLowerCase().includes('dawn')) {
            deviceSource = 'rn-wgpu';
          } else {
            deviceSource = 'unknown';
          }
        }
      } catch {
        deviceSource = 'unknown';
      }
      console.log(`[GPUPipeline] Device source: ${deviceSource}`);

      const shaderModule = device.createShaderModule({ code: SOBEL_WGSL });
      console.log('[GPUPipeline] Shader module created');

      const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
      console.log('[GPUPipeline] Compute pipeline created');

      const inputTexture = device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT,
      });

      const outputTexture = device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.RENDER_ATTACHMENT,
      });

      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: inputTexture.createView() },
          { binding: 1, resource: outputTexture.createView() },
        ],
      });

      resources.current = {
        device, computePipeline, inputTexture, outputTexture, bindGroup, width, height,
      };

      setState({ status: 'ready', deviceSource, computeSupported: true });
      console.log('[GPUPipeline] Pipeline ready');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error('[GPUPipeline] Init failed:', msg);
      setState({ status: 'error', error: msg, deviceSource: 'unknown', computeSupported: false });
    }
  }, [width, height]);

  const processFrame = useCallback((pixels: Uint8Array, bytesPerRow: number) => {
    const res = resources.current;
    if (!res) return;

    const t0 = performance.now();

    res.device.queue.writeTexture(
      { texture: res.inputTexture },
      pixels.buffer as ArrayBuffer,
      { bytesPerRow },
      { width: res.width, height: res.height },
    );
    const tImport = performance.now();

    const encoder = res.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(res.computePipeline);
    pass.setBindGroup(0, res.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(res.width / 16),
      Math.ceil(res.height / 16),
    );
    pass.end();
    res.device.queue.submit([encoder.finish()]);
    const tCompute = performance.now();

    return { importMs: tImport - t0, computeMs: tCompute - tImport };
  }, []);

  const cleanup = useCallback(() => {
    const res = resources.current;
    if (res) {
      res.inputTexture.destroy();
      res.outputTexture.destroy();
      resources.current = null;
    }
    setState({ status: 'idle', deviceSource: 'unknown', computeSupported: false });
  }, []);

  return { state, initialize, processFrame, cleanup, resources };
}
