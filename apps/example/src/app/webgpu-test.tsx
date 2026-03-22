// Minimal WebGPU Canvas test — Triangle from react-native-skia examples
// If this renders a colored triangle, WebGPUCanvas works.

import React, { useEffect, useRef } from 'react';
import { StyleSheet, View, Text, Pressable, Platform } from 'react-native';
import type { WebGPUCanvasRef } from '@shopify/react-native-skia';
import { WebGPUCanvas, Skia } from '@shopify/react-native-skia';
import { useRouter } from 'expo-router';

const triangleShader = `
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f( 0.0,  0.5),
    vec2f(-0.5, -0.5),
    vec2f( 0.5, -0.5)
  );
  return vec4f(pos[vertexIndex], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
  return vec4f(1.0, 0.5, 0.2, 1.0);
}
`;

export default function WebGPUTestScreen() {
  const router = useRouter();
  const canvasRef = useRef<WebGPUCanvasRef>(null);
  const animRef = useRef<number>(0);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const timeoutId = setTimeout(async () => {
      if (!canvasRef.current) return;
      if (typeof RNWebGPU === 'undefined') return;

      const ctx = canvasRef.current.getContext('webgpu');
      if (!ctx) return;

      const device = Skia.getDevice();
      const format = navigator.gpu.getPreferredCanvasFormat();

      ctx.configure({ device, format, alphaMode: 'opaque' });

      const shaderModule = device.createShaderModule({ code: triangleShader });

      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shaderModule, entryPoint: 'vs_main' },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_main',
          targets: [{ format }],
        },
        primitive: { topology: 'triangle-list' },
      });

      let running = true;

      const render = () => {
        if (!running) return;

        const texture = ctx.getCurrentTexture();
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass({
          colorAttachments: [{
            view: texture.createView(),
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        passEncoder.setPipeline(pipeline);
        passEncoder.draw(3);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        ctx.present();

        animRef.current = requestAnimationFrame(render);
      };

      animRef.current = requestAnimationFrame(render);
      cleanupRef.current = () => { running = false; cancelAnimationFrame(animRef.current); };
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      cleanupRef.current?.();
    };
  }, []);

  if (typeof RNWebGPU === 'undefined') {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>WebGPU not available (SK_GRAPHITE required)</Text>
        <Pressable style={styles.button} onPress={() => router.back()}>
          <Text style={styles.buttonText}>Back</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <WebGPUCanvas ref={canvasRef} style={styles.canvas} />
      <View style={styles.controls}>
        <Pressable style={styles.button} onPress={() => router.back()}>
          <Text style={styles.buttonText}>Back</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a1a' },
  canvas: { flex: 1 },
  text: { color: '#fff', fontSize: 16, textAlign: 'center', marginTop: 100 },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'center',
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 20,
    paddingHorizontal: 16, paddingVertical: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
