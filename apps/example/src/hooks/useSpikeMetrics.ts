import { useRef, useCallback } from 'react';

export interface SpikeResults {
  spike1Path: 'zero-copy' | 'copy-fallback' | 'unknown';
  spike1AvgMs: number;
  spike2Path: 'worklet-compute' | 'main-thread-compute' | 'unknown';
  spike2AvgMs: number;
  spike3Path: 'graphite' | 'ganesh-fallback' | 'unknown';
  spike3OverheadMs: number;
  spike4Path: 'surface-record' | 'readback-record' | 'unknown';
  sustainedFps: number;
  frameDrops: number;
  totalFrames: number;
  peakMemoryMb: number;
  thermalTransitions: string[];
}

interface FrameTiming {
  importMs: number;
  computeMs: number;
  skiaMs: number;
  totalMs: number;
}

export function useSpikeMetrics() {
  const startTime = useRef(0);
  const frameTimings = useRef<FrameTiming[]>([]);
  const lastFrameTime = useRef(0);
  const frameDropCount = useRef(0);
  const thermalTransitions = useRef<string[]>([]);
  const lastThermalState = useRef('nominal');

  const recordFrame = useCallback((timing: FrameTiming) => {
    const now = performance.now();

    if (startTime.current === 0) {
      startTime.current = now;
    }

    // Detect frame drops (>20ms gap = missed a frame at 60fps)
    if (lastFrameTime.current > 0) {
      const gap = now - lastFrameTime.current;
      if (gap > 20) {
        frameDropCount.current += Math.floor(gap / 16.67) - 1;
      }
    }
    lastFrameTime.current = now;

    frameTimings.current.push(timing);
  }, []);

  const recordThermalChange = useCallback((newState: string) => {
    const now = performance.now();
    if (newState !== lastThermalState.current) {
      const elapsed = ((now - startTime.current) / 1000).toFixed(0);
      thermalTransitions.current.push(
        `${lastThermalState.current} -> ${newState} at ${elapsed}s`
      );
      lastThermalState.current = newState;
    }
  }, []);

  const getSummary = useCallback((): SpikeResults | null => {
    const timings = frameTimings.current;
    if (timings.length === 0) return null;

    const elapsed = (performance.now() - startTime.current) / 1000;
    const avgImport =
      timings.reduce((s, t) => s + t.importMs, 0) / timings.length;
    const avgCompute =
      timings.reduce((s, t) => s + t.computeMs, 0) / timings.length;
    const avgSkia =
      timings.reduce((s, t) => s + t.skiaMs, 0) / timings.length;

    return {
      spike1Path: 'unknown',
      spike1AvgMs: avgImport,
      spike2Path: 'unknown',
      spike2AvgMs: avgCompute,
      spike3Path: 'unknown',
      spike3OverheadMs: avgSkia,
      spike4Path: 'unknown',
      sustainedFps: timings.length / elapsed,
      frameDrops: frameDropCount.current,
      totalFrames: timings.length,
      peakMemoryMb: 0,
      thermalTransitions: thermalTransitions.current,
    };
  }, []);

  const logSummary = useCallback(
    (paths: Partial<SpikeResults>) => {
      const summary = getSummary();
      if (!summary) return;

      const merged = { ...summary, ...paths };

      console.log('=== SPIKE RESULTS ===');
      console.log(
        `Spike 1: ${merged.spike1Path} (${merged.spike1AvgMs.toFixed(2)}ms avg)`
      );
      console.log(
        `Spike 2: ${merged.spike2Path} (${merged.spike2AvgMs.toFixed(2)}ms avg)`
      );
      console.log(
        `Spike 3: ${merged.spike3Path} (${merged.spike3OverheadMs.toFixed(2)}ms overhead)`
      );
      console.log(`Spike 4: ${merged.spike4Path}`);
      console.log(`Sustained FPS: ${merged.sustainedFps.toFixed(1)}`);
      console.log(
        `Frame drops: ${merged.frameDrops}/${merged.totalFrames} (${((merged.frameDrops / merged.totalFrames) * 100).toFixed(1)}%)`
      );
      console.log(`Memory: ${merged.peakMemoryMb.toFixed(0)}MB peak`);
      console.log(
        `Thermal: ${merged.thermalTransitions.length === 0 ? 'nominal (no transitions)' : merged.thermalTransitions.join(', ')}`
      );
      console.log('========================');
    },
    [getSummary]
  );

  const reset = useCallback(() => {
    startTime.current = 0;
    frameTimings.current = [];
    lastFrameTime.current = 0;
    frameDropCount.current = 0;
    thermalTransitions.current = [];
    lastThermalState.current = 'nominal';
  }, []);

  return { recordFrame, recordThermalChange, getSummary, logSummary, reset };
}
