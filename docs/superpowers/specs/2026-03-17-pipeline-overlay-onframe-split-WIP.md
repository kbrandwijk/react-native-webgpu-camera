# Pipeline / Overlay / onFrame Split — WIP Design Notes

**Status:** Work in progress. Open questions remain around compositing order and alpha handling.

## Goal

Separate frame processing into three layers with clear recording semantics:

| Layer | Type | Recorded? | Purpose |
|-------|------|-----------|---------|
| **pipeline** | Compute shaders | Yes | Image processing (color grading, LUT, edge detection) |
| **onFrame** | Skia canvas draws | Yes | Baked-in overlays (bounding boxes, text burn-in) |
| **overlay** | Compute shaders | No | Display-only monitoring (histogram viz, zebra stripes, focus peaking) |

## API (from existing custom-bind-groups spec)

```ts
useGPUFrameProcessor(camera, {
  resources: {
    lut: GPUResource.texture3D(cubeData, { width: 33, height: 33, depth: 33 }),
  },
  pipeline: (frame, { lut }) => {
    'worklet';
    frame.runShader(LUT_WGSL, { inputs: { lut } });
    const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
    return { hist };
  },
  overlay: (frame, { hist }) => {
    'worklet';
    frame.runShader(HISTOGRAM_OVERLAY_WGSL, { inputs: { hist } });
    frame.runShader(ZEBRA_WGSL);
  },
  onFrame: (frame, { hist }) => {
    'worklet';
    if (hist) {
      frame.canvas.drawText(`peak: ${Math.max(...hist)}`, 100, 100);
    }
  },
});
```

- `pipeline` and `overlay` share the same `ProcessorFrame` signature (can run shaders, access resources)
- `onFrame` uses `RenderFrame` (Skia canvas)
- `overlay` receives pipeline's buffer outputs but does NOT feed into the recorded frame
- `pipeline` runs once — both `onFrame` and `overlay` consume its output

## Agreed Architecture

### Non-recording path (zero overhead)

When not recording, overlay shaders continue on the same ping-pong textures after pipeline passes. No snapshot, no copy, no extra texture needed. The display shows the final result (pipeline + overlay + onFrame composited).

### Recording path

When recording starts:
1. Snapshot the post-pipeline output (the "recorded frame" source)
2. Copy to overlay texture
3. Run overlay passes on the copy for display
4. Recording captures pipeline + onFrame output only (no overlay)

This means the recording path adds one texture copy. The non-recording path has zero overhead compared to the current implementation.

## Open Questions

### 1. Execution order

Pipeline runs first (established). But the order of overlay and onFrame relative to each other matters:

**Option A: pipeline → onFrame → overlay**
- onFrame draws are baked into the recorded frame
- overlay runs on pipeline output (not on onFrame draws)
- Display shows: recorded frame (pipeline + onFrame) with overlay composited on top
- Question: how to composite? Two SkImages? Alpha blending?

**Option B: pipeline → overlay → onFrame**
- overlay runs on raw pipeline output (good for analysis shaders like histogram)
- onFrame draws on top of everything for display, but only pipeline+onFrame for recording
- Problem: onFrame draws would appear on top of overlay on screen, but overlay isn't in the recording — visual mismatch between preview and recording

**Option C: pipeline output consumed in parallel by onFrame and overlay**
```
pipeline → pipelineOutput
              ├→ onFrame (Skia canvas draws on copy) → recorded frame
              └→ overlay (compute shaders on copy) → display-only frame
Display: recorded frame + overlay composited on top
```
- Clean separation but needs two copies of pipeline output
- Overlay analyzes the raw pipeline output (correct for histogram/zebra)
- onFrame draws on a separate copy (correct for recording)
- Display composites both layers

### 2. Alpha / compositing

If overlay output is composited on top of the recorded frame for display:
- Does overlay need alpha transparency? (semi-transparent histogram backgrounds, zebra stripes over the image)
- Or is overlay always fully opaque? (replaces the display entirely)
- This determines whether we need alpha blending in the Skia display layer or just stacking two SkImages

### 3. Non-recording performance

In the common case (not recording), can we avoid the parallel path entirely?
- Just run pipeline → overlay → onFrame sequentially on the same textures
- Everything composites into one output for display
- Only split into parallel paths when recording starts

This would mean the overlay output IS visible in the "recorded frame" during preview (since there's no recording happening). When recording starts, the split activates. This seems pragmatic but means the preview doesn't perfectly match what gets recorded.

## What's Already Implemented

- `pipeline` callback — fully working (compute shaders, multi-pass, resources, buffer outputs)
- `onFrame` callback — fully working (Skia canvas draws baked into frame via flushCanvasAndGetImage)
- `overlay` callback — **not yet implemented** (this spec)
- Recording — **not yet implemented** (separate future work)

## References

- `docs/superpowers/specs/2026-03-15-custom-bind-groups-design.md` — API types and binding system
- `docs/superpowers/specs/2026-03-13-use-gpu-frame-processor-design.md` — original hook design
