# Architecture Decisions Log

Decisions made during spike implementation that affect the system design.
These inform Phase 1 architecture — they are not final, but document what we
learned and why things are the way they are.

---

## AD-1: SkImage creation in processFrame (camera thread)

**Date:** 2026-03-14
**Status:** Active (revised)
**Context:**

We initially believed Skia Graphite's thread-local `Recorder` would prevent
an `SkImage` created on the camera thread from rendering on the UI thread.
This led to a deferred-creation approach (dirty flag + lazy `MakeImageFromTexture`
on the UI thread).

**What actually happened:** The deferred approach produced frames but they
rendered as nothing. Reverting to create the SkImage directly in `processFrame()`
on the camera thread — matching the working single-pass implementation — fixed
rendering immediately.

**Decision:** `processFrame()` calls `MakeImageFromTexture()` at the end of
each frame, on the camera thread. `getOutputSkImage()` simply returns the
cached image. This matches the single-pass pattern that was proven to work.

**Lesson:** The thread-local recorder theory was incorrect (or at least not
the root cause). The working approach is simpler and avoids the complexity of
cross-thread lazy creation.

---

## AD-2: Obtaining a Graphite Recorder from DawnContext

**Date:** 2026-03-14
**Status:** Active (workaround in place, upstream PR pending)
**PR:** https://github.com/Shopify/react-native-skia/pull/3751

**Context:**

To create an `SkSurface` backed by an existing `wgpu::Texture` (for canvas
overlays on the compute output), we need `SkSurfaces::WrapBackendTexture()`
which requires a `skgpu::graphite::Recorder*`. `DawnContext::getRecorder()` is
private.

**Decision:** Use a 1×1 throwaway offscreen surface to obtain the recorder:
```cpp
auto tempSurface = ctx.MakeOffscreen(1, 1);
auto* recorder = tempSurface->recorder();
```

This works because the recorder is a thread-local singleton — all surfaces on
the same thread share the same recorder instance. The 1×1 surface is cheap and
only created once during `setup()`.

**Upstream fix:** PR #3751 to react-native-skia makes `getRecorder()` public,
consistent with `getWGPUDevice()` and `getWGPUInstance()` already being public.
Once merged, the workaround can be replaced with `ctx.getRecorder()`.

---

## AD-3: Canvas overlay draws directly onto compute output texture

**Date:** 2026-03-14
**Status:** Active

**Context:**

When `useCanvas` is enabled, users can draw Skia overlays (text, shapes, debug
visualizations) on the compute output. Two approaches:

1. **Separate surface** — canvas draws to its own texture, JS composites both
   layers. Requires JS-side composition logic and an extra texture.
2. **Shared surface** — canvas draws directly onto the compute output texture
   via `WrapBackendTexture`. Single texture, no composition needed.

**Decision:** Option 2 — `WrapBackendTexture` wraps the final ping-pong output
texture as an `SkSurface`. Canvas drawing commands land directly on the compute
result. `flushCanvas()` snaps the surface's recording and submits it, then
re-wraps the texture as a fresh SkImage for display.

**Consequence:** Canvas content overwrites the compute output pixels. This is
the desired behavior (overlays on top of processed frames). If non-destructive
layering is needed later, we'd switch to option 1.

---

## AD-4: StagingBuffer uses plain bool instead of std::atomic

**Date:** 2026-03-14
**Status:** Active

**Context:**

`StagingBuffer` originally used `std::atomic<bool> mapped[2]` for the
double-buffered map state. `std::atomic` is not move-constructible, which
prevents `std::vector<StagingBuffer>` from compiling (vector resize needs
move/copy).

**Decision:** Use plain `bool mapped[2]` instead. All access to `mapped[]` is
already serialized by the pipeline's `_mutex` — `processFrame()`,
`readBuffer()`, and `cleanupLocked()` all hold the lock. The atomic was
redundant.

**Exception:** The `MapAsync` callback sets `mapped[idx] = true` asynchronously.
In sync mode this is safe (we spin on `device.Tick()` under the lock). In async
mode this is technically a data race, but benign — the worst case is reading a
stale `false` and skipping one frame of buffer data. If this becomes a problem,
wrap `mapped` in `std::unique_ptr<std::atomic<bool>[]>` to restore atomicity
without breaking move semantics.
