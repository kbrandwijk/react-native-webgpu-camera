# Slug GPU Text Rendering — Design Spec

## What This Is

A pure TypeScript/WGSL package (`webgpu-slug-text`) that renders resolution-independent text on the GPU using the Slug algorithm. Text is sharp at any zoom level, any 3D angle, any resolution — no texture atlases, no signed distance fields, no pre-rasterization.

The algorithm was created by Eric Lengyel and published in the Journal of Computer Graphics Techniques. This implementation ports the core algorithm to WebGPU/WGSL.

## How It Works

The Slug algorithm renders text by evaluating quadratic Bézier curves directly in the fragment shader. Each glyph is drawn as a quad, and the shader determines per-pixel coverage by counting curve crossings — the same winding-number approach used in CPU font rasterizers, but running on the GPU.

### Three-Stage Pipeline

```
Stage 1 (CPU, one-time)     Stage 2 (GPU compute, per text change)     Stage 3 (GPU fragment, per frame)
─────────────────────────   ──────────────────────────────────────────  ─────────────────────────────────
Parse .ttf file             Decompose glyphs into horizontal bands     Render quads with curve evaluation
Extract Bézier control      Pack curve segments per band               Per-pixel coverage calculation
points per glyph            Write band-data + curve-data buffers       Antialiasing at all scales
Layout text string          Generate per-glyph quad instances          3D perspective transforms
```

### Stage 1: Font Parsing (CPU, JavaScript)

Uses `opentype.js` to parse `.ttf`/`.otf` files. Extracts:
- Glyph outlines as quadratic Bézier curves (start, control, end points)
- Glyph metrics (advance width, bearing, bounding box)
- Kerning pairs

The output is a `GlyphAtlas` — a flat typed array of control points indexed by glyph ID, plus a metrics table. This is computed once per font and can be cached.

```typescript
interface GlyphAtlas {
  /** Flat Float32Array of control points: [x0,y0, cx,cy, x1,y1, ...] per curve */
  curves: Float32Array;
  /** Per-glyph index into curves array: [offset, curveCount, ...] */
  glyphIndex: Uint32Array;
  /** Per-glyph metrics: [advanceWidth, lsb, bbox_x0, bbox_y0, bbox_x1, bbox_y1, ...] */
  metrics: Float32Array;
  /** Units per em (for normalization) */
  unitsPerEm: number;
}
```

Text layout converts a string + font into a list of glyph instances with positions:

```typescript
interface TextLayout {
  /** Glyph IDs in display order */
  glyphIds: Uint32Array;
  /** Per-glyph x,y positions (in em units) */
  positions: Float32Array;
  /** Total bounding box */
  bounds: { x: number; y: number; width: number; height: number };
}
```

### Stage 2: Band Decomposition (GPU Compute Shader)

The compute shader takes the raw curve data and decomposes each glyph into horizontal bands. Each band stores references to the curves that cross it. This is the key optimization that makes per-pixel evaluation fast — instead of testing all curves, the fragment shader only tests curves in the pixel's band.

**Input:**
- `curves` buffer — flat array of Bézier control points
- `glyphIndex` buffer — per-glyph offset and count into curves
- Uniform: number of bands per glyph (typically 8-16)

**Output:**
- `bandData` buffer — per-band: offset into curveRefs, count of curves in this band
- `curveRefs` buffer — sorted list of curve indices per band

The compute shader dispatches one thread per glyph. Each thread:
1. Reads the glyph's curves from the curves buffer
2. For each curve, determines which bands it crosses (based on Y range)
3. Writes curve references to the appropriate bands
4. Stores band metadata (offset + count)

```wgsl
@compute @workgroup_size(64)
fn decompose(@builtin(global_invocation_id) id: vec3u) {
  let glyphId = id.x;
  // Read curves for this glyph
  // For each curve, compute Y range → assign to bands
  // Write band data + curve references
}
```

This runs once per text change (not per frame).

### Stage 3: Fragment Rendering (GPU Render Pipeline)

Each glyph is rendered as a screen-space quad (two triangles) covering the glyph's bounding box. The vertex shader transforms the quad corners by the text's model-view-projection matrix, enabling arbitrary 3D positioning.

The fragment shader:
1. Maps the pixel position back to glyph-local coordinates
2. Determines which band the pixel falls in
3. Reads the curve references for that band
4. For each curve: evaluates the quadratic Bézier, counts horizontal ray crossings
5. Uses the crossing count (odd = inside, even = outside) for coverage
6. Applies antialiasing based on screen-space derivatives (`dpdx`/`dpdy`)

```wgsl
@fragment
fn render(in: VertexOutput) -> @location(0) vec4f {
  let glyphUV = in.glyphUV; // normalized position within glyph bbox
  let bandIdx = u32(glyphUV.y * f32(BAND_COUNT));

  // Read band data
  let bandOffset = bandData[glyphId * BAND_COUNT + bandIdx].offset;
  let bandCount = bandData[glyphId * BAND_COUNT + bandIdx].count;

  // Count crossings
  var crossings = 0;
  for (var i = 0u; i < bandCount; i++) {
    let curveIdx = curveRefs[bandOffset + i];
    let p0 = curves[curveIdx * 3 + 0];
    let cp = curves[curveIdx * 3 + 1];
    let p1 = curves[curveIdx * 3 + 2];
    crossings += evaluateQuadraticCrossing(glyphUV, p0, cp, p1);
  }

  // Coverage: odd crossings = inside
  let inside = (crossings & 1) == 1;

  // Antialiasing via screen-space derivatives
  let coverage = computeAntialiasedCoverage(glyphUV, crossings, dpdx(glyphUV), dpdy(glyphUV));

  return vec4f(in.color.rgb, in.color.a * coverage);
}
```

## Package Structure

```
webgpu-slug-text/
├── src/
│   ├── index.ts                 # Public exports
│   ├── parse-font.ts            # TTF parsing via opentype.js → GlyphAtlas
│   ├── layout-text.ts           # String + font → TextLayout (positions, kerning)
│   ├── band-decompose.ts        # Compute shader generator for band decomposition
│   ├── slug-renderer.ts         # Fragment shader generator for glyph rendering
│   ├── slug-pipeline.ts         # High-level: font + text + transform → rendered text
│   └── types.ts                 # GlyphAtlas, TextLayout, SlugRenderParams
├── package.json
├── tsconfig.json
├── tsup.config.ts
└── README.md
```

## API

### Low-level (shader generators)

```typescript
import { parseFont, layoutText, createBandDecomposeShader, createSlugRenderShader } from 'webgpu-slug-text';

// Stage 1: Parse font (CPU, one-time)
const atlas = await parseFont('path/to/font.ttf');
const layout = layoutText(atlas, 'Hello World', { fontSize: 48 });

// Stage 2: Band decomposition (GPU compute, per text change)
const { wgsl: decomposeShader, bindings } = createBandDecomposeShader({
  bandCount: 12,
});

// Stage 3: Rendering (GPU fragment, per frame)
const { vertexShader, fragmentShader, bindings } = createSlugRenderShader({
  bandCount: 12,
  antialiasing: true,
});
```

### High-level (batteries-included)

```typescript
import { SlugTextRenderer } from 'webgpu-slug-text';

const renderer = await SlugTextRenderer.create(device, 'font.ttf');
renderer.setText('Hello World', { fontSize: 48, color: [1, 1, 1, 1] });

// In render loop:
renderer.render(renderPass, {
  transform: viewProjectionMatrix,
  position: [0, 0, 0],
});
```

### With WebGPUCanvas (React Native)

```typescript
import { SlugTextRenderer } from 'webgpu-slug-text';
import { WebGPUCanvas, Skia } from '@shopify/react-native-skia';

// In component:
const device = Skia.getDevice();
const renderer = await SlugTextRenderer.create(device, bundledFont);
renderer.setText('GPU Text', { fontSize: 64 });

// In render loop:
const texture = ctx.getCurrentTexture();
const renderPass = encoder.beginRenderPass({
  colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
});
renderer.render(renderPass, { transform: mvp });
renderPass.end();
ctx.present();
```

## Key Design Decisions

### Why bands, not a full curve search?

A glyph like 'B' has ~30 curves. Testing all 30 per pixel at 4K resolution is expensive. Bands partition the glyph vertically so each pixel only tests 3-5 curves. This is the core Slug optimization.

### Why a compute shader for decomposition?

Band decomposition is embarrassingly parallel — each glyph is independent. The compute shader processes all glyphs in one dispatch. The output buffers persist until the text changes, so the per-frame cost is zero.

### Why quadratic Béziers only?

TrueType fonts use quadratic Béziers natively. OpenType/CFF fonts use cubic Béziers, which can be decomposed into quadratics at parse time. Quadratic evaluation is simpler and faster in the shader (one fewer multiply per curve).

### Antialiasing approach

The Slug algorithm uses analytic antialiasing based on screen-space derivatives. At each pixel, the fragment shader knows the exact distance to the nearest curve and the pixel's screen-space size. Coverage is computed as a smooth transition over a 1-pixel-wide band at the glyph edge. This produces perfect antialiasing at all scales — no blur at magnification, no aliasing at minification.

### Band count

8-16 bands per glyph is typical. More bands = fewer curves per band = faster fragment shader, but more memory for band data. 12 is a good default.

## Dependencies

- `opentype.js` — TTF/OTF parsing (peer dependency, ~100KB)
- No native code
- No GPU framework dependency (works with any WebGPU pipeline)

## Performance Expectations

- **Parse**: ~10ms for a typical font (one-time)
- **Layout**: <1ms for a paragraph of text
- **Band decompose**: <0.5ms compute dispatch for 1000 glyphs
- **Render**: Per glyph quad is ~5-15 curve evaluations in the fragment shader. At 4K, a full screen of text renders in <2ms on mobile GPU.
- **Memory**: ~50KB per font (curves + metrics), ~2KB per 1000 characters (band + instance data)

## What This Does NOT Include

- Text shaping (complex script support like Arabic, Devanagari) — use HarfBuzz for that
- Paragraph layout (line breaking, justification) — use a layout engine
- Color emoji — requires raster fallback
- Variable fonts — could be added later
- SDF fallback — not needed, Slug is superior at all scales
