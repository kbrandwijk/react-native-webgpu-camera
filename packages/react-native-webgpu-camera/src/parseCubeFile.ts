/**
 * Parse a .cube 3D LUT file into a Float32Array suitable for GPU upload.
 *
 * .cube format:
 *   LUT_3D_SIZE N
 *   R G B    (N³ lines, each with 3 floats in [0,1])
 *
 * Returns RGBA (A=1.0) data for use with GPUResource.texture3D({ format: 'rgba32float' }).
 */
export function parseCubeFile(text: string): { data: Float32Array; size: number } {
  const lines = text.split('\n');
  let size = 0;
  const values: number[] = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;

    if (line.startsWith('LUT_3D_SIZE')) {
      size = parseInt(line.split(/\s+/)[1], 10);
      continue;
    }

    // Skip other metadata lines (TITLE, DOMAIN_MIN, DOMAIN_MAX, LUT_1D_SIZE, etc.)
    if (line.match(/^[A-Z_]/)) continue;

    const parts = line.split(/\s+/);
    if (parts.length >= 3) {
      values.push(
        parseFloat(parts[0]),
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        1.0,
      );
    }
  }

  if (size === 0) {
    throw new Error('parseCubeFile: missing LUT_3D_SIZE header');
  }

  const expected = size * size * size * 4;
  if (values.length !== expected) {
    throw new Error(
      `parseCubeFile: expected ${expected / 4} entries for size ${size}, got ${values.length / 4}`,
    );
  }

  return { data: new Float32Array(values), size };
}
