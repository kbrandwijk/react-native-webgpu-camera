/** Type tag for resource handles */
export interface ResourceHandle<T extends string> {
  readonly __resourceType: T;
  readonly __handle: number;
  readonly __data?: ArrayBuffer;
  readonly __dims?: { width: number; height: number; depth?: number; format?: string };
}

/** Sentinel for output type tokens */
export interface OutputTypeToken<T extends string> {
  readonly __outputType: T;
}

function texture3D(
  data: ArrayBuffer,
  dims: { width: number; height: number; depth: number; format?: 'rgba8unorm' | 'rgba32float' },
): ResourceHandle<'texture3d'> {
  return {
    __resourceType: 'texture3d',
    __handle: -1, // assigned by capture proxy
    __data: data,
    __dims: dims,
  };
}

function texture2DResource(
  data: ArrayBuffer,
  dims: { width: number; height: number },
): ResourceHandle<'texture2d'> {
  return {
    __resourceType: 'texture2d',
    __handle: -1, // assigned by capture proxy
    __data: data,
    __dims: dims,
  };
}

function storageBuffer(data: ArrayBuffer): ResourceHandle<'storageBuffer'> {
  return {
    __resourceType: 'storageBuffer',
    __handle: -1, // assigned by capture proxy
    __data: data,
  };
}

/** Token used as output type: frame.runShader(WGSL, { output: GPUResource.texture2D }) */
const texture2DToken: OutputTypeToken<'texture2d'> = {
  __outputType: 'texture2d',
};

/**
 * GPUResource constructors for creating typed GPU handles.
 *
 * In `resources` block: GPUResource.texture3D(data, dims) — uploads once at setup
 * As output type: GPUResource.texture2D (no args) — declares shader output type
 */
export const GPUResource = {
  texture3D,
  /** Call with (data, dims) for resource upload, or use without args as output type token */
  texture2D: Object.assign(texture2DResource, texture2DToken) as {
    (data: ArrayBuffer, dims: { width: number; height: number }): ResourceHandle<'texture2d'>;
    readonly __outputType: 'texture2d';
  },
  storageBuffer,
};

/** Type guard: is this a ResourceHandle? */
export function isResourceHandle(v: any): v is ResourceHandle<any> {
  return v != null && typeof v === 'object' && '__resourceType' in v;
}

/** Type guard: is this a texture output token? */
export function isTextureOutputToken(v: any): v is OutputTypeToken<any> {
  return v != null && typeof v === 'object' && '__outputType' in v;
}
