/** Type tag for resource handles */
export interface ResourceHandle<T extends string> {
  readonly __resourceType: T;
  readonly __handle: number;
  readonly __data?: ArrayBuffer;
  readonly __fileUri?: string;
  readonly __dims?: { width: number; height: number; depth?: number; format?: string };
}

/** Sentinel for output type tokens */
export interface OutputTypeToken<T extends string> {
  readonly __outputType: T;
}

function texture3D(
  dataOrFileUri: ArrayBuffer | string,
  dims: { width: number; height: number; depth: number; format?: 'rgba8unorm' | 'rgba32float' },
): ResourceHandle<'texture3d'> {
  if (typeof dataOrFileUri === 'string') {
    return {
      __resourceType: 'texture3d',
      __handle: -1,
      __fileUri: dataOrFileUri,
      __dims: dims,
    };
  }
  return {
    __resourceType: 'texture3d',
    __handle: -1,
    __data: dataOrFileUri,
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

function cameraDepth(): ResourceHandle<'cameraDepth'> {
  return {
    __resourceType: 'cameraDepth',
    __handle: -1,
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

export interface ModelOptions {
  /** Model input shape, e.g. [1, 3, 518, 518]. Inferred from model if omitted. */
  inputShape?: number[];
  /** ImageNet normalization params. Default: ImageNet standard. */
  normalization?: { mean: [number, number, number]; std: [number, number, number] };
  /** When true, inference blocks the pipeline (for small models). Default: false (async). */
  sync?: boolean;
}

/** Model-specific resource handle — extends ResourceHandle with model options */
export interface ModelResourceHandle extends ResourceHandle<'model'> {
  readonly __modelOptions?: ModelOptions;
}

function model(
  pathOrUrl: string,
  options?: ModelOptions,
): ModelResourceHandle {
  return {
    __resourceType: 'model',
    __handle: -1,
    __fileUri: pathOrUrl,
    __modelOptions: options,
  };
}

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
  cameraDepth,
  model,
};

/** Type guard: is this a ResourceHandle? */
export function isResourceHandle(v: any): v is ResourceHandle<any> {
  return v != null && typeof v === 'object' && '__resourceType' in v;
}

/** Type guard: is this a texture output token? */
export function isTextureOutputToken(v: any): v is OutputTypeToken<any> {
  return v != null && typeof v === 'object' && '__outputType' in v;
}
