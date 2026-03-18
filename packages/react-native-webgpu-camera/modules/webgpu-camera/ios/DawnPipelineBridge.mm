#import "DawnPipelineBridge.h"
#include "DawnComputePipeline.h"
#import <ExpoModulesJSI/EXJavaScriptRuntime.h>

@implementation DawnPipelineBridge {
  DawnComputePipelineRef _pipeline;
  int _frameCount;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _pipeline = dawn_pipeline_create();
  }
  return self;
}

- (void)dealloc {
  if (_pipeline) {
    dawn_pipeline_destroy(_pipeline);
    _pipeline = nullptr;
  }
}

- (BOOL)setupMultiPassWithShaders:(NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                         appleLog:(BOOL)appleLog
                         useDepth:(BOOL)useDepth
                         lidarYUV:(BOOL)lidarYUV
                        resources:(NSArray<NSDictionary *> *)resources
                       passInputs:(NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(NSArray<NSNumber *> *)textureOutputPasses {
  if (!_pipeline) return NO;

  // Convert NSArray<NSString*> to C string array
  int shaderCount = (int)shaders.count;
  std::vector<const char*> cShaders(shaderCount);
  std::vector<std::string> shaderStorage(shaderCount); // keep strings alive
  for (int i = 0; i < shaderCount; i++) {
    shaderStorage[i] = [shaders[i] UTF8String];
    cShaders[i] = shaderStorage[i].c_str();
  }

  // Convert buffer specs to flat int array [passIndex, elementSize, count, ...]
  int bufferCount = (int)bufferSpecs.count;
  std::vector<int> flatSpecs(bufferCount * 3);
  for (int i = 0; i < bufferCount; i++) {
    NSArray<NSNumber *> *spec = bufferSpecs[i];
    flatSpecs[i * 3 + 0] = [spec[0] intValue]; // passIndex
    flatSpecs[i * 3 + 1] = [spec[1] intValue]; // elementSize
    flatSpecs[i * 3 + 2] = [spec[2] intValue]; // count
  }

  // Convert resources to C++ structs
  std::vector<dawn_pipeline::ResourceSpec> resourceSpecs;
  for (NSDictionary *res in resources) {
    dawn_pipeline::ResourceSpec rs;
    NSString *type = res[@"type"];
    if ([type isEqualToString:@"texture3d"]) {
      rs.type = dawn_pipeline::ResourceType::Texture3D;
    } else if ([type isEqualToString:@"texture2d"]) {
      rs.type = dawn_pipeline::ResourceType::Texture2D;
    } else if ([type isEqualToString:@"cameraDepth"]) {
      rs.type = dawn_pipeline::ResourceType::CameraDepth;
    } else {
      rs.type = dawn_pipeline::ResourceType::StorageBuffer;
    }
    rs.width = [res[@"width"] intValue];
    rs.height = [res[@"height"] intValue];
    rs.depth = [res[@"depth"] intValue];
    NSString *format = res[@"format"];
    if ([format isEqualToString:@"rgba32float"]) {
      rs.format = dawn_pipeline::ResourceFormat::RGBA32Float;
    } else {
      rs.format = dawn_pipeline::ResourceFormat::RGBA8Unorm;
    }
    // File URI path (native side will load and parse)
    NSString *fileUri = res[@"fileUri"];
    if (fileUri) {
      // Strip file:// prefix if present
      NSString *path = fileUri;
      if ([path hasPrefix:@"file://"]) {
        path = [path substringFromIndex:7];
      }
      rs.fileUri = [path UTF8String];
    }
    // Copy data to owned buffer (NSData may be released after this scope)
    NSData *data = res[@"data"];
    if (data) {
      const uint8_t *bytes = (const uint8_t *)data.bytes;
      rs.data.assign(bytes, bytes + data.length);
    }
    resourceSpecs.push_back(rs);
  }

  // Convert passInputs to C++ structs
  std::vector<dawn_pipeline::PassInputSpec> passInputSpecs;
  for (NSDictionary *pi in passInputs) {
    dawn_pipeline::PassInputSpec pis;
    pis.passIndex = [pi[@"passIndex"] intValue];
    NSArray<NSDictionary *> *bindings = pi[@"bindings"];
    for (NSDictionary *b in bindings) {
      dawn_pipeline::InputBinding ib;
      ib.bindingIndex = [b[@"index"] intValue];
      NSString *btype = b[@"type"];
      if ([btype isEqualToString:@"texture3d"]) {
        ib.type = dawn_pipeline::InputBindingType::Texture3D;
      } else if ([btype isEqualToString:@"texture2d"]) {
        ib.type = dawn_pipeline::InputBindingType::Texture2D;
      } else if ([btype isEqualToString:@"sampler"]) {
        ib.type = dawn_pipeline::InputBindingType::Sampler;
      } else {
        ib.type = dawn_pipeline::InputBindingType::StorageBufferRead;
      }
      ib.resourceHandle = b[@"resourceHandle"] ? [b[@"resourceHandle"] intValue] : -1;
      ib.sourcePass = b[@"sourcePass"] ? [b[@"sourcePass"] intValue] : -1;
      ib.sourceBuffer = b[@"sourceBuffer"] ? [b[@"sourceBuffer"] intValue] : -1;
      pis.bindings.push_back(ib);
    }
    passInputSpecs.push_back(pis);
  }

  // Convert textureOutputPasses
  std::vector<int> texOutPasses;
  for (NSNumber *n in textureOutputPasses) {
    texOutPasses.push_back([n intValue]);
  }

  return dawn_pipeline_setup_multipass(
    _pipeline,
    cShaders.data(), shaderCount,
    width, height,
    flatSpecs.data(), bufferCount,
    useCanvas, sync, (bool)appleLog, (bool)useDepth, (bool)lidarYUV,
    resourceSpecs.data(), (int)resourceSpecs.size(),
    passInputSpecs.data(), (int)passInputSpecs.size(),
    texOutPasses.data(), (int)texOutPasses.size()
  );
}

- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer {
  return [self processFrame:pixelBuffer depthBuffer:nil];
}

- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer depthBuffer:(CVPixelBufferRef)depthBuffer {
  if (!_pipeline) return NO;
  _frameCount++;
  // Log first 5 frames with depth status (ObjC NSLog — not privacy-redacted)
  if (_frameCount <= 5) {
    OSType vfmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    NSLog(@"[DawnBridge] frame #%d, hasDepth=%d, videoFmt=0x%08x (%c%c%c%c)",
          _frameCount, depthBuffer != nil ? 1 : 0,
          (unsigned)vfmt,
          (char)((vfmt >> 24) & 0xFF), (char)((vfmt >> 16) & 0xFF),
          (char)((vfmt >> 8) & 0xFF), (char)(vfmt & 0xFF));
    if (depthBuffer && _frameCount == 1) {
      OSType depthFmt = CVPixelBufferGetPixelFormatType(depthBuffer);
      size_t dw = CVPixelBufferGetWidth(depthBuffer);
      size_t dh = CVPixelBufferGetHeight(depthBuffer);
      size_t dPlanes = CVPixelBufferGetPlaneCount(depthBuffer);
      size_t dBpr = CVPixelBufferGetBytesPerRowOfPlane(depthBuffer, 0);
      IOSurfaceRef dSurface = CVPixelBufferGetIOSurface(depthBuffer);
      NSLog(@"[DawnBridge] depth fmt=0x%08x (%c%c%c%c), %zux%zu, %zu planes, bpr=%zu, ioSurface=%s",
            (unsigned)depthFmt,
            (char)((depthFmt >> 24) & 0xFF), (char)((depthFmt >> 16) & 0xFF),
            (char)((depthFmt >> 8) & 0xFF), (char)(depthFmt & 0xFF),
            dw, dh, dPlanes, dBpr, dSurface ? "YES" : "NO");
    }
  }
  static bool loggedFormat = false;
  if (!loggedFormat) {
    OSType fmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    size_t w = CVPixelBufferGetWidth(pixelBuffer);
    size_t h = CVPixelBufferGetHeight(pixelBuffer);
    size_t planes = CVPixelBufferGetPlaneCount(pixelBuffer);
    NSLog(@"[DawnBridge] First frame pixel format: 0x%08x (%c%c%c%c), %zux%zu, %zu planes",
          (unsigned)fmt,
          (char)((fmt >> 24) & 0xFF), (char)((fmt >> 16) & 0xFF),
          (char)((fmt >> 8) & 0xFF), (char)(fmt & 0xFF),
          w, h, planes);
    if (depthBuffer) {
      OSType depthFmt = CVPixelBufferGetPixelFormatType(depthBuffer);
      size_t dw = CVPixelBufferGetWidth(depthBuffer);
      size_t dh = CVPixelBufferGetHeight(depthBuffer);
      NSLog(@"[DawnBridge] First frame depth format: 0x%08x, %zux%zu",
            (unsigned)depthFmt, dw, dh);
    }
    loggedFormat = true;
  }
  return dawn_pipeline_process_frame_with_depth(_pipeline, pixelBuffer, depthBuffer);
}

- (void)cleanup {
  if (!_pipeline) return;
  dawn_pipeline_cleanup(_pipeline);
}

- (void)installJSIBindings:(id)expoRuntime {
  if (!_pipeline) return;
  EXJavaScriptRuntime *runtime = (EXJavaScriptRuntime *)expoRuntime;
  facebook::jsi::Runtime *jsiRuntime = [runtime get];
  if (!jsiRuntime) return;
  dawn_pipeline_install_jsi(_pipeline, jsiRuntime);
}

@end
