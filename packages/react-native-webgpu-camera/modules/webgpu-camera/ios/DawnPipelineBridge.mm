#import "DawnPipelineBridge.h"
#include "DawnComputePipeline.h"
#import <ExpoModulesJSI/EXJavaScriptRuntime.h>

@implementation DawnPipelineBridge {
  DawnComputePipelineRef _pipeline;
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
                             sync:(BOOL)sync {
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

  return dawn_pipeline_setup_multipass(
    _pipeline,
    cShaders.data(), shaderCount,
    width, height,
    flatSpecs.data(), bufferCount,
    useCanvas, sync
  );
}

- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer {
  if (!_pipeline) return NO;
  return dawn_pipeline_process_frame(_pipeline, pixelBuffer);
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
