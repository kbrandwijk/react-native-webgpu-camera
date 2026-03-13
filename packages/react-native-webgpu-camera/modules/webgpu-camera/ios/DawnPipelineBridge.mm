#import "DawnPipelineBridge.h"
#import "DawnComputePipeline.h"

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
    _pipeline = nil;
  }
}

- (BOOL)setupWithWGSL:(NSString *)wgslCode width:(int)width height:(int)height {
  if (!_pipeline) return NO;
  return dawn_pipeline_setup(_pipeline, [wgslCode UTF8String], width, height);
}

- (BOOL)processFrame:(CVPixelBufferRef)pixelBuffer {
  if (!_pipeline) return NO;
  return dawn_pipeline_process_frame(_pipeline, pixelBuffer);
}

- (void)cleanup {
  if (_pipeline) {
    dawn_pipeline_cleanup(_pipeline);
  }
}

- (void)installJSIBindings:(id)expoRuntime {
  if (!_pipeline) {
    NSLog(@"[DawnPipelineBridge] Cannot install JSI — pipeline not created");
    return;
  }

  // EXJavaScriptRuntime has a - (jsi::Runtime *)get method
  EXJavaScriptRuntime *runtime = (EXJavaScriptRuntime *)expoRuntime;
  jsi::Runtime *jsiRuntime = [runtime get];

  if (!jsiRuntime) {
    NSLog(@"[DawnPipelineBridge] Cannot install JSI — no jsi::Runtime available");
    return;
  }

  dawn_pipeline_install_jsi(_pipeline, jsiRuntime);
}

@end
