#pragma once

#import <Foundation/Foundation.h>
#import <CoreVideo/CVPixelBuffer.h>

/// Obj-C bridge to the C++ DawnComputePipeline.
/// Swift can call these methods; they forward to the C interface internally.
@interface DawnPipelineBridge : NSObject

- (BOOL)setupWithWGSL:(nonnull NSString *)wgslCode width:(int)width height:(int)height;
- (BOOL)processFrame:(CVPixelBufferRef _Nonnull)pixelBuffer;
- (void)cleanup;

/// Install JSI bindings using the given Expo runtime.
/// Must be called on the JS thread after setup.
- (void)installJSIBindings:(nonnull id)expoRuntime;

@end
