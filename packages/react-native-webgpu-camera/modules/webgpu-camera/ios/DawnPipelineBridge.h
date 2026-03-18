#pragma once

#import <Foundation/Foundation.h>
#import <CoreVideo/CVPixelBuffer.h>

@interface DawnPipelineBridge : NSObject

- (BOOL)setupMultiPassWithShaders:(nonnull NSArray<NSString *> *)shaders
                            width:(int)width
                           height:(int)height
                      bufferSpecs:(nonnull NSArray<NSArray<NSNumber *> *> *)bufferSpecs
                        useCanvas:(BOOL)useCanvas
                             sync:(BOOL)sync
                         appleLog:(BOOL)appleLog
                         useDepth:(BOOL)useDepth
                         lidarYUV:(BOOL)lidarYUV
                        resources:(nonnull NSArray<NSDictionary *> *)resources
                       passInputs:(nonnull NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(nonnull NSArray<NSNumber *> *)textureOutputPasses
                       modelSpecs:(nonnull NSArray<NSDictionary *> *)modelSpecs;

- (BOOL)processFrame:(nonnull CVPixelBufferRef)pixelBuffer;
- (BOOL)processFrame:(nonnull CVPixelBufferRef)pixelBuffer
         depthBuffer:(nullable CVPixelBufferRef)depthBuffer;
- (void)cleanup;
- (void)installJSIBindings:(nonnull id)expoRuntime;

/// Returns Dawn device/instance/proc-table pointers as decimal strings for ONNX Runtime WebGPU EP.
+ (nonnull NSDictionary<NSString *, NSString *> *)getDawnPointers;

@end
