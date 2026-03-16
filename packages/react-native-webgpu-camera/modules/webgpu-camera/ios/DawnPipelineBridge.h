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
                        resources:(nonnull NSArray<NSDictionary *> *)resources
                       passInputs:(nonnull NSArray<NSDictionary *> *)passInputs
               textureOutputPasses:(nonnull NSArray<NSNumber *> *)textureOutputPasses;

- (BOOL)processFrame:(nonnull CVPixelBufferRef)pixelBuffer;
- (void)cleanup;
- (void)installJSIBindings:(nonnull id)expoRuntime;

@end
