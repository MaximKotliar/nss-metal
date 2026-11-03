// NSSMetal: Arm Neural Super Sampling v1 (NSS) as a Metal temporal upscaler.
//
// Shaped like MetalFX's temporal scaler: set the frame's textures and parameters on the upscaler, then encode it into
// your command buffer (Metal 3 or Metal 4). Conventions follow MetalFX, so code that drives an MTLFXTemporalScaler can
// drive this the same way.
//
//   NSSUpscalerDescriptor *d = [NSSUpscalerDescriptor new];
//   d.inputWidth = 1488, d.inputHeight = 928, d.outputWidth = 2560, d.outputHeight = 1600;
//   d.resourceDirectory = [NSURL fileURLWithPath:@"…/nss"];   // nss.metallib, nss_midlow.bin, nss_high.bin
//   NSSUpscaler *up = [[NSSUpscaler alloc] initWithDevice:device descriptor:d error:&error];
//   …each frame:
//   up.colorTexture = color; up.depthTexture = depth; up.motionTexture = motion; up.outputTexture = output;
//   up.jitterOffsetX = jx; up.jitterOffsetY = jy; up.motionVectorScaleX = w; up.motionVectorScaleY = h;
//   [up encodeToCommandBuffer:commandBuffer];
//
// Setup compiles and times kernel variants for this GPU and size (about 0.15–0.3 s; slower the first time, while
// Metal's shader cache is cold). Create the upscaler once per size and quality, not per frame.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, NSSQuality) {
    /// Half-resolution network, bilinear history. Cheapest.
    NSSQualityLow = 0,
    /// Half-resolution network, Catmull-Rom history. About the GPU cost of MetalFX; the usual choice.
    NSSQualityMid = 1,
    /// Full-resolution network, 9-tap filter. About 3x the cost of mid.
    NSSQualityHigh = 2,
};

@interface NSSUpscalerDescriptor : NSObject <NSCopying>
/// Render size: the region at the top-left of the input textures that holds the frame.
@property (nonatomic) NSUInteger inputWidth, inputHeight;
@property (nonatomic) NSUInteger outputWidth, outputHeight;
@property (nonatomic) NSSQuality quality;   // default NSSQualityMid
/// Folder with nss_midlow.bin (low, mid) or nss_high.bin (high), made by tools/export_weights.py, and
/// nss.metallib (scripts/build-metallib.sh) unless `library` is set.
@property (nonatomic, copy, nullable) NSURL *resourceDirectory;
/// The NSS kernels, if you build them into your own library. Default: resourceDirectory/nss.metallib.
@property (nonatomic, strong, nullable) id<MTLLibrary> library;
/// View-space depth: near and far planes and vertical field of view (radians). NSS uses them to compare depths;
/// rough values work. Defaults 0.1, 1000, 1.0.
@property (nonatomic) float zNear, zFar, verticalFieldOfView;
@end

@interface NSSUpscaler : NSObject
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(NSSUpscalerDescriptor *)descriptor
                                  error:(NSError *_Nullable *_Nullable)error NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly) NSUInteger inputWidth, inputHeight, outputWidth, outputHeight;
@property (nonatomic, readonly) NSSQuality quality;
/// Size of the network's input (half the render size for low and mid, rounded up to 8).
@property (nonatomic, readonly) NSUInteger networkWidth, networkHeight;

// Per-frame inputs, as for MTLFXTemporalScaler.
/// Linear HDR (or SDR) color at render size. Any float or unorm format readable by a shader.
@property (nonatomic, strong, nullable) id<MTLTexture> colorTexture;
/// Depth at render size: a depth format (Depth32Float, Depth32Float_Stencil8, …) or a float color format.
@property (nonatomic, strong, nullable) id<MTLTexture> depthTexture;
/// Motion at render size, RG: texel * motionVectorScale = (previous - current) position in input pixels.
@property (nonatomic, strong, nullable) id<MTLTexture> motionTexture;
/// Optional 1x1 exposure (R): color is multiplied by exposure / preExposure before upscaling and divided after.
@property (nonatomic, strong, nullable) id<MTLTexture> exposureTexture;
/// Output at output size; needs MTLTextureUsageShaderWrite.
@property (nonatomic, strong, nullable) id<MTLTexture> outputTexture;
/// The sub-pixel jitter of this frame's projection, in input pixels, with MetalFX's sign.
@property (nonatomic) float jitterOffsetX, jitterOffsetY;
/// Scales motion texels to input pixels (for UV-space motion: the render size).
@property (nonatomic) float motionVectorScaleX, motionVectorScaleY;
@property (nonatomic) float preExposure;     // default 1
@property (nonatomic, getter=isDepthReversed) BOOL depthReversed;
/// Discard history (camera cut, first frame after a pause). Cleared by the next encode.
@property (nonatomic) BOOL reset;

/// Encodes the upscale into a Metal 3 command buffer (its own compute encoder).
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
/// Encodes into a Metal 4 command buffer. Waits for `fence` before reading the inputs and updates it after writing
/// the output, if given. The upscaler adds its resources to residency sets it attaches to the command buffer; your
/// textures are added to one of them as they appear.
- (void)encodeToMTL4CommandBuffer:(id<MTL4CommandBuffer>)commandBuffer fence:(nullable id<MTLFence>)fence
    API_AVAILABLE(macos(26.0), ios(26.0));

/// Diagnostics from setup and from the NSS_* debug switches (nil: silent). Global; set it before creating upscalers.
+ (void)setLogHandler:(nullable void (^)(NSString *message))handler;
@end

NS_ASSUME_NONNULL_END
