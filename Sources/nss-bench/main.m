// nss-bench: renders a moving, jittered test scene, upscales it with NSSMetal (or MetalFX, for comparison) and reports
// the upscale's GPU time; writes the last output frame as a PNG.
//
//   swift run -c release nss-bench --resources DIR [--quality low|mid|high] [--size 1488x928:2560x1600]
//                                  [--frames N] [--metalfx] [--game-like] [--png OUT.png]
//
// --resources: folder with nss_midlow.bin / nss_high.bin (tools/export_weights.py) and, optionally, nss.metallib
// (scripts/build-metallib.sh; without it the bundled kernel source is compiled at startup).
// --game-like: the formats a D3DMetal game hands MetalFX (RG11B10 color and output, Depth32Float_Stencil8 depth,
// motion in UV units, a 1x1 exposure texture).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import "NSSMetal/NSSMetal.h"

static const char *kSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct A { float2 jitter; float t; float2 outScale; float2 mvDiv; float depthFwd; };\n"
    "static float3 scene(float2 p, float t) {\n"
    "  float2 q = p + float2(3.0 * t, 0);\n"
    "  float v = (fract(q.x / 23.0) < 0.15 || fract(q.y / 17.0) < 0.15) ? 1.0 : 0.15;\n"
    "  float2 c = p - float2(400 + 200 * sin(t * 0.05), 300);\n"
    "  if (length(c) < 120) v = fract(atan2(c.y, c.x) * 6.0) < 0.5 ? 0.9 : 0.1;\n"
    "  return float3(v, v * 0.8, 0.3 + 0.5 * v);\n"
    "}\n"
    "kernel void render(constant A& a [[buffer(0)]], texture2d<float, access::write> col [[texture(0)]],\n"
    "                   texture2d<float, access::write> dep [[texture(1)]], texture2d<float, access::write> mv [[texture(2)]],\n"
    "                   uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x >= col.get_width() || g.y >= col.get_height()) return;\n"
    "  float2 p = (float2(g) + 0.5 + a.jitter) * a.outScale;\n"
    "  col.write(float4(scene(p, a.t), 1), g);\n"
    "  float2 c = p - float2(400 + 200 * sin(a.t * 0.05), 300);\n"
    "  bool disc = length(c) < 120;\n"
    "  float d = disc ? 0.6 : 0.2; dep.write(float4(a.depthFwd > 0 ? 1.0 - d : d), g);\n"
    "  float2 prevMinusCur = disc ? float2(200 * (sin((a.t - 1) * 0.05) - sin(a.t * 0.05)), 0) : float2(-3, 0);\n"
    "  mv.write(float4(prevMinusCur / a.outScale / a.mvDiv, 0, 0), g);\n"
    "}\n"
    "kernel void toRGBA(texture2d<float> s [[texture(0)]], texture2d<float, access::write> d [[texture(1)]],\n"
    "                   uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x < d.get_width() && g.y < d.get_height()) d.write(float4(s.read(g).rgb, 1), g);\n"
    "}\n";

static float halton(int i, int b) { float f = 1, r = 0; for (; i > 0; i /= b) f /= b, r += f * (i % b); return r; }

static void fail(NSString *msg) { fprintf(stderr, "%s\n", msg.UTF8String); exit(1); }

static void writePNG(id<MTLTexture> t, NSString *path) {
    const NSUInteger w = t.width, h = t.height;
    _Float16 *px = malloc(w * h * 8);
    [t getBytes:px bytesPerRow:w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    uint8_t *rgba = malloc(w * h * 4);
    for (size_t i = 0; i < w * h * 4; i++) rgba[i] = (uint8_t)lrintf(fminf(fmaxf((float)px[i], 0), 1) * 255);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(rgba, w, h, 8, w * 4, cs, kCGImageAlphaNoneSkipLast);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    CGImageDestinationRef dst = CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:path],
                                                                (__bridge CFStringRef)UTTypePNG.identifier, 1, nil);
    CGImageDestinationAddImage(dst, img, nil);
    CGImageDestinationFinalize(dst);
    CFRelease(dst), CGImageRelease(img), CGContextRelease(ctx), CGColorSpaceRelease(cs);
    free(px), free(rgba);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *resources = nil, *png = nil;
        int Wr = 1488, Hr = 928, Wo = 2560, Ho = 1600, frames = 600;
        NSSQuality quality = NSSQualityMid;
        BOOL useMetalFX = NO, gameLike = NO;
        for (int i = 1; i < argc; i++) {
            NSString *a = @(argv[i]);
#define next() (i + 1 < argc ? @(argv[++i]) : (fail([a stringByAppendingString:@" needs a value"]), @""))
            if ([a isEqual:@"--resources"]) resources = next();
            else if ([a isEqual:@"--png"]) png = next();
            else if ([a isEqual:@"--frames"]) frames = next().intValue;
            else if ([a isEqual:@"--size"]) { if (sscanf(next().UTF8String, "%dx%d:%dx%d", &Wr, &Hr, &Wo, &Ho) != 4) fail(@"--size WxH:WxH"); }
            else if ([a isEqual:@"--quality"]) {
                NSString *q = next();
                quality = [q isEqual:@"low"] ? NSSQualityLow : [q isEqual:@"high"] ? NSSQualityHigh : NSSQualityMid;
            }
            else if ([a isEqual:@"--metalfx"]) useMetalFX = YES;
            else if ([a isEqual:@"--game-like"]) gameLike = YES;
            else fail([@"unknown option " stringByAppendingString:a]);
#undef next
        }
        if (!useMetalFX && !resources) fail(@"usage: nss-bench --resources DIR [--quality low|mid|high] [--size WxH:WxH] [--frames N] [--metalfx] [--game-like] [--png OUT.png]");

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&e];
        if (!lib) fail(e.localizedDescription);
        id<MTLComputePipelineState> render = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"render"] error:&e];
        id<MTLComputePipelineState> toRGBA = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"toRGBA"] error:&e];

        const MTLPixelFormat colorFmt = gameLike ? MTLPixelFormatRG11B10Float : MTLPixelFormatRGBA16Float;
        const MTLPixelFormat depthFmt = gameLike ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
        MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:colorFmt width:Wr height:Hr mipmapped:NO];
        t.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, t.storageMode = MTLStorageModePrivate;
        id<MTLTexture> color = [dev newTextureWithDescriptor:t];
        t.pixelFormat = MTLPixelFormatR32Float;
        id<MTLTexture> depthW = [dev newTextureWithDescriptor:t];
        t.pixelFormat = depthFmt, t.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        id<MTLTexture> depth = [dev newTextureWithDescriptor:t];
        t.pixelFormat = MTLPixelFormatRG16Float, t.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> motion = [dev newTextureWithDescriptor:t];
        MTLTextureDescriptor *o = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:colorFmt width:Wo height:Ho mipmapped:NO];
        o.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget, o.storageMode = MTLStorageModePrivate;
        id<MTLTexture> out = [dev newTextureWithDescriptor:o];
        id<MTLBuffer> depthBuf = [dev newBufferWithLength:(size_t)Wr * Hr * 4 options:MTLResourceStorageModePrivate];
        id<MTLTexture> exposure = nil;
        if (gameLike) {
            MTLTextureDescriptor *ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:1 height:1 mipmapped:NO];
            exposure = [dev newTextureWithDescriptor:ed];
            const float one = 1.0f;
            [exposure replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:4];
        }

        NSSUpscaler *nss = nil;
        id<MTLFXTemporalScaler> fx = nil;
        CFTimeInterval t0 = CACurrentMediaTime();
        if (useMetalFX) {
            MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
            d.colorTextureFormat = colorFmt, d.depthTextureFormat = depthFmt, d.motionTextureFormat = MTLPixelFormatRG16Float;
            d.outputTextureFormat = colorFmt, d.inputWidth = Wr, d.inputHeight = Hr, d.outputWidth = Wo, d.outputHeight = Ho;
            fx = [d newTemporalScalerWithDevice:dev];
            if (!fx) fail(@"MetalFX temporal scaling isn't available");
        } else {
            [NSSUpscaler setLogHandler:^(NSString *m) { if (getenv("NSS_VERBOSE")) fprintf(stderr, "%s\n", m.UTF8String); }];
            NSSUpscalerDescriptor *d = [NSSUpscalerDescriptor new];
            d.inputWidth = Wr, d.inputHeight = Hr, d.outputWidth = Wo, d.outputHeight = Ho, d.quality = quality;
            d.resourceDirectory = [NSURL fileURLWithPath:resources];
            nss = [[NSSUpscaler alloc] initWithDevice:dev descriptor:d error:&e];
            if (!nss) fail(e.localizedDescription);
        }
        const double setupMs = (CACurrentMediaTime() - t0) * 1e3;

        NSMutableArray<NSNumber *> *ms = [NSMutableArray array];
        for (int f = 0; f < frames; f++) {
            const float jx = halton(f % 16 + 1, 2) - 0.5f, jy = halton(f % 16 + 1, 3) - 0.5f;
            struct { float jitter[2]; float t; float pad; float outScale[2]; float mvDiv[2]; float depthFwd; } a =   // float2 is 8-aligned
                {{jx, jy}, (float)f, 0, {(float)Wo / Wr, (float)Ho / Hr}, {gameLike ? (float)Wr : 1.0f, gameLike ? (float)Hr : 1.0f}, gameLike ? 1.0f : 0.0f};
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
            [c setComputePipelineState:render];
            [c setBytes:&a length:sizeof(a) atIndex:0];
            [c setTexture:color atIndex:0], [c setTexture:depthW atIndex:1], [c setTexture:motion atIndex:2];
            [c dispatchThreads:MTLSizeMake(Wr, Hr, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [c endEncoding];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];   // R32F -> the depth format, like a game's depth buffer
            [b copyFromTexture:depthW sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(Wr, Hr, 1)
                      toBuffer:depthBuf destinationOffset:0 destinationBytesPerRow:Wr * 4 destinationBytesPerImage:(NSUInteger)Wr * Hr * 4];
            [b copyFromBuffer:depthBuf sourceOffset:0 sourceBytesPerRow:Wr * 4 sourceBytesPerImage:(NSUInteger)Wr * Hr * 4
                   sourceSize:MTLSizeMake(Wr, Hr, 1) toTexture:depth destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)
                      options:gameLike ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
            [b endEncoding];
            [cb commit];

            // the upscale alone in its own command buffer, so its GPU time can be read
            id<MTLCommandBuffer> ub = [q commandBuffer];
            if (fx) {
                fx.colorTexture = color, fx.depthTexture = depth, fx.motionTexture = motion, fx.outputTexture = out;
                fx.jitterOffsetX = -jx, fx.jitterOffsetY = -jy;
                fx.motionVectorScaleX = gameLike ? Wr : 1, fx.motionVectorScaleY = gameLike ? Hr : 1;
                fx.depthReversed = !gameLike, fx.reset = f == 0;
                if (exposure) fx.exposureTexture = exposure;
                [fx encodeToCommandBuffer:ub];
            } else {
                nss.colorTexture = color, nss.depthTexture = depth, nss.motionTexture = motion, nss.outputTexture = out;
                nss.exposureTexture = exposure;
                nss.jitterOffsetX = -jx, nss.jitterOffsetY = -jy;
                nss.motionVectorScaleX = gameLike ? Wr : 1, nss.motionVectorScaleY = gameLike ? Hr : 1;
                nss.depthReversed = !gameLike, nss.reset = f == 0;
                [nss encodeToCommandBuffer:ub];
            }
            [ub commit];
            [ub waitUntilCompleted];
            if (ub.error) fail(ub.error.localizedDescription);
            if (f >= 30) [ms addObject:@((ub.GPUEndTime - ub.GPUStartTime) * 1e3)];
        }
        [ms sortUsingSelector:@selector(compare:)];
        printf("%s %dx%d -> %dx%d%s: setup %.0f ms, upscale GPU median %.3f ms, p95 %.3f ms (%lu frames)\n",
               fx ? "MetalFX" : quality == NSSQualityLow ? "NSS low" : quality == NSSQualityHigh ? "NSS high" : "NSS mid",
               Wr, Hr, Wo, Ho, gameLike ? ", game-like" : "", setupMs, ms.count ? ms[ms.count / 2].doubleValue : 0,
               ms.count ? ms[ms.count * 95 / 100].doubleValue : 0, (unsigned long)ms.count);

        if (png) {   // last output, as 8-bit sRGB-tagged PNG
            MTLTextureDescriptor *cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:Wo height:Ho mipmapped:NO];
            cd.usage = MTLTextureUsageShaderWrite, cd.storageMode = MTLStorageModeShared;
            id<MTLTexture> rgba = [dev newTextureWithDescriptor:cd];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
            [c setComputePipelineState:toRGBA];
            [c setTexture:out atIndex:0], [c setTexture:rgba atIndex:1];
            [c dispatchThreads:MTLSizeMake(Wo, Ho, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [c endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            writePNG(rgba, png);
            printf("wrote %s\n", png.UTF8String);
        }
    }
    return 0;
}
