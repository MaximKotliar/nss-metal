// NSSMetal: Arm Neural Super Sampling v1 on Metal.
//
// Ports the NSS pass schedule, offset LUT and pre/post-processing from Arm's Apache-2.0 reference
// (arm/neural-graphics-model-gym, usecases/nss); the 14-layer network runs as simdgroup-matrix convolutions.
// The network weights are Arm's (Hugging Face, Arm AI Model Community License) and are not part of this
// project: tools/export_weights.py turns their release into the files loaded here.
//
// Public API: include/NSSMetal/NSSMetal.h; everything else here is internal. Debug switches are environment
// variables NSS_* (README.md lists them).

#import "NSSMetal/NSSMetal.h"
#import <QuartzCore/QuartzCore.h>
#import <stdatomic.h>

static void (^nssLogHandler)(NSString *);
static void nss_log(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);
static void nss_log(NSString *fmt, ...) {
    void (^h)(NSString *) = nssLogHandler;
    if (!h) return;
    va_list ap;
    va_start(ap, fmt);
    NSString *s = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);
    h(s);
}

// NSS_<name>, or LSFG_NSS_<name> (their name when this lived inside the LSFG hook)
static const char *nssEnv(const char *name) {
    char buf[96];
    snprintf(buf, sizeof buf, "NSS_%s", name);
    const char *v = getenv(buf);
    if (v) return v;
    snprintf(buf, sizeof buf, "LSFG_NSS_%s", name);
    return getenv(buf);
}

#pragma mark - Engine

typedef struct { __unsafe_unretained id<MTLBuffer> b; int H, W, C, wpad; } NssT;   // channels-last half, zero border
typedef struct { int ic, oc, icp, ocp; size_t wOff, bOff; } NssLayer;

static int nss_gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

@interface NssEngine : NSObject {
@public
    int Hr, Wr, Ho, Wo, Hp, Wp, Hq, Wq, Hd, Wd, quality, kpnCh, taps, idxModH, idxModW, p, frame;
    BOOL half_;
    float d2v[4];
    id<MTLDevice> dev;
    id<MTLComputePipelineState> prepQ, prepQD, disLqS, gprep, gprepD, dclear, dscatter, dscatterQ, disLq, pre, preHalf, post;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *conv, *upconv;
    id<MTLBuffer> depthTm1Alt, blockStats;   // fused mid/low prep (nss_prep_q): second scatter target, per-block stats
    BOOL fusedPrep;
    BOOL packConv0;    // conv0 with packed horizontal taps (nss_conv_mma, fcPackKx), weights at pack0Off
    size_t pack0Off;
    id<MTLBuffer> dilMotion;   // mid/low: motion at each render pixel's nearest-depth neighbour (nss_pre_half -> nss_post)
    id<MTLBuffer> weights, biases, depth, motion, depthTm1, dis, disLqB, nearest, kBuf, deriv[2], lut, lutMap;
    id<MTLComputePipelineState> lutClear, lutScatter, lutSelect;
    // offset LUTs kept per jitter value (DLSS cycles through a fixed set): slot -> built once
    id<MTLBuffer> lutCache;
    size_t lutBytes;
    int lutSlots, lutUsed;
    NSMutableDictionary<NSNumber *, NSNumber *> *lutKeys;   // jitter bits -> slot
    int64_t lutLastUse[64];
    float lutWgt[36];
    NSMutableArray<id<MTLBuffer>> *tensors;   // keeps the NssT buffers alive
    NssT IN, A0, CAT10, A2, CAT8, A4, A5, A6, A8, KPN, A10, A11, TEMP[2];
    id<MTLTexture> tempView[2];
    id<MTLTexture> derivTex[2];   // linear textures over deriv[], so the derivative reads filter in hardware   // RGBA16F views over TEMP, so the temporal/feedback reads filter in hardware
    NssLayer layers[14];
    int layerPX[14];
    int layerSplit[14];                   // output-channel groups (threadgroup z), picked with layerPX                      // pixel tiles per simdgroup, picked by timing at setup
    id<MTLComputePipelineState> layerPSO[14];   // that variant compiled with the layer's shape as function constants
    id<MTLLibrary> lib;
    id<MTLTexture> history[2], dummy;
    // Metal 4: argument table, constant uploads, and residency for our resources (and the game's, as seen)
    id<MTL4ArgumentTable> table;
    id<MTLBuffer> upload[3];
    int uploadSlot;
    id<MTLResidencySet> residency, gameResidency;
    id profHeap;                          // id<MTL4CounterHeap>
    NSArray<NSString *> *profPending;     // names of the profiled frame not read back yet
    NSMutableDictionary<NSString *, NSMutableArray<NSNumber *> *> *profAcc;
    int profFrames;
}
@end

// Profiling (NSS_PROFILE=N, Metal 4 only): every Nth frame, a precise GPU timestamp after each dispatch; the
// breakdown by pass is logged once that frame is done. Pipelines are named by their kernel.
static NSMapTable<id, NSString *> *psoNames;

// Encodes the pass list into a Metal 3 compute encoder, or into a Metal 4 one through an argument table
// (constants go through an upload buffer; a barrier before each dispatch keeps the passes in order, like a
// Metal 3 serial encoder).
@interface NssEnc : NSObject {
@public
    id<MTLComputeCommandEncoder> m3;
    id<MTL4ComputeCommandEncoder> m4;
    id<MTL4ArgumentTable> table;
    id<MTLBuffer> upload;
    NSUInteger uploadOff;
    BOOL dispatched;
    BOOL overlapNext;   // next dispatch doesn't depend on the previous one: no barrier (Metal 4)
    id profHeap;                          // id<MTL4CounterHeap>, set on profiled frames
    NSMutableArray<NSString *> *profNames;
    id<MTLComputePipelineState> current;
}
@end
@implementation NssEnc
- (void)setComputePipelineState:(id<MTLComputePipelineState>)s {
    current = s;
    if (m4) [m4 setComputePipelineState:s]; else [m3 setComputePipelineState:s];
}
- (void)stamp:(NSString *)name {
    if (!profHeap || profNames.count >= 63) return;
    if (@available(macOS 26.0, *)) {
        [m4 writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:profHeap atIndex:profNames.count];
        [profNames addObject:name];
    }
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)n atIndex:(NSUInteger)i {
    if (!m4) { [m3 setBytes:bytes length:n atIndex:i]; return; }
    uploadOff = (uploadOff + 255) & ~(NSUInteger)255;
    if (uploadOff + n > upload.length) { nss_log(@"nss: upload buffer full"); return; }
    memcpy((char *)upload.contents + uploadOff, bytes, n);
    [table setAddress:upload.gpuAddress + uploadOff atIndex:i];
    uploadOff += n;
}
- (void)setBuffer:(id<MTLBuffer>)b offset:(NSUInteger)o atIndex:(NSUInteger)i {
    if (m4) [table setAddress:b.gpuAddress + o atIndex:i]; else [m3 setBuffer:b offset:o atIndex:i];
}
- (void)setTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i {
    if (m4) [table setTexture:t.gpuResourceID atIndex:i]; else [m3 setTexture:t atIndex:i];
}
- (void)setThreadgroupMemoryLength:(NSUInteger)n atIndex:(NSUInteger)i {
    if (m4) [m4 setThreadgroupMemoryLength:n atIndex:i]; else [m3 setThreadgroupMemoryLength:n atIndex:i];
}
- (void)order {
    if (overlapNext) { overlapNext = NO; dispatched = YES; return; }
    if (m4 && dispatched)
        [m4 barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch
                    visibilityOptions:MTL4VisibilityOptionDevice];
    dispatched = YES;
}
- (void)dispatchThreads:(MTLSize)g threadsPerThreadgroup:(MTLSize)t {
    [self order];
    if (m4) [m4 dispatchThreads:g threadsPerThreadgroup:t]; else [m3 dispatchThreads:g threadsPerThreadgroup:t];
    [self stamp:[psoNames objectForKey:current] ?: @"?"];
}
- (void)dispatchThreadgroups:(MTLSize)g threadsPerThreadgroup:(MTLSize)t {
    [self order];
    if (m4) [m4 dispatchThreadgroups:g threadsPerThreadgroup:t]; else [m3 dispatchThreadgroups:g threadsPerThreadgroup:t];
    [self stamp:[psoNames objectForKey:current] ?: @"?"];
}
@end
@implementation NssEngine
@end


static NssT nssTensor(NssEngine *e, int H, int W, int C) {
    NssT t = {nil, H, W, C, (W + 15) / 16 * 16 + 16};
    id<MTLBuffer> b = [e->dev newBufferWithLength:(size_t)(H + 4) * t.wpad * C * 2 options:MTLResourceStorageModePrivate];
    [e->tensors addObject:b];
    t.b = b;
    return t;
}

static void nssTune(NssEngine *e, id<MTLCommandQueue> q);

// quality: 0 low, 1 mid, 2 high. Returns nil (and logs) when something is missing.
static NssEngine *nssCreate(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib, NSString *weightsDir,
                            int wr, int hr, int wo, int ho, int quality, float zNear, float zFar, float fovY) {
    NssEngine *e = [NssEngine new];
    e->dev = dev, e->Wr = wr, e->Hr = hr, e->Wo = wo, e->Ho = ho, e->quality = quality;
    e->half_ = quality < 2;
    e->kpnCh = e->half_ ? 16 : 36, e->taps = e->half_ ? 4 : 9;
    e->Hq = e->half_ ? hr / 2 : hr, e->Wq = e->half_ ? wr / 2 : wr;
    e->Hp = (e->Hq + 7) / 8 * 8, e->Wp = (e->Wq + 7) / 8 * 8;
    e->Hd = e->half_ ? hr / 4 : hr / 2, e->Wd = e->half_ ? wr / 4 : wr / 2;
    e->tensors = [NSMutableArray array];

    e->lib = lib;
    id<MTLComputePipelineState> (^pso)(NSString *) = ^id<MTLComputePipelineState>(NSString *n) {
        NSError *pe = nil;
        // through the specialization API: kernels that declare function constants need it even with none set
        id<MTLFunction> f = [lib newFunctionWithName:n constantValues:[MTLFunctionConstantValues new] error:&pe];
        id<MTLComputePipelineState> s = f ? [dev newComputePipelineStateWithFunction:f error:&pe] : nil;
        if (!s) nss_log(@"nss: pipeline %@: %@", n, pe.localizedDescription ?: @"missing");
        @synchronized(NSNull.null) {
            if (!psoNames) psoNames = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsObjectPointerPersonality valueOptions:NSPointerFunctionsStrongMemory];
            if (s) [psoNames setObject:[n stringByReplacingOccurrencesOfString:@"nss_" withString:@""] forKey:s];
        }
        return s;
    };
    e->gprep = pso(@"nss_game_prep"), e->gprepD = pso(@"nss_game_prep_d");
    e->prepQ = pso(@"nss_prep_q"), e->prepQD = pso(@"nss_prep_q_d"), e->disLqS = pso(@"nss_disocclusion_lq_s");
    e->dclear = pso(@"nss_depth_clear"), e->dscatter = pso(@"nss_depth_scatter"), e->dscatterQ = pso(@"nss_depth_scatter_q");
    e->disLq = pso(@"nss_disocclusion_lq"), e->pre = pso(@"nss_pre"), e->preHalf = pso(@"nss_pre_half"), e->post = pso(@"nss_post");
    e->conv = [NSMutableDictionary dictionary], e->upconv = [NSMutableDictionary dictionary];
    for (NSNumber *oc in @[@8, @16, @32, @40, @64]) {
        for (int px = 1; px <= 4; px *= 2) {
            e->conv[@(oc.intValue * 10 + px)] = pso([NSString stringWithFormat:@"nss_conv_mma_%@_p%d", oc, px]);
            e->upconv[@(oc.intValue * 10 + px)] = pso([NSString stringWithFormat:@"nss_upconv_mma_%@_p%d", oc, px]);
        }
    }
    if (!e->gprep || !e->gprepD || !e->post || !e->preHalf || !e->pre || !e->disLq || !e->dscatter || !e->dscatterQ)
        return nil;
    for (NSNumber *oc in @[@8, @16, @32, @40, @64])
        if (!e->conv[@(oc.intValue * 10 + 1)] || !e->conv[@(oc.intValue * 10 + 2)] || !e->conv[@(oc.intValue * 10 + 4)] || !e->upconv[@(oc.intValue * 10 + 1)] || !e->upconv[@(oc.intValue * 10 + 2)] || !e->upconv[@(oc.intValue * 10 + 4)]) return nil;

    // weights: [oc][ic][3][3] + bias[oc] per layer (float32), repacked to half [tap][icp][ocp] (or the four 2x2
    // phase kernels for the layers that read a 2x nearest upsample) + float bias[ocp]
    NSString *wPath = [weightsDir stringByAppendingPathComponent:e->half_ ? @"nss_midlow.bin" : @"nss_high.bin"];
    NSData *w = [NSData dataWithContentsOfFile:wPath];
    if (!w) { nss_log(@"nss: cannot read %@", wPath); return nil; }
    const float *src = (const float *)w.bytes;
    const int shape[14][2] = {{12, 32}, {32, 32}, {32, 32}, {32, 32}, {32, 64}, {64, 64}, {64, 32},
                              {32, 16}, {48, 32}, {32, e->kpnCh}, {32, 16}, {48, 16}, {16, 16}, {16, 4}};
    size_t total = 0, btotal = 0;
    for (int li = 0; li < 14; li++) {
        const int icp = (shape[li][0] + 7) / 8 * 8, ocp = (shape[li][1] + 7) / 8 * 8;
        total += (size_t)((li == 7 || li == 10 || li == 13) ? 16 : 9) * icp * ocp, btotal += ocp;
    }
    // conv0 again with packed taps: [ky][kx * 12 + ic, rounded up to 40][oc]
    e->pack0Off = total;
    total += (size_t)3 * 40 * 32;
    _Float16 *packed = calloc(total, sizeof(_Float16));
    float *bias = calloc(btotal, sizeof(float));
    size_t so = 0, wo_ = 0, bo = 0;
    for (int li = 0; li < 14; li++) {
        NssLayer l = {shape[li][0], shape[li][1], (shape[li][0] + 7) / 8 * 8, (shape[li][1] + 7) / 8 * 8, wo_, bo};
        const BOOL upc = li == 7 || li == 10 || li == 13;
#define NSS_W(o, i, ky, kx) src[so + ((size_t)(o) * l.ic + (i)) * 9 + (ky) * 3 + (kx)]
        if (!upc) {
            for (int o = 0; o < l.oc; o++)
                for (int i = 0; i < l.ic; i++)
                    for (int t = 0; t < 9; t++) packed[l.wOff + ((size_t)t * l.icp + i) * l.ocp + o] = (_Float16)NSS_W(o, i, t / 3, t % 3);
            wo_ += (size_t)9 * l.icp * l.ocp;
            if (li == 0)
                for (int o = 0; o < l.oc; o++)
                    for (int i = 0; i < l.ic; i++)
                        for (int t = 0; t < 9; t++)
                            packed[e->pack0Off + ((size_t)(t / 3) * 40 + (t % 3) * 12 + i) * 32 + o] = (_Float16)NSS_W(o, i, t / 3, t % 3);
        } else {
            // phase p: tap 0 <- k0 (p=0) or k0+k1 (p=1); tap 1 <- k1+k2 (p=0) or k2 (p=1)
            for (int py = 0; py < 2; py++) for (int px = 0; px < 2; px++)
            for (int ty = 0; ty < 2; ty++) for (int tx = 0; tx < 2; tx++)
            for (int o = 0; o < l.oc; o++) for (int i = 0; i < l.ic; i++) {
                float acc = 0;
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                    const BOOL uy = py == 0 ? (ty == 0 ? ky == 0 : ky >= 1) : (ty == 0 ? ky <= 1 : ky == 2);
                    const BOOL ux = px == 0 ? (tx == 0 ? kx == 0 : kx >= 1) : (tx == 0 ? kx <= 1 : kx == 2);
                    if (uy && ux) acc += NSS_W(o, i, ky, kx);
                }
                packed[l.wOff + ((((size_t)(py * 2 + px) * 2 + ty) * 2 + tx) * l.icp + i) * l.ocp + o] = (_Float16)acc;
            }
            wo_ += (size_t)16 * l.icp * l.ocp;
        }
#undef NSS_W
        so += (size_t)l.oc * l.ic * 9;
        for (int o = 0; o < l.oc; o++) bias[bo + o] = src[so + o];
        so += l.oc, bo += l.ocp;
        e->layers[li] = l;
    }
    if (so * 4 != w.length) { nss_log(@"nss: %@ has %lu bytes, expected %zu", wPath, (unsigned long)w.length, so * 4); free(packed); free(bias); return nil; }
    e->weights = [dev newBufferWithBytes:packed length:total * 2 options:MTLResourceStorageModeShared];
    e->biases = [dev newBufferWithBytes:bias length:btotal * 4 options:MTLResourceStorageModeShared];
    free(packed), free(bias);

    id<MTLBuffer> (^buf)(size_t) = ^id<MTLBuffer>(size_t floats) {
        return [dev newBufferWithLength:MAX(floats, 1) * 4 options:MTLResourceStorageModePrivate];
    };
    const size_t R = (size_t)hr * wr, P = (size_t)e->Hp * e->Wp;
    e->depth = buf(R), e->motion = buf(2 * R);
    e->depthTm1 = buf((size_t)e->Hd * e->Wd), e->disLqB = buf((size_t)e->Hd * e->Wd);
    e->fusedPrep = e->half_ && hr == 4 * e->Hd && wr == 4 * e->Wd && e->prepQ && e->prepQD && e->disLqS && !nssEnv("NOFUSE");
    if (e->fusedPrep) e->depthTm1Alt = buf((size_t)e->Hd * e->Wd), e->blockStats = buf((size_t)e->Hd * e->Wd * 4);
    e->dis = buf(R), e->nearest = buf(e->half_ ? 2 * P : R), e->kBuf = buf(1), e->dilMotion = buf(2 * R);
    {
        // derivative state: RGBA16F (NSS_DERIV32=1: RGBA32F, 0.01 ms slower; both differ from the float buffer by < 0.004/255 mean)
        const BOOL f32 = dev.supports32BitFloatFiltering && nssEnv("DERIV32");
        const MTLPixelFormat fmt = f32 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float;
        const int dh = e->half_ ? e->Hp : hr, dw = e->half_ ? e->Wp : wr;
        const NSUInteger al = [dev minimumLinearTextureAlignmentForPixelFormat:fmt];
        const NSUInteger pitch = ((NSUInteger)dw * (f32 ? 16 : 8) + al - 1) / al * al;
        MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:dw height:dh mipmapped:NO];
        dd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, dd.storageMode = MTLStorageModePrivate;
        for (int i = 0; i < 2; i++) {
            e->deriv[i] = buf(pitch * dh / 4);
            e->derivTex[i] = [e->deriv[i] newTextureWithDescriptor:dd offset:0 bytesPerRow:pitch];
        }
    }
    const int Hp = e->Hp, Wp = e->Wp;
    e->IN = nssTensor(e, Hp, Wp, 12);   // 12 network inputs; conv0 reads 16 (or packed 40) with zero weights past them
    e->A0 = nssTensor(e, Hp / 2, Wp / 2, 32), e->CAT10 = nssTensor(e, Hp / 2, Wp / 2, 48);
    e->A2 = nssTensor(e, Hp / 4, Wp / 4, 32), e->CAT8 = nssTensor(e, Hp / 4, Wp / 4, 48);
    e->A4 = nssTensor(e, Hp / 8, Wp / 8, 64), e->A5 = nssTensor(e, Hp / 8, Wp / 8, 64), e->A6 = nssTensor(e, Hp / 8, Wp / 8, 32);
    e->A8 = nssTensor(e, Hp / 4, Wp / 4, 32), e->KPN = nssTensor(e, Hp / 4, Wp / 4, (e->kpnCh + 7) / 8 * 8);
    e->A10 = nssTensor(e, Hp / 2, Wp / 2, 16), e->A11 = nssTensor(e, Hp / 2, Wp / 2, 16);
    e->TEMP[0] = nssTensor(e, Hp, Wp, 4), e->TEMP[1] = nssTensor(e, Hp, Wp, 4);
    {
        MTLTextureDescriptor *vd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                     width:e->TEMP[0].wpad height:Hp + 4 mipmapped:NO];
        vd.usage = MTLTextureUsageShaderRead, vd.storageMode = MTLStorageModePrivate;
        if ((e->TEMP[0].wpad * 8) % [dev minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA16Float]) return nil;
        for (int i = 0; i < 2; i++) e->tempView[i] = [e->TEMP[i].b newTextureWithDescriptor:vd offset:0 bytesPerRow:e->TEMP[i].wpad * 8];
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:wo height:ho mipmapped:NO];
    td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    e->history[0] = [dev newTextureWithDescriptor:td], e->history[1] = [dev newTextureWithDescriptor:td];
    td.width = td.height = 1;
    e->dummy = [dev newTextureWithDescriptor:td];

    e->idxModH = ho / nss_gcd(hr, ho), e->idxModW = wo / nss_gcd(wr, wo);
    // offset LUT, built on the GPU each frame (see nss_lut_*)
    e->lutBytes = (size_t)e->idxModH * e->idxModW * e->taps * 4;   // one packed uint per tap
    e->lut = [dev newBufferWithLength:e->lutBytes options:MTLResourceStorageModePrivate];
    e->lutSlots = (int)MIN(64, MAX(1, (32u << 20) / e->lutBytes));
    e->lutCache = [dev newBufferWithLength:e->lutBytes * e->lutSlots options:MTLResourceStorageModePrivate];
    e->lutKeys = [NSMutableDictionary dictionary];
    e->lutMap = [dev newBufferWithLength:(size_t)16 * e->idxModH * e->idxModW options:MTLResourceStorageModePrivate];
    e->lutClear = pso(@"nss_lut_clear"), e->lutScatter = pso(@"nss_lut_scatter"), e->lutSelect = pso(@"nss_lut_select");
    if (!e->lutClear || !e->lutScatter || !e->lutSelect) return nil;
    for (int k = 0; k < 36; k++) {   // Gaussian (sigma 2) over the 6x6 window, plus an order bias that breaks ties
        const int tdx = -2 + k / 6, tdy = -2 + k % 6;
        e->lutWgt[k] = expf(-0.5f * (float)(tdy * tdy + tdx * tdx) / 4.0f) + (float)(36 - k) * 1.1920929e-07f * 16.0f;
    }

    // Metal 4: one argument table (bindings are captured at each dispatch), constant uploads, residency
    if (@available(macOS 26.0, *)) {
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new];
        ad.maxBufferBindCount = 16, ad.maxTextureBindCount = 8;
        e->table = [dev newArgumentTableWithDescriptor:ad error:nil];
        for (int i = 0; i < 3; i++) e->upload[i] = [dev newBufferWithLength:64 * 1024 options:MTLResourceStorageModeShared];
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 64;
        e->residency = [dev newResidencySetWithDescriptor:rd error:nil];
        e->gameResidency = [dev newResidencySetWithDescriptor:rd error:nil];
        NSMutableArray *all = [NSMutableArray arrayWithArray:e->tensors];
        [all addObjectsFromArray:@[e->weights, e->biases, e->depth, e->motion, e->depthTm1, e->dis, e->disLqB, e->nearest, e->kBuf, e->dilMotion,
                                   (id)e->depthTm1Alt ?: (id)e->dummy, (id)e->blockStats ?: (id)e->dummy,
                                   e->deriv[0], e->deriv[1], e->history[0], e->history[1], e->dummy, e->tempView[0], e->tempView[1], e->derivTex[0], e->derivTex[1]]];
        [all addObject:e->lut], [all addObject:e->lutMap], [all addObject:e->lutCache];
        for (int i = 0; i < 3; i++) [all addObject:e->upload[i]];
        for (id<MTLAllocation> a in all) [e->residency addAllocation:a];
        [e->residency commit];
        [e->residency requestResidency];
    }

    // view-space depth parameters (forward Z, finite far plane). The game doesn't hand these to MetalFX;
    // they only shape the disocclusion test.
    const float fQ = zFar / (zNear - zFar);
    const float aspect = (float)wr / hr, cot = cosf(0.5f * fovY) / sinf(0.5f * fovY);
    e->d2v[0] = -fQ, e->d2v[1] = fQ * zNear, e->d2v[2] = aspect / cot, e->d2v[3] = 1.0f / cot;

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    for (id<MTLBuffer> b in e->tensors) [be fillBuffer:b range:NSMakeRange(0, b.length) value:0];
    for (int i = 0; i < 2; i++) [be fillBuffer:e->deriv[i] range:NSMakeRange(0, e->deriv[i].length) value:0];
    // history must start finite: the first frame blends it in with weight 0, and NaN * 0 is NaN
    id<MTLBuffer> zeros = [dev newBufferWithLength:(size_t)wo * ho * 8 options:MTLResourceStorageModePrivate];
    [be fillBuffer:zeros range:NSMakeRange(0, zeros.length) value:0];
    for (int i = 0; i < 2; i++)
        [be copyFromBuffer:zeros sourceOffset:0 sourceBytesPerRow:(NSUInteger)wo * 8 sourceBytesPerImage:(NSUInteger)wo * ho * 8
                sourceSize:MTLSizeMake(wo, ho, 1) toTexture:e->history[i] destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    e->packConv0 = !(nssEnv("SPECIALIZE") && nssEnv("SPECIALIZE")[0] == '0') && !nssEnv("NOPACK0");
    nssTune(e, q);
    // post-process compiled for this engine: fixed sizes and quality switches as function constants (30+)
    if (!(nssEnv("SPECIALIZE") && nssEnv("SPECIALIZE")[0] == '0')) {
        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        int i2[7][2] = {{e->Hr, e->Wr}, {e->Ho, e->Wo}, {e->Hp / 4, e->Wp / 4}, {e->Hp, e->Wp}, {e->idxModH, e->idxModW},
                        {e->Hq, e->Wq}, {e->half_ ? e->Hp : e->Hr, e->half_ ? e->Wp : e->Wr}};
        const int i2idx[7] = {30, 31, 32, 33, 34, 41, 42};
        for (int k = 0; k < 7; k++) [fc setConstantValue:i2[k] type:MTLDataTypeInt2 atIndex:i2idx[k]];
        const float one = 1.0f;
        int ints[11][2] = {{35, e->taps}, {36, e->kpnCh}, {38, e->KPN.wpad}, {39, e->KPN.C}, {40, e->TEMP[0].wpad},
                           {43, e->half_}, {44, e->half_}, {45, e->quality >= 1}, {46, 1},
                           {47, nssEnv("POST_ABLATE") ? atoi(nssEnv("POST_ABLATE")) : 0}, {-1, 0}};
        for (int k = 0; ints[k][0] >= 0; k++) [fc setConstantValue:&ints[k][1] type:MTLDataTypeInt atIndex:ints[k][0]];
        [fc setConstantValue:&one type:MTLDataTypeFloat atIndex:37];
        NSError *ferr = nil;
        if (e->half_) {   // mid/low pre-process, same idea (60+)
            MTLFunctionConstantValues *hc = [MTLFunctionConstantValues new];
            int h2[4][2] = {{e->Hr, e->Wr}, {e->Hp, e->Wp}, {e->Hq, e->Wq}, {e->Hd, e->Wd}};
            for (int k = 0; k < 4; k++) [hc setConstantValue:h2[k] type:MTLDataTypeInt2 atIndex:60 + k];
            const float rs[2] = {(float)e->Hr, (float)e->Wr};
            [hc setConstantValue:rs type:MTLDataTypeFloat2 atIndex:64];
            [hc setConstantValue:&one type:MTLDataTypeFloat atIndex:65];
            const int wp[2] = {e->IN.wpad, e->TEMP[0].wpad};
            [hc setConstantValue:&wp[0] type:MTLDataTypeInt atIndex:66];
            [hc setConstantValue:&wp[1] type:MTLDataTypeInt atIndex:67];
            const int abl = nssEnv("PRE_ABLATE") ? atoi(nssEnv("PRE_ABLATE")) : 0;
            [hc setConstantValue:&abl type:MTLDataTypeInt atIndex:68];
            NSError *herr = nil;
            id<MTLFunction> hf = [e->lib newFunctionWithName:@"nss_pre_half" constantValues:hc error:&herr];
            id<MTLComputePipelineState> hp = hf ? [dev newComputePipelineStateWithFunction:hf error:&herr] : nil;
            if (hp) {
                e->preHalf = hp;
                @synchronized(NSNull.null) { [psoNames setObject:@"pre_half" forKey:hp]; }
            } else nss_log(@"nss: specializing pre-process: %@", herr.localizedDescription);
        }
        id<MTLFunction> f = [e->lib newFunctionWithName:@"nss_post" constantValues:fc error:&ferr];
        id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:&ferr] : nil;
        if (ps) {
            e->post = ps;
            @synchronized(NSNull.null) { [psoNames setObject:@"post" forKey:ps]; }
        } else nss_log(@"nss: specializing post-process: %@", ferr.localizedDescription);
    }
    nss_log(@"nss: pipeline limits (max threads per group / SIMD width): post %lu/%lu, pre_half %lu, conv32p4 %lu",
             (unsigned long)e->post.maxTotalThreadsPerThreadgroup, (unsigned long)e->post.threadExecutionWidth,
             (unsigned long)e->preHalf.maxTotalThreadsPerThreadgroup, (unsigned long)e->conv[@324].maxTotalThreadsPerThreadgroup);
    return e;
}

typedef struct {
    __unsafe_unretained id<MTLTexture> color, depth, motion, exposure, output;
    float jy, jx;            // NSS jitter: sample offset from the pixel centre, (row, col) input pixels
    float mvScale[2];        // texel * mvScale = previous - current, input pixels
    BOOL depthReversed, reset;
    float preExposure;
} NssFrame;

static BOOL isDepthFormat(MTLPixelFormat f) {
    return f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float_Stencil8 ||
           f == MTLPixelFormatDepth24Unorm_Stencil8 || f == MTLPixelFormatX32_Stencil8;
}

// Reads back the previous profiled frame (done by now) and sets this frame up to be profiled.
static void nssProfileFrame(NssEngine *e, NssEnc *c) {
    if (@available(macOS 26.0, *)) {
        if (!e->profHeap) {
            MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new];
            hd.type = MTL4CounterHeapTypeTimestamp, hd.count = 64;
            e->profHeap = [e->dev newCounterHeapWithDescriptor:hd error:nil];
            e->profAcc = [NSMutableDictionary dictionary];
        }
        if (e->profPending.count > 1) {
            NSData *d = [(id<MTL4CounterHeap>)e->profHeap resolveCounterRange:NSMakeRange(0, e->profPending.count)];
            const MTL4TimestampHeapEntry *t = d.bytes;
            const double tpm = (double)[e->dev queryTimestampFrequency] / 1000.0;
            // entry 0 marks the start (after the first dispatch's wait); each later entry closes one dispatch
            BOOL ok = d.length >= e->profPending.count * sizeof(MTL4TimestampHeapEntry);
            for (NSUInteger i = 1; ok && i < e->profPending.count; i++) if (t[i].timestamp < t[i - 1].timestamp) ok = NO;
            if (ok) {
                for (NSUInteger i = 1; i < e->profPending.count; i++) {
                    NSString *n = e->profPending[i];
                    if (!e->profAcc[n]) e->profAcc[n] = [NSMutableArray array];
                    [e->profAcc[n] addObject:@((double)(t[i].timestamp - t[i - 1].timestamp) / tpm)];
                }
                if (++e->profFrames % 10 == 0) {   // every 10 profiled frames: median per pass, sorted by cost
                    NSMutableArray *rows = [NSMutableArray array];
                    double total = 0;
                    for (NSString *n in e->profAcc) {
                        NSArray *v = [e->profAcc[n] sortedArrayUsingSelector:@selector(compare:)];
                        // a pass name can repeat within a frame (e.g. several convs): per-frame sum = median * count/frames
                        const double med = [v[v.count / 2] doubleValue] * (double)v.count / 10.0;
                        [rows addObject:@[n, @(med)]];
                        total += med;
                    }
                    [rows sortUsingComparator:^NSComparisonResult(NSArray *a, NSArray *b) { return [b[1] compare:a[1]]; }];
                    NSMutableString *line = [NSMutableString stringWithFormat:@"nss: profile %dx%d -> %dx%d q%d, %.3f ms:", e->Wr, e->Hr, e->Wo, e->Ho, e->quality, total];
                    for (NSArray *r in rows) [line appendFormat:@" %@ %.3f", r[0], [r[1] doubleValue]];
                    nss_log(@"%@", line);
                    [e->profAcc removeAllObjects];
                }
            }
            [(id<MTL4CounterHeap>)e->profHeap invalidateCounterRange:NSMakeRange(0, 64)];
        }
        c->profHeap = e->profHeap;
        c->profNames = [NSMutableArray array];
        e->profPending = nil;
    }
}

// The network as 14 conv dispatches: input, output, and where in the concat buffers they read and write.
typedef struct { int li; NssT in; int inCoff; NssT out; int outCoff, stride, act; BOOL upc; } NssConv;

static void nssConvs(NssEngine *e, int p, NssConv *k) {
    const NssConv list[14] = {
        {0, e->IN, 0, e->A0, 0, 2, 0, NO},
        {1, e->A0, 0, e->CAT10, 16, 1, 0, NO},        // skip1 -> CAT10[16:48]
        {2, e->CAT10, 16, e->A2, 0, 2, 0, NO},
        {3, e->A2, 0, e->CAT8, 16, 1, 0, NO},         // skip2 -> CAT8[16:48]
        {4, e->CAT8, 16, e->A4, 0, 2, 0, NO},
        {5, e->A4, 0, e->A5, 0, 1, 0, NO},
        {6, e->A5, 0, e->A6, 0, 1, 0, NO},
        {7, e->A6, 0, e->CAT8, 0, 1, 0, YES},         // up(A6) -> CAT8[0:16]
        {8, e->CAT8, 0, e->A8, 0, 1, 0, NO},
        {9, e->A8, 0, e->KPN, 0, 1, 1, NO},
        {10, e->A8, 0, e->CAT10, 0, 1, 0, YES},       // up(A8) -> CAT10[0:16]
        {11, e->CAT10, 0, e->A10, 0, 1, 0, NO},
        {12, e->A10, 0, e->A11, 0, 1, 0, NO},
        {13, e->A11, 0, e->TEMP[1 - p], 0, 1, 1, YES},
    };
    memcpy(k, list, sizeof(list));
}

static void nssEncodeConv(NssEngine *e, NssEnc *c, NssConv k, int px) {
    typedef struct { int hOut, wOut, inWpad, inC, inCoff, outWpad, outC, outCoff, IC, stride, act; } MmaArgs;
    const NssLayer l = e->layers[k.li];
    const int h = k.upc ? k.in.H : k.out.H, w = k.upc ? k.in.W : k.out.W;   // upconv: grid over the low-res source
    MmaArgs m = {h, w, k.in.wpad, k.in.C, k.inCoff, k.out.wpad, k.out.C, k.outCoff, l.icp, k.stride, k.act};
    id<MTLComputePipelineState> special = e->layerPSO[k.li];
    const BOOL useSpecial = special && px == e->layerPX[k.li];
    [c setComputePipelineState:useSpecial ? special : (k.upc ? e->upconv : e->conv)[@(l.ocp * 10 + px)]];
    [c setBytes:&m length:sizeof(m) atIndex:0];
    [c setBuffer:k.in.b offset:0 atIndex:1];
    [c setBuffer:e->weights offset:(k.li == 0 && e->packConv0 && useSpecial ? e->pack0Off : l.wOff) * 2 atIndex:2];
    [c setBuffer:e->biases offset:l.bOff * 4 atIndex:3];
    [c setBuffer:k.out.b offset:0 atIndex:4];
    [c setThreadgroupMemoryLength:4 * 8 * l.ocp * 4 atIndex:0];
    [c dispatchThreadgroups:MTLSizeMake((w + 32 * px - 1) / (32 * px), h, useSpecial ? MAX(e->layerSplit[k.li], 1) : 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// Picks each layer's pixel tiles per simdgroup (1, 2 or 4) by timing them on this GPU at this size.
// Picks each layer's kernel: pixel tiles per simdgroup (1, 2 or 4), compiled with the layer's shape as function
// constants (fixed loop counts and strides), by timing all three on this GPU at this size.
static void nssTune(NssEngine *e, id<MTLCommandQueue> q) {
    NssConv convs[14];
    nssConvs(e, 0, convs);
    const BOOL specialize = !(nssEnv("SPECIALIZE") && nssEnv("SPECIALIZE")[0] == '0');
    NssConv *cv = convs;   // blocks can't capture C arrays
    // variants per layer: (pixel tiles, output-channel groups); splits only for plain convs with enough channels
    enum { NV = 6 };
    static const int vPx[NV] = {1, 2, 4, 1, 2, 1}, vSplit[NV] = {1, 1, 1, 2, 2, 4};
    const BOOL trySplit = !nssEnv("NOSPLIT");
    NSMutableArray *cands = [NSMutableArray arrayWithCapacity:14 * NV];   // [layer * NV + variant], NSNull if missing
    for (int n = 0; n < 14 * NV; n++) [cands addObject:NSNull.null];
    CFTimeInterval t0 = CACurrentMediaTime();
    if (specialize) {
        NSLock *lock = [NSLock new];
        dispatch_apply(14 * NV, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
            const int i = (int)n / NV, px = vPx[n % NV], split = vSplit[n % NV];
            const NssConv k = cv[i];
            const NssLayer l = e->layers[i];
            if (split > 1 && (!trySplit || k.upc || l.ocp / 8 < split * 1 || (l.ocp / 8) % split)) return;
            MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
            const BOOL pack = i == 0 && e->packConv0;
            int shapeVals[9] = {pack ? 40 : l.icp, k.in.C, k.inCoff, k.out.C, k.outCoff, k.stride, k.act, k.in.wpad, k.out.wpad};
            for (int j = 0; j < 9; j++) [fc setConstantValue:&shapeVals[j] type:MTLDataTypeInt atIndex:10 + j];
            if (pack) { const bool yes = true; [fc setConstantValue:&yes type:MTLDataTypeBool atIndex:19]; }
            if (split > 1) [fc setConstantValue:&split type:MTLDataTypeInt atIndex:20];
            NSError *err = nil;
            NSString *name = [NSString stringWithFormat:@"nss_%@conv_mma_%d_p%d", k.upc ? @"up" : @"", l.ocp, px];
            id<MTLFunction> f = [e->lib newFunctionWithName:name constantValues:fc error:&err];
            id<MTLComputePipelineState> ps = f ? [e->dev newComputePipelineStateWithFunction:f error:&err] : nil;
            if (!ps) nss_log(@"nss: specializing layer %d: %@", i, err.localizedDescription);
            [lock lock];
            if (ps) cands[n] = ps;
            [lock unlock];
        });
    }
    const double compileMs = (CACurrentMediaTime() - t0) * 1e3;
    id<MTLComputePipelineState> (^cand)(int, int) = ^id<MTLComputePipelineState>(int i, int v) {
        id x = cands[i * NV + v];
        return x == NSNull.null ? nil : x;
    };
    NSMutableString *picked = [NSMutableString string];
    int forced[14][2] = {{0}};
    if (nssEnv("TILES")) {
        NSArray *f = [@(nssEnv("TILES")) componentsSeparatedByString:@" "];
        for (int i = 0; i < 14 && i < (int)f.count; i++) {
            NSArray *ps = [f[i] componentsSeparatedByString:@"x"];
            forced[i][0] = [ps[0] intValue], forced[i][1] = ps.count > 1 ? [ps[1] intValue] : 1;
        }
    }
    for (int i = 0; i < 14; i++) {
        double best = 1e9;
        int bestPx = 1, bestSplit = 1;
        id<MTLComputePipelineState> bestPso = nil;
        double vBest[NV];
        for (int v = 0; v < NV; v++) vBest[v] = 1e9;
        for (int round = 0; round < 4; round++)          // round 0 warms up; best of rounds 1-3 per variant
            for (int v = 0; v < NV; v++) {
                const int px = vPx[v];
                if (vSplit[v] > 1 && !cand(i, v)) continue;
                e->layerPSO[i] = cand(i, v), e->layerPX[i] = px, e->layerSplit[i] = vSplit[v];   // used when px matches
                id<MTLCommandBuffer> cb = [q commandBuffer];
                NssEnc *c = [NssEnc new];
                c->m3 = [cb computeCommandEncoder];
                for (int r = 0; r < 8; r++) nssEncodeConv(e, c, convs[i], px);
                [c->m3 endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double t = cb.GPUEndTime - cb.GPUStartTime;
                if (round > 0) vBest[v] = MIN(vBest[v], t);
                if (round == 3) {
                    // switch from an earlier (simpler) variant only when clearly faster: isolated timings are noisy
                    if (vBest[v] < best * 0.97) best = vBest[v], bestPx = px, bestSplit = vSplit[v], bestPso = cand(i, v);
                    if (nssEnv("TUNE_LOG"))
                        nss_log(@"nss: tune layer %d variant px%d split%d: %.1f us per dispatch", i, px, vSplit[v], vBest[v] * 1e6 / 8);
                }
            }
        if (forced[i][0]) {   // NSS_TILES: fixed choice per layer ("PX" or "PXxSPLIT"), for A/B tests
            for (int v = 0; v < NV; v++)
                if (vPx[v] == forced[i][0] && vSplit[v] == forced[i][1] && cand(i, v)) bestPx = vPx[v], bestSplit = vSplit[v], bestPso = cand(i, v);
        }
        e->layerPX[i] = bestPx, e->layerPSO[i] = bestPso, e->layerSplit[i] = bestPso ? bestSplit : 1;
        if (bestPso) @synchronized(NSNull.null) { [psoNames setObject:[NSString stringWithFormat:@"conv%d", i] forKey:bestPso]; }
        [picked appendFormat:e->layerSplit[i] > 1 ? @" %dx%d" : @" %d", bestPx, e->layerSplit[i]];
    }
    nss_log(@"nss: tuned conv layers (pixel tiles:%@), %s, compiled in %.0f ms, total %.0f ms", picked,
             specialize ? "specialized" : "generic", compileMs, (CACurrentMediaTime() - t0) * 1e3);
}

static void nssEncode(NssEngine *e, id cbObj, BOOL metal4, id<MTLFence> fence, const NssFrame *fr) {
    const int Hr = e->Hr, Wr = e->Wr, Ho = e->Ho, Wo = e->Wo, Hp = e->Hp, Wp = e->Wp, Hq = e->Hq, Wq = e->Wq, Hd = e->Hd, Wd = e->Wd;
    const int p = e->p;
    const BOOL reset = e->frame == 0 || fr->reset;
    const float resetF = reset ? 1.f : 0.f;
    // this frame's offset LUT: cached per jitter value, built (and kept) on a miss
    id<MTLBuffer> lutB = e->lut;
    size_t lutOff = 0;
    BOOL buildLut = YES;
    {
        uint32_t jyb, jxb;
        memcpy(&jyb, &fr->jy, 4), memcpy(&jxb, &fr->jx, 4);
        NSNumber *key = @(((uint64_t)jyb << 32) | jxb);
        NSNumber *slot = e->lutKeys[key];
        if (slot) {
            buildLut = NO;
        } else {
            int victim = -1;
            if (e->lutUsed < e->lutSlots) victim = e->lutUsed++;
            else {   // least recently used, if no frame in flight can still be reading it
                int64_t oldest = INT64_MAX;
                for (int k = 0; k < e->lutSlots; k++) if (e->lutLastUse[k] < oldest) oldest = e->lutLastUse[k], victim = k;
                if (e->frame - oldest < 4) victim = -1;
                else for (NSNumber *k in e->lutKeys.allKeys) if (e->lutKeys[k].intValue == victim) [e->lutKeys removeObjectForKey:k];
            }
            if (victim >= 0) slot = @(victim), e->lutKeys[key] = slot;
        }
        if (slot) {
            lutB = e->lutCache, lutOff = (size_t)slot.intValue * e->lutBytes;
            e->lutLastUse[slot.intValue] = e->frame;
        }
    }

    NssEnc *c = [NssEnc new];
    const BOOL cut = reset && e->frame > 0;   // camera cut: clear the recurrent state this frame reads
    if (!metal4) {
        id<MTLCommandBuffer> cb = cbObj;
        if (cut) {
            id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
            [be fillBuffer:e->deriv[p] range:NSMakeRange(0, e->deriv[p].length) value:0];
            [be fillBuffer:e->TEMP[p].b range:NSMakeRange(0, e->TEMP[p].b.length) value:0];
            [be endEncoding];
        }
        c->m3 = [cb computeCommandEncoder];
        c->m3.label = @"NSS";
    } else if (@available(macOS 26.0, *)) {
        id<MTL4CommandBuffer> cb = cbObj;
        // the game's textures must be resident too; add new ones as they appear
        BOOL added = NO;
        for (id<MTLTexture> t in @[fr->color, fr->depth, fr->motion, fr->output, fr->exposure ?: e->dummy])
            if (![e->gameResidency containsAllocation:t]) [e->gameResidency addAllocation:t], added = YES;
        if (added) [e->gameResidency commit];
        [cb useResidencySet:e->residency];
        [cb useResidencySet:e->gameResidency];
        c->m4 = [cb computeCommandEncoder];
        c->m4.label = @"NSS";
        c->table = e->table;
        c->upload = e->upload[e->uploadSlot];
        e->uploadSlot = (e->uploadSlot + 1) % 3;
        [c->m4 setArgumentTable:e->table];
        static int profileEvery = -1;
        if (profileEvery < 0) profileEvery = nssEnv("PROFILE") ? atoi(nssEnv("PROFILE")) : 0;
        if (profileEvery > 0 && e->frame % profileEvery == 0) nssProfileFrame(e, c);
        // after the game's work on the inputs (MetalFX's own contract: the scaler's fence, if the game set one)
        [c->m4 barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        if (fence) [c->m4 waitForFence:fence beforeEncoderStages:MTLStageDispatch | MTLStageBlit];
        [c stamp:@"start"];
        if (cut) {
            [c->m4 fillBuffer:e->deriv[p] range:NSMakeRange(0, e->deriv[p].length) value:0];
            [c->m4 fillBuffer:e->TEMP[p].b range:NSMakeRange(0, e->TEMP[p].b.length) value:0];
            c->dispatched = YES;   // the first dispatch waits for the fills
        }
    }
    void (^grid)(id<MTLComputePipelineState>, int, int) = ^(id<MTLComputePipelineState> s, int w, int h) {   // c: NssEnc
        [c setComputePipelineState:s];
        [c dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    };
    // experiment: threadgroup shape for the big per-pixel passes (NSS_TG_POST / _PRE = "WxH")
    void (^gridTG)(id<MTLComputePipelineState>, int, int, const char *) = ^(id<MTLComputePipelineState> s, int w, int h, const char *env) {
        int tw = 16, th = 16;
        if (getenv(env)) sscanf(getenv(env), "%dx%d", &tw, &th);
        [c setComputePipelineState:s];
        [c dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    };

    // game inputs
    struct { int dims[2]; float reset; int pad0 /* Metal aligns float2 to 8 bytes */; float mvScale[2]; int depthReversed; float preExposureInv; int hasExposure; } ga =
        {{Hr, Wr}, resetF, 0, {fr->mvScale[0], fr->mvScale[1]}, fr->depthReversed, fr->preExposure > 0 ? 1.0f / fr->preExposure : 1.0f, fr->exposure != nil};
    const int gh = nss_gcd(Hr, Ho), gw = nss_gcd(Wr, Wo);
    struct { float jitter[2], scale[2]; int lo[2], hi[2]; int taps; } la =
        {{fr->jy, fr->jx}, {(float)Ho / Hr, (float)Wo / Wr}, {Hr / gh, Wr / gw}, {Ho / gh, Wo / gw}, e->taps};
    struct { int in[2], out[2]; } sa = {{Hr, Wr}, {Hd, Wd}};
    if (e->fusedPrep) {
        // game prep + depth scatter in one pass; disocclusion from its block stats (it also clears next frame's target)
        id<MTLBuffer> dTm = p ? e->depthTm1Alt : e->depthTm1, dNext = p ? e->depthTm1 : e->depthTm1Alt;
        if (e->frame == 0) {
            [c setBytes:&sa length:sizeof(sa) atIndex:0];
            [c setBuffer:dTm offset:0 atIndex:1];
            grid(e->dclear, Wd, Hd);
        }
        struct { typeof(ga) g; int pad1[3]; float d2v[4]; int out[2], pad2[2]; } qa_ = {ga, {0}, {e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {Hd, Wd}, {0}};
        _Static_assert(sizeof(ga) == 36, "GamePrepArgs");
        [c setBytes:&qa_ length:sizeof(qa_) atIndex:0];
        [c setTexture:fr->depth atIndex:1], [c setTexture:fr->motion atIndex:2], [c setTexture:fr->exposure ?: e->dummy atIndex:3];
        [c setBuffer:e->depth offset:0 atIndex:2], [c setBuffer:e->motion offset:0 atIndex:3], [c setBuffer:e->kBuf offset:0 atIndex:4];
        [c setBuffer:dTm offset:0 atIndex:5], [c setBuffer:e->blockStats offset:0 atIndex:6];
        grid(isDepthFormat(fr->depth.pixelFormat) ? e->prepQD : e->prepQ, Wd, Hd);
        if (buildLut) {
            c->overlapNext = YES;
            [c setBytes:&la length:sizeof(la) atIndex:0];
            [c setBuffer:e->lutMap offset:0 atIndex:1];
            [c setComputePipelineState:e->lutClear];
            [c dispatchThreads:MTLSizeMake(la.hi[0] * la.hi[1], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            grid(e->lutScatter, la.lo[1], la.lo[0]);
            c->overlapNext = YES;
        }
        struct { float d2v[4]; float renderSize[2]; int in[2], out[2]; } da =
            {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hd, Wd}};
        [c setBytes:&da length:sizeof(da) atIndex:0];
        [c setBuffer:e->blockStats offset:0 atIndex:1], [c setBuffer:dNext offset:0 atIndex:2], [c setBuffer:dTm offset:0 atIndex:3];
        [c setBuffer:e->disLqB offset:0 atIndex:4];
        grid(e->disLqS, Wd, Hd);
        if (buildLut) {
            [c setBytes:&la length:sizeof(la) atIndex:0];
            [c setBuffer:e->lutMap offset:0 atIndex:1];
            [c setBytes:e->lutWgt length:sizeof(e->lutWgt) atIndex:2];
            [c setBuffer:lutB offset:lutOff atIndex:3];
            [c setComputePipelineState:e->lutSelect];
            [c dispatchThreads:MTLSizeMake(la.hi[0] * la.hi[1], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
    } else {
        [c setBytes:&ga length:sizeof(ga) atIndex:0];
        [c setTexture:fr->color atIndex:0], [c setTexture:fr->depth atIndex:1], [c setTexture:fr->motion atIndex:2];
        [c setTexture:fr->exposure ?: e->dummy atIndex:3];
        [c setBuffer:e->depth offset:0 atIndex:2], [c setBuffer:e->motion offset:0 atIndex:3], [c setBuffer:e->kBuf offset:0 atIndex:4];
        grid(isDepthFormat(fr->depth.pixelFormat) ? e->gprepD : e->gprep, Wr, Hr);

        // Offset LUT for this frame's jitter and the depth scatter are independent chains: on Metal 4 each LUT pass
        // shares a barrier slot with a depth pass (overlapNext), instead of each waiting for the other.
        if (buildLut) {
            // slot 1 (alongside game prep, which neither reads): clear both
            c->overlapNext = YES;
            [c setBytes:&la length:sizeof(la) atIndex:0];
            [c setBuffer:e->lutMap offset:0 atIndex:1];
            [c setComputePipelineState:e->lutClear];
            [c dispatchThreads:MTLSizeMake(la.hi[0] * la.hi[1], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            c->overlapNext = YES;
        }
        [c setBytes:&sa length:sizeof(sa) atIndex:0];
        [c setBuffer:e->depthTm1 offset:0 atIndex:1];
        grid(e->dclear, Wd, Hd);
        if (buildLut) {   // slot 2: scatter both
            [c setBytes:&la length:sizeof(la) atIndex:0];
            [c setBuffer:e->lutMap offset:0 atIndex:1];
            grid(e->lutScatter, la.lo[1], la.lo[0]);
            c->overlapNext = YES;
        }
        [c setBytes:&sa length:sizeof(sa) atIndex:0];
        [c setBuffer:e->motion offset:0 atIndex:1], [c setBuffer:e->depth offset:0 atIndex:2], [c setBuffer:e->depthTm1 offset:0 atIndex:3];
        grid(e->half_ ? e->dscatterQ : e->dscatter, Wd, Hd);
        if (buildLut) {   // slot 3: LUT selection (+ low-res disocclusion for mid/low)
            [c setBytes:&la length:sizeof(la) atIndex:0];
            [c setBuffer:e->lutMap offset:0 atIndex:1];
            [c setBytes:e->lutWgt length:sizeof(e->lutWgt) atIndex:2];
            [c setBuffer:lutB offset:lutOff atIndex:3];
            [c setComputePipelineState:e->lutSelect];
            [c dispatchThreads:MTLSizeMake(la.hi[0] * la.hi[1], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        if (e->half_) {
            if (buildLut) c->overlapNext = YES;
            struct { float d2v[4]; float renderSize[2]; int in[2], out[2]; } da =
                {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hd, Wd}};
            [c setBytes:&da length:sizeof(da) atIndex:0];
            [c setBuffer:e->motion offset:0 atIndex:1], [c setBuffer:e->depth offset:0 atIndex:2], [c setBuffer:e->depthTm1 offset:0 atIndex:3];
            [c setBuffer:e->disLqB offset:0 atIndex:4];
            grid(e->disLq, Wd, Hd);
        }
    }

    // pre-process
    [c setTexture:fr->color atIndex:0], [c setTexture:e->history[p] atIndex:1], [c setTexture:e->tempView[p] atIndex:2];
    [c setTexture:e->derivTex[p] atIndex:3], [c setTexture:e->derivTex[1 - p] atIndex:4];
    [c setBuffer:e->kBuf offset:0 atIndex:12];
    const float ctInv[2] = {1.0f / fr->color.height, 1.0f / fr->color.width};   // (row, col)
    if (e->half_) {
        struct { float d2v[4]; float jitter[2], renderSize[2]; int inDims[2], padDims[2], procDims[2], depthDims[2];
                 float exposure; int inWpad, tempWpad; int pad0 /* Metal: float2 at byte 80 */; float colorTexInv[2]; } ha =
            {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {fr->jy, fr->jx}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hp, Wp}, {Hq, Wq},
             {Hd, Wd}, 1.0f, e->IN.wpad, e->TEMP[0].wpad, 0, {ctInv[0], ctInv[1]}};
        [c setBytes:&ha length:sizeof(ha) atIndex:0];
        id<MTLBuffer> hb[11] = {nil, e->depth, e->motion, nil, e->disLqB, nil, nil, e->IN.b, nil, e->dis, e->nearest};
        for (int i = 0; i < 11; i++) if (hb[i]) [c setBuffer:hb[i] offset:0 atIndex:i + 1];
        [c setBuffer:e->dilMotion offset:0 atIndex:13];
        gridTG(e->preHalf, Wp, Hp, "NSS_TG_PRE");
    } else {
        struct { float d2v[4]; float jitter[2], renderSize[2]; int inDims[2], padDims[2], depthDims[2], outDims[2];
                 float exposure; int inWpad, tempWpad; int pad0 /* Metal: float2 at byte 80 */; float colorTexInv[2]; } ra =
            {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {fr->jy, fr->jx}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hp, Wp}, {Hd, Wd},
             {Ho, Wo}, 1.0f, e->IN.wpad, e->TEMP[0].wpad, 0, {ctInv[0], ctInv[1]}};
        [c setBytes:&ra length:sizeof(ra) atIndex:0];
        id<MTLBuffer> pb[11] = {nil, e->depth, e->motion, nil, e->depthTm1, nil, nil, e->IN.b, nil, e->dis, e->nearest};
        for (int i = 0; i < 11; i++) if (pb[i]) [c setBuffer:pb[i] offset:0 atIndex:i + 1];
        grid(e->pre, Wp, Hp);
    }

    // network
    NssConv convs[14];
    nssConvs(e, p, convs);
    for (int i = 0; i < 14; i++) {
        if (i == 10) c->overlapNext = YES;   // conv9 (up from A8) and the KPN head both only read A8
        nssEncodeConv(e, c, convs[i], e->layerPX[i]);
    }

    // post-process
    struct {
        int inDims[2], outDims[2], kpnDims[2], tempDims[2], idxMod[2];
        int taps, kpnChannels;
        float exposure, reset;
        int kpnWpad, kpnC, tempWpad;
        int pad0;   // Metal aligns the int2 that follows to 8 bytes
        int procDims[2], nearestDims[2];
        int halfRes, sparse, catmullHist;
        int writeGame;
        int ablate;
    } qa = {{Hr, Wr}, {Ho, Wo}, {Hp / 4, Wp / 4}, {Hp, Wp}, {e->idxModH, e->idxModW}, e->taps, e->kpnCh, 1.0f, 1.0f - resetF,
            e->KPN.wpad, e->KPN.C, e->TEMP[0].wpad, 0, {Hq, Wq}, {e->half_ ? Hp : Hr, e->half_ ? Wp : Wr},
            e->half_, e->half_, e->quality >= 1, 1, nssEnv("POST_ABLATE") ? atoi(nssEnv("POST_ABLATE")) : 0};
    [c setBytes:&qa length:sizeof(qa) atIndex:0];
    [c setBuffer:e->KPN.b offset:0 atIndex:3], [c setTexture:e->tempView[1 - p] atIndex:4];
    [c setBuffer:e->motion offset:0 atIndex:5], [c setBuffer:e->nearest offset:0 atIndex:6], [c setBuffer:lutB offset:lutOff atIndex:7];
    [c setBuffer:e->dilMotion offset:0 atIndex:8];
    [c setTexture:e->history[1 - p] atIndex:0], [c setTexture:fr->color atIndex:1], [c setTexture:e->history[p] atIndex:2];
    [c setTexture:fr->output atIndex:3], [c setBuffer:e->kBuf offset:0 atIndex:9];
    gridTG(e->post, Wo, Ho, "NSS_TG_POST");   // also writes the game's output
    if (c->m4) {
        if (c->profHeap) e->profPending = c->profNames;
        if (fence) [c->m4 updateFence:fence afterEncoderStages:MTLStageDispatch];
        [c->m4 endEncoding];
    } else {
        [c->m3 endEncoding];
    }
    e->p = 1 - p;
    e->frame++;
}

#pragma mark - Public API

@implementation NSSUpscalerDescriptor
- (instancetype)init {
    if ((self = [super init])) _quality = NSSQualityMid, _zNear = 0.1f, _zFar = 1000.f, _verticalFieldOfView = 1.0f;
    return self;
}
- (id)copyWithZone:(NSZone *)zone {
    NSSUpscalerDescriptor *d = [NSSUpscalerDescriptor new];
    d.inputWidth = _inputWidth, d.inputHeight = _inputHeight, d.outputWidth = _outputWidth, d.outputHeight = _outputHeight;
    d.quality = _quality, d.resourceDirectory = _resourceDirectory, d.library = _library;
    d.zNear = _zNear, d.zFar = _zFar, d.verticalFieldOfView = _verticalFieldOfView;
    return d;
}
@end

static NSError *nssError(NSString *what) {
    return [NSError errorWithDomain:@"NSSMetal" code:1 userInfo:@{NSLocalizedDescriptionKey: what}];
}

// The kernels: the descriptor's library, resourceDirectory/nss.metallib, or (Swift package builds) the bundled
// source compiled now.
static id<MTLLibrary> nssLoadLibrary(id<MTLDevice> dev, NSSUpscalerDescriptor *d, NSError **error) {
    if (d.library) return d.library;
    NSError *err = nil;
    if (d.resourceDirectory) {
        NSURL *u = [d.resourceDirectory URLByAppendingPathComponent:@"nss.metallib"];
        if ([NSFileManager.defaultManager fileExistsAtPath:u.path]) {
            id<MTLLibrary> lib = [dev newLibraryWithURL:u error:&err];
            if (lib) return lib;
            if (error) *error = nssError([NSString stringWithFormat:@"cannot load %@: %@", u.path, err.localizedDescription]);
            return nil;
        }
    }
#ifdef SWIFTPM_MODULE_BUNDLE
    NSURL *src = nil;
    @try {   // the accessor throws when the resource bundle isn't next to the binary
        src = [SWIFTPM_MODULE_BUNDLE URLForResource:@"nss" withExtension:@"metal" subdirectory:@"Shaders"];
    } @catch (NSException *x) {
        nss_log(@"nss: no bundled kernel source: %@", x.reason);
    }
    NSString *text = src ? [NSString stringWithContentsOfURL:src encoding:NSUTF8StringEncoding error:nil] : nil;
    if (text) {
        MTLCompileOptions *o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [dev newLibraryWithSource:text options:o error:&err];
        if (lib) return lib;
        if (error) *error = nssError([NSString stringWithFormat:@"compiling nss.metal: %@", err.localizedDescription]);
        return nil;
    }
#endif
    if (error) *error = nssError(@"no NSS kernels: set library, or put nss.metallib in resourceDirectory");
    return nil;
}

@implementation NSSUpscaler {
    NssEngine *_engine;
    id<MTLCommandQueue> _queue;
}

+ (void)setLogHandler:(void (^)(NSString *))handler {
    nssLogHandler = [handler copy];
}

- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(NSSUpscalerDescriptor *)d error:(NSError **)error {
    if (!(self = [super init])) return nil;
    const int wr = (int)d.inputWidth, hr = (int)d.inputHeight, wo = (int)d.outputWidth, ho = (int)d.outputHeight;
    if (wr < 8 || hr < 8 || wo <= wr || ho <= hr) {
        if (error) *error = nssError(@"the output must be larger than the input, and the input at least 8x8");
        return nil;
    }
    if (!d.resourceDirectory) {
        if (error) *error = nssError(@"resourceDirectory (with the weights) is not set");
        return nil;
    }
    id<MTLLibrary> lib = nssLoadLibrary(device, d, error);
    if (!lib) return nil;
    _queue = [device newCommandQueue];
    _engine = nssCreate(device, _queue, lib, d.resourceDirectory.path, wr, hr, wo, ho, (int)d.quality, d.zNear, d.zFar,
                        d.verticalFieldOfView);
    if (!_engine) {
        if (error) *error = nssError(@"setup failed (missing weights or kernels; set a log handler for details)");
        return nil;
    }
    _inputWidth = d.inputWidth, _inputHeight = d.inputHeight, _outputWidth = d.outputWidth, _outputHeight = d.outputHeight;
    _quality = d.quality, _networkWidth = (NSUInteger)_engine->Wp, _networkHeight = (NSUInteger)_engine->Hp;
    _preExposure = 1.0f;
    return self;
}

// MetalFX conventions -> the engine's: NSS wants the sample offset as (row, col), MetalFX gives its negation as (x, y).
- (BOOL)frame:(NssFrame *)fr {
    if (!_colorTexture || !_depthTexture || !_motionTexture || !_outputTexture) {
        nss_log(@"nss: color, depth, motion and output textures are required");
        return NO;
    }
    *fr = (NssFrame){_colorTexture, _depthTexture, _motionTexture, _exposureTexture, _outputTexture,
                     -_jitterOffsetY, -_jitterOffsetX, {_motionVectorScaleX, _motionVectorScaleY},
                     _depthReversed, _reset, _preExposure};
    return YES;
}

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NssFrame fr;
    if (![self frame:&fr]) return;
    nssEncode(_engine, commandBuffer, NO, nil, &fr);
    _reset = NO;
}

- (void)encodeToMTL4CommandBuffer:(id<MTL4CommandBuffer>)commandBuffer fence:(id<MTLFence>)fence {
    NssFrame fr;
    if (![self frame:&fr]) return;
    nssEncode(_engine, commandBuffer, YES, fence, &fr);
    _reset = NO;
}
@end
