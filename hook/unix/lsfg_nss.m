// NSS upscaling in place of MetalFX's temporal upscaler.
//
// D3DMetal turns a game's DLSS calls into MetalFX temporal upscaling. When a game's profile picks
// "upscaler": "nss", the hook catches -[<MetalFX temporal scaler> encodeToCommandBuffer:] and, instead,
// encodes Arm Neural Super Sampling v1 (kernels in nss.metallib, weights from Arm's Hugging Face release)
// into the same command buffer, reading the inputs the game handed MetalFX: color, depth, motion,
// exposure, jitter, motion-vector scale, reversed depth, reset.
//
// The pass schedule and the offset LUT port Arm's Apache-2.0 reference (arm/neural-graphics-model-gym,
// usecases/nss). Metal 3 command buffers only for now; on anything else MetalFX runs as usual.
//
// Included by lsfghook_unix.m (uses lsfg_log and config).

#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>

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
    id<MTLComputePipelineState> gprep, gprepD, dclear, dscatter, dscatterQ, disLq, pre, preHalf, post;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *conv, *upconv;
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
    NssLayer layers[14];
    int layerPX[14];                      // pixel tiles per simdgroup, picked by timing at setup
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

// Profiling (LSFG_NSS_PROFILE=N, Metal 4 only): every Nth frame, a precise GPU timestamp after each dispatch; the
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
    if (uploadOff + n > upload.length) { lsfg_log(@"nss: upload buffer full"); return; }
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

static NSString *nssDir(void) {
    if (getenv("LSFG_NSS_DIR")) return @(getenv("LSFG_NSS_DIR"));   // tests: freshly built kernels
    return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/lsfg-metal/nss"];
}

static NssT nssTensor(NssEngine *e, int H, int W, int C) {
    NssT t = {nil, H, W, C, (W + 15) / 16 * 16 + 16};
    id<MTLBuffer> b = [e->dev newBufferWithLength:(size_t)(H + 4) * t.wpad * C * 2 options:MTLResourceStorageModePrivate];
    [e->tensors addObject:b];
    t.b = b;
    return t;
}

static void nssTune(NssEngine *e, id<MTLCommandQueue> q);

// quality: 0 low, 1 mid, 2 high. Returns nil (and logs) when something is missing.
static NssEngine *nssCreate(id<MTLDevice> dev, id<MTLCommandQueue> q, int wr, int hr, int wo, int ho, int quality,
                            float zNear, float zFar, float fovY) {
    NssEngine *e = [NssEngine new];
    e->dev = dev, e->Wr = wr, e->Hr = hr, e->Wo = wo, e->Ho = ho, e->quality = quality;
    e->half_ = quality < 2;
    e->kpnCh = e->half_ ? 16 : 36, e->taps = e->half_ ? 4 : 9;
    e->Hq = e->half_ ? hr / 2 : hr, e->Wq = e->half_ ? wr / 2 : wr;
    e->Hp = (e->Hq + 7) / 8 * 8, e->Wp = (e->Wq + 7) / 8 * 8;
    e->Hd = e->half_ ? hr / 4 : hr / 2, e->Wd = e->half_ ? wr / 4 : wr / 2;
    e->tensors = [NSMutableArray array];

    NSError *err = nil;
    NSString *libPath = [nssDir() stringByAppendingPathComponent:@"nss.metallib"];
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (!lib) { lsfg_log(@"nss: cannot load %@: %@", libPath, err.localizedDescription); return nil; }
    e->lib = lib;
    id<MTLComputePipelineState> (^pso)(NSString *) = ^id<MTLComputePipelineState>(NSString *n) {
        NSError *pe = nil;
        // through the specialization API: kernels that declare function constants need it even with none set
        id<MTLFunction> f = [lib newFunctionWithName:n constantValues:[MTLFunctionConstantValues new] error:&pe];
        id<MTLComputePipelineState> s = f ? [dev newComputePipelineStateWithFunction:f error:&pe] : nil;
        if (!s) lsfg_log(@"nss: pipeline %@: %@", n, pe.localizedDescription ?: @"missing");
        @synchronized(NSNull.null) {
            if (!psoNames) psoNames = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsObjectPointerPersonality valueOptions:NSPointerFunctionsStrongMemory];
            if (s) [psoNames setObject:[n stringByReplacingOccurrencesOfString:@"nss_" withString:@""] forKey:s];
        }
        return s;
    };
    e->gprep = pso(@"nss_game_prep"), e->gprepD = pso(@"nss_game_prep_d");
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
    NSString *wPath = [nssDir() stringByAppendingPathComponent:e->half_ ? @"nss_midlow.bin" : @"nss_high.bin"];
    NSData *w = [NSData dataWithContentsOfFile:wPath];
    if (!w) { lsfg_log(@"nss: cannot read %@", wPath); return nil; }
    const float *src = (const float *)w.bytes;
    const int shape[14][2] = {{12, 32}, {32, 32}, {32, 32}, {32, 32}, {32, 64}, {64, 64}, {64, 32},
                              {32, 16}, {48, 32}, {32, e->kpnCh}, {32, 16}, {48, 16}, {16, 16}, {16, 4}};
    size_t total = 0, btotal = 0;
    for (int li = 0; li < 14; li++) {
        const int icp = (shape[li][0] + 7) / 8 * 8, ocp = (shape[li][1] + 7) / 8 * 8;
        total += (size_t)((li == 7 || li == 10 || li == 13) ? 16 : 9) * icp * ocp, btotal += ocp;
    }
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
    if (so * 4 != w.length) { lsfg_log(@"nss: %@ has %lu bytes, expected %zu", wPath, (unsigned long)w.length, so * 4); free(packed); free(bias); return nil; }
    e->weights = [dev newBufferWithBytes:packed length:total * 2 options:MTLResourceStorageModeShared];
    e->biases = [dev newBufferWithBytes:bias length:btotal * 4 options:MTLResourceStorageModeShared];
    free(packed), free(bias);

    id<MTLBuffer> (^buf)(size_t) = ^id<MTLBuffer>(size_t floats) {
        return [dev newBufferWithLength:MAX(floats, 1) * 4 options:MTLResourceStorageModePrivate];
    };
    const size_t R = (size_t)hr * wr, P = (size_t)e->Hp * e->Wp;
    e->depth = buf(R), e->motion = buf(2 * R);
    e->depthTm1 = buf((size_t)e->Hd * e->Wd), e->disLqB = buf((size_t)e->Hd * e->Wd);
    e->dis = buf(R), e->nearest = buf(e->half_ ? 2 * P : R), e->kBuf = buf(1);
    for (int i = 0; i < 2; i++) e->deriv[i] = buf(4 * (e->half_ ? P : R));
    const int Hp = e->Hp, Wp = e->Wp;
    e->IN = nssTensor(e, Hp, Wp, 16);
    e->A0 = nssTensor(e, Hp / 2, Wp / 2, 32), e->CAT10 = nssTensor(e, Hp / 2, Wp / 2, 48);
    e->A2 = nssTensor(e, Hp / 4, Wp / 4, 32), e->CAT8 = nssTensor(e, Hp / 4, Wp / 4, 48);
    e->A4 = nssTensor(e, Hp / 8, Wp / 8, 64), e->A5 = nssTensor(e, Hp / 8, Wp / 8, 64), e->A6 = nssTensor(e, Hp / 8, Wp / 8, 32);
    e->A8 = nssTensor(e, Hp / 4, Wp / 4, 32), e->KPN = nssTensor(e, Hp / 4, Wp / 4, (e->kpnCh + 7) / 8 * 8);
    e->A10 = nssTensor(e, Hp / 2, Wp / 2, 16), e->A11 = nssTensor(e, Hp / 2, Wp / 2, 16);
    e->TEMP[0] = nssTensor(e, Hp, Wp, 8), e->TEMP[1] = nssTensor(e, Hp, Wp, 8);

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
        [all addObjectsFromArray:@[e->weights, e->biases, e->depth, e->motion, e->depthTm1, e->dis, e->disLqB, e->nearest, e->kBuf,
                                   e->deriv[0], e->deriv[1], e->history[0], e->history[1], e->dummy]];
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
    nssTune(e, q);
    // post-process compiled for this engine: fixed sizes and quality switches as function constants (30+)
    if (!(getenv("LSFG_NSS_SPECIALIZE") && getenv("LSFG_NSS_SPECIALIZE")[0] == '0')) {
        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        int i2[7][2] = {{e->Hr, e->Wr}, {e->Ho, e->Wo}, {e->Hp / 4, e->Wp / 4}, {e->Hp, e->Wp}, {e->idxModH, e->idxModW},
                        {e->Hq, e->Wq}, {e->half_ ? e->Hp : e->Hr, e->half_ ? e->Wp : e->Wr}};
        const int i2idx[7] = {30, 31, 32, 33, 34, 41, 42};
        for (int k = 0; k < 7; k++) [fc setConstantValue:i2[k] type:MTLDataTypeInt2 atIndex:i2idx[k]];
        const float one = 1.0f;
        int ints[11][2] = {{35, e->taps}, {36, e->kpnCh}, {38, e->KPN.wpad}, {39, e->KPN.C}, {40, e->TEMP[0].wpad},
                           {43, e->half_}, {44, e->half_}, {45, e->quality >= 1}, {46, 1},
                           {47, getenv("LSFG_NSS_POST_ABLATE") ? atoi(getenv("LSFG_NSS_POST_ABLATE")) : 0}, {-1, 0}};
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
            const int abl = getenv("LSFG_NSS_PRE_ABLATE") ? atoi(getenv("LSFG_NSS_PRE_ABLATE")) : 0;
            [hc setConstantValue:&abl type:MTLDataTypeInt atIndex:68];
            NSError *herr = nil;
            id<MTLFunction> hf = [e->lib newFunctionWithName:@"nss_pre_half" constantValues:hc error:&herr];
            id<MTLComputePipelineState> hp = hf ? [dev newComputePipelineStateWithFunction:hf error:&herr] : nil;
            if (hp) {
                e->preHalf = hp;
                @synchronized(NSNull.null) { [psoNames setObject:@"pre_half" forKey:hp]; }
            } else lsfg_log(@"nss: specializing pre-process: %@", herr.localizedDescription);
        }
        id<MTLFunction> f = [e->lib newFunctionWithName:@"nss_post" constantValues:fc error:&ferr];
        id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:&ferr] : nil;
        if (ps) {
            e->post = ps;
            @synchronized(NSNull.null) { [psoNames setObject:@"post" forKey:ps]; }
        } else lsfg_log(@"nss: specializing post-process: %@", ferr.localizedDescription);
    }
    lsfg_log(@"nss: pipeline limits (max threads per group / SIMD width): post %lu/%lu, pre_half %lu, conv32p4 %lu",
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
                    lsfg_log(@"%@", line);
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
    [c setComputePipelineState:special && px == e->layerPX[k.li] ? special : (k.upc ? e->upconv : e->conv)[@(l.ocp * 10 + px)]];
    [c setBytes:&m length:sizeof(m) atIndex:0];
    [c setBuffer:k.in.b offset:0 atIndex:1];
    [c setBuffer:e->weights offset:l.wOff * 2 atIndex:2];
    [c setBuffer:e->biases offset:l.bOff * 4 atIndex:3];
    [c setBuffer:k.out.b offset:0 atIndex:4];
    [c setThreadgroupMemoryLength:4 * 8 * l.ocp * 4 atIndex:0];
    [c dispatchThreadgroups:MTLSizeMake((w + 32 * px - 1) / (32 * px), h, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// Picks each layer's pixel tiles per simdgroup (1, 2 or 4) by timing them on this GPU at this size.
// Picks each layer's kernel: pixel tiles per simdgroup (1, 2 or 4), compiled with the layer's shape as function
// constants (fixed loop counts and strides), by timing all three on this GPU at this size.
static void nssTune(NssEngine *e, id<MTLCommandQueue> q) {
    NssConv convs[14];
    nssConvs(e, 0, convs);
    const BOOL specialize = !(getenv("LSFG_NSS_SPECIALIZE") && getenv("LSFG_NSS_SPECIALIZE")[0] == '0');
    NssConv *cv = convs;   // blocks can't capture C arrays
    // variants per layer: 1, 2, 4 pixel tiles
    NSMutableArray *cands = [NSMutableArray arrayWithCapacity:42];   // [layer * 3 + variant], NSNull if missing
    for (int n = 0; n < 42; n++) [cands addObject:NSNull.null];
    CFTimeInterval t0 = CACurrentMediaTime();
    if (specialize) {
        NSLock *lock = [NSLock new];
        dispatch_apply(42, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
            const int i = (int)n / 3, px = 1 << (n % 3);
            const NssConv k = cv[i];
            const NssLayer l = e->layers[i];
            MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
            int shapeVals[9] = {l.icp, k.in.C, k.inCoff, k.out.C, k.outCoff, k.stride, k.act, k.in.wpad, k.out.wpad};
            for (int j = 0; j < 9; j++) [fc setConstantValue:&shapeVals[j] type:MTLDataTypeInt atIndex:10 + j];
            NSError *err = nil;
            NSString *name = [NSString stringWithFormat:@"nss_%@conv_mma_%d_p%d", k.upc ? @"up" : @"", l.ocp, px];
            id<MTLFunction> f = [e->lib newFunctionWithName:name constantValues:fc error:&err];
            id<MTLComputePipelineState> ps = f ? [e->dev newComputePipelineStateWithFunction:f error:&err] : nil;
            if (!ps) lsfg_log(@"nss: specializing layer %d: %@", i, err.localizedDescription);
            [lock lock];
            if (ps) cands[n] = ps;
            [lock unlock];
        });
    }
    const double compileMs = (CACurrentMediaTime() - t0) * 1e3;
    id<MTLComputePipelineState> (^cand)(int, int) = ^id<MTLComputePipelineState>(int i, int v) {
        id x = cands[i * 3 + v];
        return x == NSNull.null ? nil : x;
    };
    NSMutableString *picked = [NSMutableString string];
    for (int i = 0; i < 14; i++) {
        double best = 1e9;
        int bestPx = 1;
        id<MTLComputePipelineState> bestPso = nil;
        for (int round = 0; round < 2; round++)          // round 0 warms up
            for (int v = 0; v < 3; v++) {
                const int px = 1 << v;
                e->layerPSO[i] = cand(i, v), e->layerPX[i] = px;   // nssEncodeConv uses layerPSO when px matches
                id<MTLCommandBuffer> cb = [q commandBuffer];
                NssEnc *c = [NssEnc new];
                c->m3 = [cb computeCommandEncoder];
                for (int r = 0; r < 8; r++) nssEncodeConv(e, c, convs[i], px);
                [c->m3 endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double t = cb.GPUEndTime - cb.GPUStartTime;
                if (round == 1 && t < best) best = t, bestPx = px, bestPso = cand(i, v);
                if (round == 1 && getenv("LSFG_NSS_TUNE_LOG"))
                    lsfg_log(@"nss: tune layer %d variant px%d: %.1f us per dispatch", i, px, t * 1e6 / 8);
            }
        e->layerPX[i] = bestPx, e->layerPSO[i] = bestPso;
        if (bestPso) @synchronized(NSNull.null) { [psoNames setObject:[NSString stringWithFormat:@"conv%d", i] forKey:bestPso]; }
        [picked appendFormat:@" %d", bestPx];
    }
    lsfg_log(@"nss: tuned conv layers (pixel tiles:%@), %s, compiled in %.0f ms, total %.0f ms", picked,
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
        if (profileEvery < 0) profileEvery = getenv("LSFG_NSS_PROFILE") ? atoi(getenv("LSFG_NSS_PROFILE")) : 0;
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
    // experiment: threadgroup shape for the big per-pixel passes (LSFG_NSS_TG_POST / _PRE = "WxH")
    void (^gridTG)(id<MTLComputePipelineState>, int, int, const char *) = ^(id<MTLComputePipelineState> s, int w, int h, const char *env) {
        int tw = 16, th = 16;
        if (getenv(env)) sscanf(getenv(env), "%dx%d", &tw, &th);
        [c setComputePipelineState:s];
        [c dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    };

    // game inputs
    struct { int dims[2]; float reset; int pad0 /* Metal aligns float2 to 8 bytes */; float mvScale[2]; int depthReversed; float preExposureInv; int hasExposure; } ga =
        {{Hr, Wr}, resetF, 0, {fr->mvScale[0], fr->mvScale[1]}, fr->depthReversed, fr->preExposure > 0 ? 1.0f / fr->preExposure : 1.0f, fr->exposure != nil};
    [c setBytes:&ga length:sizeof(ga) atIndex:0];
    [c setTexture:fr->color atIndex:0], [c setTexture:fr->depth atIndex:1], [c setTexture:fr->motion atIndex:2];
    [c setTexture:fr->exposure ?: e->dummy atIndex:3];
    [c setBuffer:e->depth offset:0 atIndex:2], [c setBuffer:e->motion offset:0 atIndex:3], [c setBuffer:e->kBuf offset:0 atIndex:4];
    grid(isDepthFormat(fr->depth.pixelFormat) ? e->gprepD : e->gprep, Wr, Hr);

    // Offset LUT for this frame's jitter and the depth scatter are independent chains: on Metal 4 each LUT pass
    // shares a barrier slot with a depth pass (overlapNext), instead of each waiting for the other.
    const int gh = nss_gcd(Hr, Ho), gw = nss_gcd(Wr, Wo);
    struct { float jitter[2], scale[2]; int lo[2], hi[2]; int taps; } la =
        {{fr->jy, fr->jx}, {(float)Ho / Hr, (float)Wo / Wr}, {Hr / gh, Wr / gw}, {Ho / gh, Wo / gw}, e->taps};
    struct { int in[2], out[2]; } sa = {{Hr, Wr}, {Hd, Wd}};
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

    // pre-process
    [c setTexture:fr->color atIndex:0], [c setTexture:e->history[p] atIndex:1];
    [c setBuffer:e->kBuf offset:0 atIndex:12];
    const float ctInv[2] = {1.0f / fr->color.height, 1.0f / fr->color.width};   // (row, col)
    if (e->half_) {
        struct { float d2v[4]; float jitter[2], renderSize[2]; int inDims[2], padDims[2], procDims[2], depthDims[2];
                 float exposure; int inWpad, tempWpad; int pad0 /* Metal: float2 at byte 80 */; float colorTexInv[2]; } ha =
            {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {fr->jy, fr->jx}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hp, Wp}, {Hq, Wq},
             {Hd, Wd}, 1.0f, e->IN.wpad, e->TEMP[0].wpad, 0, {ctInv[0], ctInv[1]}};
        [c setBytes:&ha length:sizeof(ha) atIndex:0];
        id<MTLBuffer> hb[11] = {nil, e->depth, e->motion, nil, e->disLqB, e->deriv[p], e->TEMP[p].b, e->IN.b, e->deriv[1 - p], e->dis, e->nearest};
        for (int i = 0; i < 11; i++) if (hb[i]) [c setBuffer:hb[i] offset:0 atIndex:i + 1];
        gridTG(e->preHalf, Wp, Hp, "LSFG_NSS_TG_PRE");
    } else {
        struct { float d2v[4]; float jitter[2], renderSize[2]; int inDims[2], padDims[2], depthDims[2], outDims[2];
                 float exposure; int inWpad, tempWpad; int pad0 /* Metal: float2 at byte 80 */; float colorTexInv[2]; } ra =
            {{e->d2v[0], e->d2v[1], e->d2v[2], e->d2v[3]}, {fr->jy, fr->jx}, {(float)Hr, (float)Wr}, {Hr, Wr}, {Hp, Wp}, {Hd, Wd},
             {Ho, Wo}, 1.0f, e->IN.wpad, e->TEMP[0].wpad, 0, {ctInv[0], ctInv[1]}};
        [c setBytes:&ra length:sizeof(ra) atIndex:0];
        id<MTLBuffer> pb[11] = {nil, e->depth, e->motion, nil, e->depthTm1, e->deriv[p], e->TEMP[p].b, e->IN.b, e->deriv[1 - p], e->dis, e->nearest};
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
            e->half_, e->half_, e->quality >= 1, 1, getenv("LSFG_NSS_POST_ABLATE") ? atoi(getenv("LSFG_NSS_POST_ABLATE")) : 0};
    [c setBytes:&qa length:sizeof(qa) atIndex:0];
    [c setBuffer:e->KPN.b offset:0 atIndex:3], [c setBuffer:e->TEMP[1 - p].b offset:0 atIndex:4];
    [c setBuffer:e->motion offset:0 atIndex:5], [c setBuffer:e->nearest offset:0 atIndex:6], [c setBuffer:lutB offset:lutOff atIndex:7];
    [c setTexture:e->history[1 - p] atIndex:0], [c setTexture:fr->color atIndex:1], [c setTexture:e->history[p] atIndex:2];
    [c setTexture:fr->output atIndex:3], [c setBuffer:e->kBuf offset:0 atIndex:9];
    gridTG(e->post, Wo, Ho, "LSFG_NSS_TG_POST");   // also writes the game's output
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

#pragma mark - MetalFX interception

static const void *kNssKey = &kNssKey;
static atomic_int nssLogged;

// Per scaler: the engine (rebuilt when the quality setting changes) and its command queue for setup.
@interface NssScalerState : NSObject {
@public
    NssEngine *engine;
    int quality;
    BOOL failed, loggedFallback;
    id<MTLCommandQueue> queue;
    uint64_t frames;
    int lastW, lastH, sizeChanges, rebuilds;
    CFTimeInterval firstRebuild;
}
@end
@implementation NssScalerState
@end

static float envFloat(const char *name, float fallback) {
    const char *v = getenv(name);
    return v ? (float)atof(v) : fallback;
}

static void nssDescribe(id<MTLFXTemporalScalerBase> s, id cb, const char *what) {
    lsfg_log(@"nss: %s scaler %s %lux%lu -> %lux%lu, content %lux%lu, color %lu %@ %lux%lu, depth %lu %@, motion %lu %@ %lux%lu, "
             "output %lu %@ usage 0x%lx, exposure %@, preExposure %.3f, jitter %.3f %.3f, mv scale %.3f %.3f, depth reversed %d, reset %d, cb %s",
             what, class_getName(object_getClass(s)), (unsigned long)s.inputWidth, (unsigned long)s.inputHeight,
             (unsigned long)s.outputWidth, (unsigned long)s.outputHeight,
             (unsigned long)s.inputContentWidth, (unsigned long)s.inputContentHeight,
             (unsigned long)s.colorTexture.pixelFormat, s.colorTexture ? @"" : @"(none)", (unsigned long)s.colorTexture.width, (unsigned long)s.colorTexture.height,
             (unsigned long)s.depthTexture.pixelFormat, s.depthTexture ? @"" : @"(none)",
             (unsigned long)s.motionTexture.pixelFormat, s.motionTexture ? @"" : @"(none)", (unsigned long)s.motionTexture.width, (unsigned long)s.motionTexture.height,
             (unsigned long)s.outputTexture.pixelFormat, s.outputTexture ? @"" : @"(none)", (unsigned long)s.outputTexture.usage,
             s.exposureTexture ? [NSString stringWithFormat:@"%lux%lu fmt %lu", (unsigned long)s.exposureTexture.width, (unsigned long)s.exposureTexture.height, (unsigned long)s.exposureTexture.pixelFormat] : @"none",
             s.preExposure, s.jitterOffsetX, s.jitterOffsetY, s.motionVectorScaleX, s.motionVectorScaleY, s.isDepthReversed, s.reset,
             cb ? class_getName(object_getClass(cb)) : "?");
}

// Returns YES when NSS handled the frame.
static BOOL nssTry(id<MTLFXTemporalScalerBase> s, id cbObj) {
    NssScalerState *st = objc_getAssociatedObject(s, kNssKey);
    if (!st) {
        st = [NssScalerState new];
        objc_setAssociatedObject(s, kNssKey, st, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    // The render size the game actually uses (DLSS mode, dynamic resolution), whichever upscaler runs.
    const int cw = (int)(s.inputContentWidth ?: s.inputWidth), ch = (int)(s.inputContentHeight ?: s.inputHeight);
    if (cw != st->lastW || ch != st->lastH) {
        if (st->sizeChanges++ < 20)
            lsfg_log(@"nss: scaler %p renders %dx%d -> %lux%lu (%.2fx)%s", (__bridge void *)s, cw, ch, (unsigned long)s.outputWidth,
                     (unsigned long)s.outputHeight, ch ? (double)s.outputHeight / ch : 0.0, st->sizeChanges == 20 ? " (further changes not logged)" : "");
        st->lastW = cw, st->lastH = ch;
    }
    if (config.upscaler != 1) return NO;
    if (st->failed) return NO;
    BOOL metal4 = NO;
    if (@available(macOS 26.0, *)) metal4 = [cbObj conformsToProtocol:@protocol(MTL4CommandBuffer)];
    if (!metal4 && ![cbObj conformsToProtocol:@protocol(MTLCommandBuffer)]) {
        if (!st->loggedFallback) lsfg_log(@"nss: %s is not a command buffer we know; MetalFX runs instead", class_getName(object_getClass(cbObj)));
        st->loggedFallback = YES;
        return NO;
    }
    id<MTLDevice> device = [(id<MTLCommandBuffer>)cbObj device];
    const int wr = (int)(s.inputContentWidth ?: s.inputWidth), hr = (int)(s.inputContentHeight ?: s.inputHeight);
    const int wo = (int)s.outputWidth, ho = (int)s.outputHeight;
    if (!s.colorTexture || !s.depthTexture || !s.motionTexture || !s.outputTexture || wo <= wr || ho <= hr ||
        !(s.outputTexture.usage & MTLTextureUsageShaderWrite) || s.motionTexture.width != s.colorTexture.width) {
        if (!st->loggedFallback) { nssDescribe(s, cbObj, "unsupported"); lsfg_log(@"nss: unsupported inputs; MetalFX runs instead"); }
        st->loggedFallback = YES;
        return NO;
    }
    const int quality = config.nssQuality;
    if (!st->engine || st->quality != quality || st->engine->Wr != wr || st->engine->Hr != hr || st->engine->Wo != wo || st->engine->Ho != ho) {
        // Dynamic resolution would rebuild every few frames (a stall each time): hand those scalers back to MetalFX.
        const CFTimeInterval now = CACurrentMediaTime();
        if (now - st->firstRebuild > 10) st->firstRebuild = now, st->rebuilds = 0;
        if (++st->rebuilds > 4) {
            st->failed = YES;
            lsfg_log(@"nss: the render size keeps changing (dynamic resolution?); MetalFX runs for this scaler");
            return NO;
        }
        if (!st->queue) st->queue = [device newCommandQueue];
        CFTimeInterval t0 = CACurrentMediaTime();
        st->engine = nssCreate(device, st->queue, wr, hr, wo, ho, quality,
                               envFloat("LSFG_NSS_NEAR", 0.1f), envFloat("LSFG_NSS_FAR", 1000.f), envFloat("LSFG_NSS_FOV", 1.0f));
        st->quality = quality;
        if (!st->engine) { st->failed = YES; lsfg_log(@"nss: setup failed; MetalFX runs instead"); return NO; }
        lsfg_log(@"nss: %s quality, %dx%d -> %dx%d, network input %dx%d, set up in %.0f ms",
                 quality == 0 ? "low" : quality == 1 ? "mid" : "high", wr, hr, wo, ho, st->engine->Wp, st->engine->Hp,
                 (CACurrentMediaTime() - t0) * 1e3);
    }
    // MetalFX's jitterOffset is the negated sample offset (calibrated in the bench); NSS wants the offset as (row, col).
    const float jSign = envFloat("LSFG_NSS_JSIGN", 1.0f), mvSign = envFloat("LSFG_NSS_MVSIGN", 1.0f);
    NssFrame fr = {s.colorTexture, s.depthTexture, s.motionTexture, s.exposureTexture, s.outputTexture,
                   -jSign * s.jitterOffsetY, -jSign * s.jitterOffsetX,
                   {mvSign * s.motionVectorScaleX, mvSign * s.motionVectorScaleY}, s.isDepthReversed, s.reset, s.preExposure};
    if (atomic_fetch_add(&nssLogged, 1) < 3) nssDescribe(s, cbObj, metal4 ? "frame (Metal 4)" : "frame");
    nssEncode(st->engine, cbObj, metal4, s.fence, &fr);
    const char *dump = getenv("LSFG_NSS_DUMP");   // debug: the first frame's intermediates as raw files
    if (dump && st->frames == 0) {
        NssEngine *e = st->engine;
        id<MTLCommandQueue> q = st->queue;
        NSString *dir = @(dump);
        NSDictionary<NSString *, id<MTLBuffer>> *bufs = @{@"depth": e->depth, @"motion": e->motion, @"k": e->kBuf, @"in": e->IN.b,
            @"a0": e->A0.b, @"kpn": e->KPN.b, @"temporal": e->TEMP[1 - e->p].b, @"deriv": e->deriv[1 - e->p], @"dislq": e->disLqB,
            @"nearest": e->nearest};
        if (!metal4) [(id<MTLCommandBuffer>)cbObj addCompletedHandler:^(id<MTLCommandBuffer> done) {
            for (NSString *name in bufs) {
                id<MTLBuffer> b = bufs[name];
                id<MTLBuffer> h = [b.device newBufferWithLength:b.length options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> c2 = [q commandBuffer];
                id<MTLBlitCommandEncoder> be = [c2 blitCommandEncoder];
                [be copyFromBuffer:b sourceOffset:0 toBuffer:h destinationOffset:0 size:b.length];
                [be endEncoding];
                [c2 commit];
                [c2 waitUntilCompleted];
                [[NSData dataWithBytes:h.contents length:h.length] writeToFile:[dir stringByAppendingPathComponent:name] atomically:NO];
            }
            lsfg_log(@"nss: dumped first-frame buffers to %@ (Hp %d Wp %d, wpad in %d a0 %d kpn %d temp %d)", dir, e->Hp, e->Wp,
                     e->IN.wpad, e->A0.wpad, e->KPN.wpad, e->TEMP[0].wpad);
        }];
    }
    st->frames++;
    return YES;
}

#pragma mark - Upscale GPU time

// GPU time of each upscale, MetalFX or NSS alike: Metal 4 command-buffer timestamps just before and just after the
// call (a timestamp is taken once all earlier work in the command buffer has finished). Read back 16 frames later
// from a 32-frame ring; the per-second log line gets the median and p95 per upscaler.
enum { kTsFrames = 32, kTsDelay = 16 };
static os_unfair_lock tsLock = OS_UNFAIR_LOCK_INIT;
static id tsHeap;   // id<MTL4CounterHeap>
static double tsTicksPerMs;
static uint64_t tsFrame;
static int tsKind[kTsFrames];               // 0 MetalFX, 1 NSS
static NSMutableArray<NSNumber *> *tsSamples[2];
static atomic_long nssFramesThisSecond;
static NSMutableArray<NSNumber *> *tsCpu[2];   // CPU ms per upscale call on the render thread: 0 MetalFX, 1 NSS

static void tsCollect(int slot) {
    if (@available(macOS 26.0, *)) {
        NSData *d = [(id<MTL4CounterHeap>)tsHeap resolveCounterRange:NSMakeRange(2 * slot, 2)];
        if (d.length < 2 * sizeof(MTL4TimestampHeapEntry)) return;
        const MTL4TimestampHeapEntry *t = d.bytes;
        if (!t[0].timestamp || t[1].timestamp <= t[0].timestamp) return;
        const double ms = (double)(t[1].timestamp - t[0].timestamp) / tsTicksPerMs;
        if (ms > 0 && ms < 100) [tsSamples[tsKind[slot]] addObject:@(ms)];
        [(id<MTL4CounterHeap>)tsHeap invalidateCounterRange:NSMakeRange(2 * slot, 2)];
    }
}

// Wraps one upscale call. Returns the slot to close with tsAfter, or -1 (no timing for this command buffer).
static int tsBefore(id cbObj) {
    if (@available(macOS 26.0, *)) {
        if (![cbObj conformsToProtocol:@protocol(MTL4CommandBuffer)]) return -1;
        id<MTL4CommandBuffer> cb = cbObj;
        os_unfair_lock_lock(&tsLock);
        if (!tsHeap) {
            id<MTLDevice> dev = cb.device;
            MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new];
            hd.type = MTL4CounterHeapTypeTimestamp, hd.count = 2 * kTsFrames;
            tsHeap = [dev newCounterHeapWithDescriptor:hd error:nil];
            tsTicksPerMs = (double)[dev queryTimestampFrequency] / 1000.0;
            tsSamples[0] = [NSMutableArray array], tsSamples[1] = [NSMutableArray array];
            lsfg_log(@"nss: timing upscales with GPU timestamps (%.0f MHz)", tsTicksPerMs / 1000.0);
        }
        const int slot = (int)(tsFrame % kTsFrames);
        if (tsFrame >= kTsDelay) tsCollect((int)((tsFrame - kTsDelay) % kTsFrames));
        tsFrame++;
        os_unfair_lock_unlock(&tsLock);
        if (!tsHeap) return -1;
        [cb writeTimestampIntoHeap:tsHeap atIndex:2 * slot];
        return slot;
    }
    return -1;
}
static void tsAfter(id cbObj, int slot, BOOL nss) {
    if (slot < 0) return;
    if (@available(macOS 26.0, *)) {
        tsKind[slot] = nss ? 1 : 0;
        [(id<MTL4CommandBuffer>)cbObj writeTimestampIntoHeap:tsHeap atIndex:2 * slot + 1];
    }
}

// " | upscale GPU NSS mid 2.31 ms (p95 2.60, 60 frames)" for the per-second log line; resets the window.
static NSString *nssStatsLine(void) {
    NSMutableString *out = [NSMutableString string];
    os_unfair_lock_lock(&tsLock);
    for (int k = 0; k < 2; k++) {
        NSMutableArray<NSNumber *> *v = tsSamples[k];
        if (!v.count) continue;
        [v sortUsingSelector:@selector(compare:)];
        const double med = v[v.count / 2].doubleValue, p95 = v[MIN(v.count - 1, (NSUInteger)(v.count * 0.95))].doubleValue;
        const char *q = config.nssQuality == 0 ? "low" : config.nssQuality == 2 ? "high" : "mid";
        [out appendFormat:@" | upscale GPU %s%s%s %.2f ms (p95 %.2f, %lu frames)", k ? "NSS" : "MetalFX", k ? " " : "", k ? q : "",
                          med, p95, (unsigned long)v.count];
        [v removeAllObjects];
    }
    for (int k = 0; k < 2; k++) {
        NSMutableArray<NSNumber *> *v = tsCpu[k];
        if (!v.count) continue;
        [v sortUsingSelector:@selector(compare:)];
        [out appendFormat:@", %s CPU %.3f ms (p95 %.3f)", k ? "NSS" : "MetalFX", v[v.count / 2].doubleValue,
                          v[MIN(v.count - 1, (NSUInteger)(v.count * 0.95))].doubleValue];
        [v removeAllObjects];
    }
    os_unfair_lock_unlock(&tsLock);
    const long n = atomic_exchange(&nssFramesThisSecond, 0);
    if (n && !out.length) [out appendFormat:@" | NSS upscaled %ld frames", n];
    return out;
}

static void nssHookScalerClass(Class cls) {
    static NSMutableSet *done;
    @synchronized(NSNull.null) {
        if (!done) done = [NSMutableSet set];
        NSString *name = @(class_getName(cls));
        if ([done containsObject:name]) return;
        [done addObject:name];
    }
    SEL sel = @selector(encodeToCommandBuffer:);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) { lsfg_log(@"nss: %s has no encodeToCommandBuffer:", class_getName(cls)); return; }
    IMP orig = method_getImplementation(m);
    method_setImplementation(m, imp_implementationWithBlock(^(id self, id cb) {
        const int slot = tsBefore(cb);
        BOOL nss;
        const CFTimeInterval c0 = CACurrentMediaTime();
        @autoreleasepool {
            nss = nssTry(self, cb);
        }
        if (nss) atomic_fetch_add(&nssFramesThisSecond, 1);
        else ((void (*)(id, SEL, id))orig)(self, sel, cb);
        // CPU time of the whole upscale call (NSS, or the decision to pass on plus MetalFX's own encode)
        const double cpuMs = (CACurrentMediaTime() - c0) * 1e3;
        os_unfair_lock_lock(&tsLock);
        if (!tsCpu[nss]) tsCpu[nss] = [NSMutableArray array];
        [tsCpu[nss] addObject:@(cpuMs)];
        os_unfair_lock_unlock(&tsLock);
        tsAfter(cb, slot, nss);
    }));
    lsfg_log(@"nss: hooked -[%s encodeToCommandBuffer:]", class_getName(cls));
}

// The temporal scaler classes are private; catch them where they're made.
static void nssInstall(void) {
    Class d = objc_getClass("MTLFXTemporalScalerDescriptor");
    if (!d) { lsfg_log(@"nss: MetalFX not available"); return; }
    SEL s1 = @selector(newTemporalScalerWithDevice:);
    Method m1 = class_getInstanceMethod(d, s1);
    if (m1) {
        IMP o1 = method_getImplementation(m1);
        method_setImplementation(m1, imp_implementationWithBlock(^id(MTLFXTemporalScalerDescriptor *self, id dev) {
            id sc = ((id (*)(id, SEL, id))o1)(self, s1, dev);
            lsfg_log(@"nss: game made a MetalFX temporal scaler (Metal 3) %lux%lu -> %lux%lu, color %lu depth %lu motion %lu output %lu, auto exposure %d, content properties %d",
                     (unsigned long)self.inputWidth, (unsigned long)self.inputHeight, (unsigned long)self.outputWidth, (unsigned long)self.outputHeight,
                     (unsigned long)self.colorTextureFormat, (unsigned long)self.depthTextureFormat, (unsigned long)self.motionTextureFormat,
                     (unsigned long)self.outputTextureFormat, self.isAutoExposureEnabled, self.isInputContentPropertiesEnabled);
            if (sc) nssHookScalerClass(object_getClass(sc));
            return sc;
        }));
    }
    SEL s2 = NSSelectorFromString(@"newTemporalScalerWithDevice:compiler:");
    Method m2 = class_getInstanceMethod(d, s2);
    if (m2) {
        IMP o2 = method_getImplementation(m2);
        method_setImplementation(m2, imp_implementationWithBlock(^id(MTLFXTemporalScalerDescriptor *self, id dev, id compiler) {
            id sc = ((id (*)(id, SEL, id, id))o2)(self, s2, dev, compiler);
            lsfg_log(@"nss: game made a MetalFX temporal scaler (Metal 4) %lux%lu -> %lux%lu, color %lu depth %lu motion %lu output %lu, auto exposure %d, content properties %d (scale %.2f-%.2f)",
                     (unsigned long)self.inputWidth, (unsigned long)self.inputHeight, (unsigned long)self.outputWidth, (unsigned long)self.outputHeight,
                     (unsigned long)self.colorTextureFormat, (unsigned long)self.depthTextureFormat, (unsigned long)self.motionTextureFormat,
                     (unsigned long)self.outputTextureFormat, self.isAutoExposureEnabled, self.isInputContentPropertiesEnabled,
                     self.inputContentMinScale, self.inputContentMaxScale);
            if (sc) nssHookScalerClass(object_getClass(sc));
            return sc;
        }));
    }
    lsfg_log(@"nss: watching for MetalFX temporal scalers (upscaler %s)", config.upscaler == 1 ? "NSS" : "MetalFX");
}
