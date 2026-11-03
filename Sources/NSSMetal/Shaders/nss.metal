// Arm Neural Super Sampling v1 ("high" quality path) on Metal.
//
// Port of the Apache-2.0 Slang reference in arm/neural-graphics-model-gym
// (usecases/nss/model/shaders/nss_v1/*.slang, common/model/shaders/*.slang);
// the depth-clip math there adapts AMD FidelityFX FSR2 (MIT).
//
// Layout matches the reference tensors: planar float buffers [channel][row][col], every coordinate is
// (row, col) and every uv is (v, u) = (row, col) / dims. Motion is (row, col) in input pixels, current - previous.
#include <metal_stdlib>
using namespace metal;

#define EPS 1e-7f
#define MAX_HALF 65504.0f

// ------------------------------------------------------------------------------------------------ tensor helpers
static inline float LD(const device float* t, int2 dims, int c, int2 p)
{
    return t[((long)c * dims.x + p.x) * dims.y + p.y];
}
static inline void ST(device float* t, int2 dims, int c, int2 p, float v)
{
    t[((long)c * dims.x + p.x) * dims.y + p.y] = v;
}

// Network tensors: channels-last [row+1][col+1][C] with a zero border (wpad = padded row length).
struct HWC { int2 dims; int wpad; int C; };
static inline float LDH(const device half* t, HWC h, int c, int2 p)
{
    return float(t[(((long)p.x + 1) * h.wpad + p.y + 1) * h.C + c]);
}

// A 4-channel padded tensor read through an RGBA16F texture view (texel (col+1, row+1) = pixel (row, col)), so the
// bilinear filter is done in hardware. clampEdge clamps to the edge pixels, otherwise the zero border gives zeros.
constexpr sampler kViewLin(filter::linear, address::clamp_to_zero, coord::pixel);
static inline float4 sampleView(texture2d<float> v, int2 dims, float2 uv, bool clampEdge)
{
    float2 p = uv * float2(dims);
    if (clampEdge) p = clamp(p, 0.5f, float2(dims) - 0.5f);
    return v.sample(kViewLin, (p + 1.0f).yx);
}

// sample_tensor: bilinear, texel centres at +0.5, clamp-to-edge or zero outside
static float sampleT(const device float* t, int2 dims, int c, float2 uv, bool clampEdge)
{
    const float2 idx = uv * float2(dims);
    const int2 g0 = int2(floor(idx - 0.5f));
    const int2 g1 = g0 + 1;
    float2 w0 = max(1.0f - abs(float2(g0) + 0.5f - idx), 0.0f);
    float2 w1 = max(1.0f - abs(float2(g1) + 0.5f - idx), 0.0f);
    if (!clampEdge)
    {
        w0 *= float2(g0 >= 0) * float2(g0 < dims);
        w1 *= float2(g1 >= 0) * float2(g1 < dims);
    }
    const int2 c0 = clamp(g0, int2(0), dims - 1), c1 = clamp(g1, int2(0), dims - 1);
    return LD(t, dims, c, int2(c0.x, c0.y)) * w0.x * w0.y + LD(t, dims, c, int2(c0.x, c1.y)) * w0.x * w1.y +
           LD(t, dims, c, int2(c1.x, c0.y)) * w1.x * w0.y + LD(t, dims, c, int2(c1.x, c1.y)) * w1.x * w1.y;
}
static float3 sampleT3(const device float* t, int2 dims, float2 uv, bool clampEdge)
{
    return float3(sampleT(t, dims, 0, uv, clampEdge), sampleT(t, dims, 1, uv, clampEdge), sampleT(t, dims, 2, uv, clampEdge));
}
static float4 sampleT4(const device float* t, int2 dims, float2 uv, bool clampEdge)
{
    return float4(sampleT(t, dims, 0, uv, clampEdge), sampleT(t, dims, 1, uv, clampEdge),
                  sampleT(t, dims, 2, uv, clampEdge), sampleT(t, dims, 3, uv, clampEdge));
}

// sample_tensor on an interleaved float4 tensor (one load per tap instead of four)
static float4 sampleV4(const device float4* t, int2 dims, float2 uv, bool clampEdge)
{
    const float2 idx = uv * float2(dims);
    const int2 g0 = int2(floor(idx - 0.5f));
    const int2 g1 = g0 + 1;
    float2 w0 = max(1.0f - abs(float2(g0) + 0.5f - idx), 0.0f);
    float2 w1 = max(1.0f - abs(float2(g1) + 0.5f - idx), 0.0f);
    if (!clampEdge)
    {
        w0 *= float2(g0 >= 0) * float2(g0 < dims);
        w1 *= float2(g1 >= 0) * float2(g1 < dims);
    }
    const int2 c0 = clamp(g0, int2(0), dims - 1), c1 = clamp(g1, int2(0), dims - 1);
    return t[c0.x * dims.y + c0.y] * (w0.x * w0.y) + t[c0.x * dims.y + c1.y] * (w0.x * w1.y) +
           t[c1.x * dims.y + c0.y] * (w1.x * w0.y) + t[c1.x * dims.y + c1.y] * (w1.x * w1.y);
}

// texture_gather (clamp to edge): (tl, tr, bl, br) = (g0,g0), (g0,g1), (g1,g0), (g1,g1)
static float4 gatherT(const device float* t, int2 dims, int c, float2 uv)
{
    const float2 idx = uv * float2(dims);
    const int2 g0 = int2(floor(idx - 0.5f)), g1 = g0 + 1;
    const int2 a = clamp(g0, int2(0), dims - 1), b = clamp(g1, int2(0), dims - 1);
    return float4(LD(t, dims, c, int2(a.x, a.y)), LD(t, dims, c, int2(a.x, b.y)),
                  LD(t, dims, c, int2(b.x, a.y)), LD(t, dims, c, int2(b.x, b.y)));
}

// Motion is interleaved float2 (row, col) per pixel.
static float2 sampleM(const device float2* t, int2 dims, float2 uv)   // sample_tensor, clamp to edge
{
    const float2 idx = uv * float2(dims);
    const int2 g0 = int2(floor(idx - 0.5f)), g1 = g0 + 1;
    const float2 w0 = max(1.0f - abs(float2(g0) + 0.5f - idx), 0.0f), w1 = max(1.0f - abs(float2(g1) + 0.5f - idx), 0.0f);
    const int2 c0 = clamp(g0, int2(0), dims - 1), c1 = clamp(g1, int2(0), dims - 1);
    return t[c0.x * dims.y + c0.y] * (w0.x * w0.y) + t[c0.x * dims.y + c1.y] * (w0.x * w1.y) +
           t[c1.x * dims.y + c0.y] * (w1.x * w0.y) + t[c1.x * dims.y + c1.y] * (w1.x * w1.y);
}
static void gatherM(const device float2* t, int2 dims, float2 uv, thread float4& y, thread float4& x)   // texture_gather
{
    const float2 idx = uv * float2(dims);
    const int2 g0 = int2(floor(idx - 0.5f)), g1 = g0 + 1;
    const int2 a = clamp(g0, int2(0), dims - 1), b = clamp(g1, int2(0), dims - 1);
    const float2 tl = t[a.x * dims.y + a.y], tr = t[a.x * dims.y + b.y], bl = t[b.x * dims.y + a.y], br = t[b.x * dims.y + b.y];
    y = float4(tl.x, tr.x, bl.x, br.x), x = float4(tl.y, tr.y, bl.y, br.y);
}

static inline float3 karis(float3 x)
{
    x = max(x, 0.0f);
    return clamp(x / (1.0f + max(max(x.r, x.g), x.b)), 0.0f, 1.0f);
}
static inline float3 karisInverse(float3 x)
{
    const float lim = MAX_HALF / (1.0f + MAX_HALF);
    x = clamp(max(x, 0.0f), 0.0f, lim);
    return x / (1.0f - max(max(x.r, x.g), x.b));
}
constexpr sampler kLinClamp(filter::linear, address::clamp_to_edge);
constexpr sampler kLinZero(filter::linear, address::clamp_to_zero);
// textures are addressed (x, y) = (col, row)
static inline float3 colorAt(texture2d<float> t, int2 p) { return t.read(uint2(p.y, p.x)).rgb; }

static inline bool onScreen(int2 p, int2 size) { return all(p < size) && all(p >= 0); }

// ------------------------------------------------------------------------------------------------ inputs
// Game textures -> reference tensors. depth: reversed-Z in, forward-Z out. motion: (x, y) render px,
// previous - current in; (row, col) current - previous out. reset zeroes motion (first frame of a sequence).
struct PrepArgs { int2 dims; float reset; };
kernel void nss_prep(constant PrepArgs& A [[buffer(0)]],
                     texture2d<float> color [[texture(0)]], texture2d<float> depth [[texture(1)]],
                     texture2d<float> motion [[texture(2)]],
                     device float* oColor [[buffer(1)]], device float* oDepth [[buffer(2)]], device float2* oMotion [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const int2 p = int2(gid.y, gid.x);
    if (any(p >= A.dims)) return;
    const float3 c = color.read(gid).rgb;
    ST(oColor, A.dims, 0, p, c.r), ST(oColor, A.dims, 1, p, c.g), ST(oColor, A.dims, 2, p, c.b);
    ST(oDepth, A.dims, 0, p, 1.0f - depth.read(gid).r);
    const float2 m = motion.read(gid).rg * (1.0f - A.reset);
    oMotion[p.x * A.dims.y + p.y] = float2(-m.y, -m.x);
}

// Game (MetalFX) inputs -> NSS inputs, inside the hook. Reads the content region at the top-left of each texture.
//   color: scaled by k = exposure / preExposure into our own RGBA16F texture (NSS then runs at exposure 1;
//          nss_output divides by k again).
//   depth: forward Z (1 - d when the game's depth is reversed).
//   motion: texel * mvScale = (previous - current) in input pixels, MetalFX's convention; NSS wants
//          (row, col) current - previous. Zeroed on reset.
struct GamePrepArgs { int2 dims; float reset; float2 mvScale; int depthReversed; float preExposureInv; int hasExposure; };

static void gamePrep(constant GamePrepArgs& A, float4 c, float d, float2 m, float k, uint2 gid,
                     texture2d<float, access::write> oColor, device float* oDepth, device float2* oMotion)
{
    const int2 p = int2(gid.y, gid.x);
    ST(oDepth, A.dims, 0, p, A.depthReversed ? 1.0f - d : d);
    m *= A.mvScale * (1.0f - A.reset);
    oMotion[p.x * A.dims.y + p.y] = float2(-m.y, -m.x);
}
static inline float exposureK(constant GamePrepArgs& A, texture2d<float> e)
{
    return (A.hasExposure ? max(e.read(uint2(0, 0)).r, 1e-6f) : 1.0f) * A.preExposureInv;
}

kernel void nss_game_prep(constant GamePrepArgs& A [[buffer(0)]],
                          texture2d<float> color [[texture(0)]], texture2d<float> depth [[texture(1)]],
                          texture2d<float> motion [[texture(2)]], texture2d<float> exposure [[texture(3)]],
                          texture2d<float, access::write> oColor [[texture(4)]],
                          device float* oDepth [[buffer(2)]], device float2* oMotion [[buffer(3)]], device float* oK [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (any(int2(gid.y, gid.x) >= A.dims)) return;
    const float k = exposureK(A, exposure);
    if (gid.x == 0 && gid.y == 0) oK[0] = k;
    gamePrep(A, color.read(gid), depth.read(gid).r, motion.read(gid).rg, k, gid, oColor, oDepth, oMotion);
}

kernel void nss_game_prep_d(constant GamePrepArgs& A [[buffer(0)]],
                            texture2d<float> color [[texture(0)]], depth2d<float> depth [[texture(1)]],
                            texture2d<float> motion [[texture(2)]], texture2d<float> exposure [[texture(3)]],
                            texture2d<float, access::write> oColor [[texture(4)]],
                            device float* oDepth [[buffer(2)]], device float2* oMotion [[buffer(3)]], device float* oK [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (any(int2(gid.y, gid.x) >= A.dims)) return;
    const float k = exposureK(A, exposure);
    if (gid.x == 0 && gid.y == 0) oK[0] = k;
    gamePrep(A, color.read(gid), depth.read(gid), motion.read(gid).rg, k, gid, oColor, oDepth, oMotion);
}

// ------------------------------------------------------------------------------------------------ offset LUT (GPU)
// model_v1._compute_lut / graphics_utils.generate_lr_to_hr_tile, per frame on the GPU (it depends on the jitter).
// Grid of one reduced tile: hl x wl low-res pixels map to hh x wh high-res pixels.
struct LutArgs { float2 jitter; float2 scale; int2 lo; int2 hi; int taps; };

kernel void nss_lut_clear(constant LutArgs& A [[buffer(0)]], device float4* map [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid < A.hi.x * A.hi.y) map[gid] = 0;
}

// map[hy][hx] = (dy, dx, valid): which low-res pixel (relative offset) lands in each high-res pixel of the tile
kernel void nss_lut_scatter(constant LutArgs& A [[buffer(0)]], device float4* map [[buffer(1)]], uint2 gid [[thread_position_in_grid]])
{
    const int y = gid.y, x = gid.x;
    if (y >= A.lo.x || x >= A.lo.y) return;
    const int hy = int(floor((float(y) + A.jitter.x + 0.5f) * A.scale.x)), hx = int(floor((float(x) + A.jitter.y + 0.5f) * A.scale.y));
    const int ly = int(floor((float(hy) + 0.5f) / A.scale.x)), lx = int(floor((float(hx) + 0.5f) / A.scale.y));
    if (hy < 0 || hy >= A.hi.x || hx < 0 || hx >= A.hi.y) return;
    map[hy * A.hi.y + hx] = float4(float(y - ly), float(x - lx), 1.0f, 0.0f);
}

// per tile: the `taps` highest-weighted valid taps of a 6x6 window (Gaussian weight plus an order bias that breaks ties)
// One uint per tap: low-res offset (dy, dx) and tap position (tdy, tdx) as 4-bit +8 biased fields, valid bit,
// KPN channel (0-35). Invalid taps are all-zero fields.
static inline uint packTap(int dy, int dx, uint v, int tdy, int tdx, int k)
{
    return uint(dy + 8) | (uint(dx + 8) << 4) | (v << 8) | (uint(tdy + 8) << 9) | (uint(tdx + 8) << 13) | (uint(k) << 17);
}

kernel void nss_lut_select(constant LutArgs& A [[buffer(0)]], const device float4* map [[buffer(1)]],
                           constant float* wgt [[buffer(2)]], device uint* lut [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    const int hh = A.hi.x, wh = A.hi.y, tiles = hh * wh;
    if ((int)gid >= tiles) return;
    const int by = gid / wh, bx = gid % wh;
    float score[36];
    int srcI[36];
    for (int k = 0; k < 36; ++k) {
        const int tdx = -2 + k / 6, tdy = -2 + k % 6;
        srcI[k] = (((by + tdy) % hh + hh) % hh) * wh + (((bx + tdx) % wh + wh) % wh);
        score[k] = wgt[k] * map[srcI[k]].z;
    }
    uint64_t taken = 0;
    for (int i = 0; i < A.taps; ++i) {
        int best = -1;
        for (int k = 0; k < 36; ++k)
            if (!((taken >> k) & 1) && (best < 0 || score[k] > score[best])) best = k;
        taken |= uint64_t(1) << best;
        const float4 m = map[srcI[best]];
        const bool v = m.z > 0;
        lut[int(gid) * A.taps + i] = v ? packTap(int(m.x), int(m.y), 1u, -2 + best % 6, -2 + best / 6, best) : packTap(0, 0, 0u, 0, 0, 0);
    }
}

// ------------------------------------------------------------------------------------------------ depth scatter
struct ScatterArgs { int2 inDims; int2 outDims; };
kernel void nss_depth_clear(constant ScatterArgs& A [[buffer(0)]], device int* out [[buffer(1)]], uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x < A.outDims.y && (int)gid.y < A.outDims.x) out[gid.y * A.outDims.y + gid.x] = 0x7fffffff;
}

static void reconstructPrevDepth(device atomic_int* out, int2 outDims, float2 reproj, float depthValue);

kernel void nss_depth_scatter(constant ScatterArgs& A [[buffer(0)]],
                              const device float2* motion [[buffer(1)]], const device float* depth [[buffer(2)]],
                              device atomic_int* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    const int2 idx = int2(gid.y, gid.x);
    if (any(idx >= A.outDims)) return;
    const float2 invOut = 1.0f / float2(A.outDims);
    const float2 uv = (float2(idx) + 0.5f) * invOut;
    const float2 invScale = float2(A.outDims) / float2(A.inDims);
    const float4 dq = gatherT(depth, A.inDims, 0, uv);
    float4 yq, xq;
    gatherM(motion, A.inDims, uv, yq, xq);
    float3 r = float3(yq.x, xq.x, dq.x);
    r = mix(r, float3(yq.y, xq.y, dq.y), step(dq.y, r.z));
    r = mix(r, float3(yq.z, xq.z, dq.z), step(dq.z, r.z));
    r = mix(r, float3(yq.w, xq.w, dq.w), step(dq.w, r.z));
    float2 mv = r.xy * invScale;
    mv *= float(length(mv) > 0.1f);
    mv *= invOut;
    reconstructPrevDepth(out, A.outDims, uv - mv, r.z);
}

static void reconstructPrevDepth(device atomic_int* out, int2 outDims, float2 reproj, float depthValue)
{
    const float2 px = reproj * float2(outDims) - 0.5f;
    const int2 base = int2(floor(px));
    const float2 f = fract(px);
    const int2 offs[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    const float w[4] = {(1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y};
    const int idepth = int(depthValue * float(0x7fffffff));
    for (int i = 0; i < 4; ++i)
    {
        if (w[i] <= 0.1f) continue;
        const int2 sp = base + offs[i];
        if (onScreen(sp, outDims)) atomic_fetch_min_explicit(&out[sp.x * outDims.y + sp.y], idepth, memory_order_relaxed);
    }
}

// ------------------------------------------------------------------------------------------------ pre-process
// Bilinear sample of the game's color at a content uv, clamped to the content region (clamp-to-edge semantics).
static inline float3 sampleGameColor(texture2d<float> color, float2 uv, int2 contentDims, float2 texInv)
{
    const float2 p = clamp(uv * float2(contentDims), 0.5f, float2(contentDims) - 0.5f);
    return color.sample(kLinClamp, (p * texInv).yx).rgb;
}

struct PreArgs
{
    float4 deviceToView;
    float2 jitter;        // (row, col) input pixels
    float2 renderSize;    // (H, W)
    int2 inDims, padDims, depthDims, outDims;
    float exposure;
    int inWpad;      // network input tensor row length (C = 16)
    int tempWpad;    // temporal feedback row length (C = 8)
    float2 colorTexInv;   // 1 / the game's color texture size; Metal puts it at byte 80 (the host pads)
};

static inline float viewDepth(float d, float4 d2v) { return d2v[1] / (d - d2v[0]); }
static inline float3 viewPos(int2 vp, int2 size, float d, float4 d2v)
{
    const float z = viewDepth(d, d2v);
    const float2 ndc = float2(vp) / float2(size) * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
    return float3(d2v[2] * ndc.x * z, d2v[3] * ndc.y * z, z);
}

static float depthClip(const device int* depthTm1, int2 tmDims, float2 uv, float2 renderSize, float currentDepth, float4 d2v, float tol = 0.0f)
{
    const float curView = viewDepth(currentDepth, d2v);
    const float2 px = uv * float2(tmDims) - 0.5f;
    const int2 base = int2(floor(px));
    const float2 f = fract(px);
    const int2 offs[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    const float w[4] = {(1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y};
    float depth = 0, wsum = 0;
    for (int i = 0; i < 4; ++i)
    {
        const int2 sp = base + offs[i];
        const bool on = onScreen(sp, tmDims);
        wsum += on ? 0.f : w[i];
        if (on && w[i] > 0.1f)
        {
            const float prev = float(depthTm1[sp.x * tmDims.y + sp.y]) * (1.0f / float(0x7fffffff));
            const float prevView = viewDepth(prev, d2v);
            const float diff = curView - prevView;
            if (diff > 0.0f)
            {
                const float plane = max(prev, currentDepth);
                const float3 center = viewPos(int2(renderSize * 0.5f), int2(renderSize), plane, d2v);
                const float3 corner = viewPos(int2(0, 0), int2(renderSize), plane, d2v);
                const float halfVp = length(renderSize);
                const float thr = max(curView, prevView);
                const float kFov = length(corner) / length(center);
                const float req = 1.37e-05f * kFov * halfVp * thr + tol;
                const float resF = saturate(length(renderSize) / length(float2(1080.0f, 1920.0f)));
                const float power = mix(1.0f, 3.0f, resF);
                depth += pow(saturate(req / diff), power) * w[i];
                wsum += w[i];
            }
        }
    }
    return wsum > 0 ? saturate(1.0f - depth / wsum) : 0.0f;
}

static inline int reflectIdx(int c, int size) { return c < 0 ? -c - 1 : (c >= size ? 2 * size - c - 1 : c); }

static inline float3 ycocg(float3 rgb)
{
    const float co = rgb.r - rgb.b, t = rgb.b + co * 0.5f, cg = rgb.g - t;
    return float3(t + cg * 0.5f, co, cg);
}
static inline float3 derivColor(texture2d<float> color, int2 dims, int2 p, float exposure)
{
    const int2 s = int2(reflectIdx(p.x, dims.x), reflectIdx(p.y, dims.y));
    const float3 c = max(colorAt(color, s) * exposure, 0.0f);
    return sqrt(c);
}
static inline float dDelta(float3 a, float3 b, float cw) { return length(float3(a.x - b.x, (a.y - b.y) * cw, (a.z - b.z) * cw)); }

struct LowMid
{
    bool on;
    texture2d<float> derivTm1;   // hardware-filtered view of the previous derivative state
    int2 derivDims;
    float2 derivUv, derivStep;      // uv and one-texel step in the derivative tensor
    int2 inputStep;
};

static float4 lumaDerivative(texture2d<float> color, int2 dims, int2 ref, float exposure, float4 tm1, float dis, thread float& instability,
                             LowMid lm)
{
    const float cw = 1.25f;
    const float3 c = ycocg(derivColor(color, dims, ref, exposure));
    const float3 n = ycocg(derivColor(color, dims, ref + int2(0, -1), exposure));
    const float3 s = ycocg(derivColor(color, dims, ref + int2(0, 1), exposure));
    const float3 e = ycocg(derivColor(color, dims, ref + int2(1, 0), exposure));
    const float3 w = ycocg(derivColor(color, dims, ref + int2(-1, 0), exposure));
    const float dC = dDelta(c, tm1.xyz, cw);
    const float dN = dDelta(c, n, cw), dS = dDelta(c, s, cw), dE = dDelta(c, e, cw), dW = dDelta(c, w, cw);
    const float sum = dN + dS + dE + dW, mx = max(max(dN, dS), max(dE, dW));
    const float prev = tm1.w;
    const float support = max(sum - mx, 0.0f) * (1.0f / 3.0f);
    const float sup = mix(dC, support, 0.30f) * 0.75f;
    float lmEntry = 0.0f;
    if (lm.on)
    {
        const int2 st = lm.inputStep;
        const float3 tn = ycocg(derivColor(color, dims, ref + int2(0, -st.y), exposure));
        const float3 ts = ycocg(derivColor(color, dims, ref + int2(0, st.y), exposure));
        const float3 te = ycocg(derivColor(color, dims, ref + int2(st.x, 0), exposure));
        const float3 tw = ycocg(derivColor(color, dims, ref + int2(-st.x, 0), exposure));
        const float4 dn = lm.derivTm1.sample(kLinZero, (lm.derivUv + float2(0.0f, -lm.derivStep.y)).yx);
        const float4 ds = lm.derivTm1.sample(kLinZero, (lm.derivUv + float2(0.0f, lm.derivStep.y)).yx);
        const float4 de = lm.derivTm1.sample(kLinZero, (lm.derivUv + float2(lm.derivStep.x, 0.0f)).yx);
        const float4 dw = lm.derivTm1.sample(kLinZero, (lm.derivUv + float2(-lm.derivStep.x, 0.0f)).yx);
        const float d_tn = dDelta(tn, dn.xyz, cw), d_ts = dDelta(ts, ds.xyz, cw), d_te = dDelta(te, de.xyz, cw), d_tw = dDelta(tw, dw.xyz, cw);
        const float r_tn = dDelta(c, tn, cw), r_ts = dDelta(c, ts, cw), r_te = dDelta(c, te, cw), r_tw = dDelta(c, tw, cw);
        const float tMax = max(max(dC, d_tn), max(max(d_ts, d_te), d_tw));
        const float range = max(max(r_tn, r_ts), max(r_te, r_tw));
        const float fMin = min(min(dC, d_tn), min(min(d_ts, d_te), d_tw));
        const float moireGate = saturate((tMax - 0.10f) / (0.30f - 0.10f));
        const float moireEntry = saturate((range - 0.99f) / (0.999f - 0.99f)) * moireGate * 0.50f;
        const float flatT = saturate((fMin - 0.015f) / (0.030f - 0.015f));
        const float flatR = 1.0f - saturate((range - 0.040f) / (0.120f - 0.040f));
        const float blueBias = (-0.75f * c.y) - (0.5f * c.z);
        const float flatB = saturate((blueBias - 0.220f) / (0.300f - 0.220f));
        const float flatL = saturate((c.x - 0.450f) / 0.10f) * (1.0f - saturate((c.x - 1.050f) / 0.20f));
        const float rgbB = c.x - (0.5f * (c.y + c.z));
        const float flatRgbB = saturate((rgbB - 0.800f) / 0.10f) * (1.0f - saturate((rgbB - 1.400f) / 0.20f));
        const float flatEntry = flatT * (flatR * flatB * flatL * flatRgbB) * 0.75f;
        lmEntry = max(moireEntry, flatEntry);
    }
    const float exc = max(sup - prev, 0.0f);
    const float recall = saturate((sup - 0.065f) / (0.420f - 0.065f));
    const float excS = saturate((exc - 0.025f) / (0.160f - 0.025f));
    const float meanG = saturate((sup - 0.070f) / (0.230f - 0.070f));
    const float rawEntry = max(sqrt(recall) * sqrt(excS) * meanG, lmEntry);
    const float heat = saturate((prev - 0.110f) / (0.210f - 0.110f));
    const float susSupport = mix(prev, sup, 0.30f);
    const float susFloor = mix(0.177f, 0.157f, heat), susCeil = mix(0.330f, 0.305f, heat);
    float susGate = saturate((susSupport - susFloor) / (susCeil - susFloor));
    susGate *= susGate;
    float hot = saturate((prev - 0.180f) / (0.280f - 0.180f));
    hot *= hot;
    const float carry = max(susGate, hot * 0.12f);
    const float rawSus = prev * carry * 0.80f;
    const float raw = max(rawEntry, rawSus);
    const float decay = max(susGate, heat * heat * 0.50f);
    const float fall = mix(0.24f, 0.05f, decay);
    const float rise = mix(0.08f, 0.22f, sqrt(recall * meanG));
    const float a = raw > prev ? rise : fall;
    const float filtered = mix(prev, raw, a);
    const float oa = filtered > prev ? 0.75f : 0.80f;
    const float visible = max(mix(prev, filtered, oa), lmEntry);
    const float4 state = float4(c, filtered);
    instability = visible;
    const float disB = float(dis > 0.01f);
    const float uninit = float((abs(tm1.x) + abs(tm1.y) + abs(tm1.z) + abs(tm1.w)) < 1e-4f);
    instability *= 1.0f - disB;
    const float4 resetS = float4(c, 0.0f);
    const float4 seeded = mix(state, resetS, disB);
    instability = mix(instability, 0.f, uninit);
    return mix(seeded, resetS, uninit);
}

kernel void nss_pre(constant PreArgs& A [[buffer(0)]],
                    texture2d<float> color [[texture(0)]], const device float* depth [[buffer(2)]],
                    const device float2* motion [[buffer(3)]], texture2d<float> history [[texture(1)]],
                    const device int* depthTm1 [[buffer(5)]], texture2d<float> derivTm1 [[texture(3)]],
                    texture2d<float> feedbackView [[texture(2)]],
                    device half* inputTensor [[buffer(8)]], texture2d<float, access::write> derivOut [[texture(4)]],
                    device float* disOut [[buffer(10)]], device float* nearestOut [[buffer(11)]],
                    const device float* gameK [[buffer(12)]],
                    uint2 gid [[thread_position_in_grid]])
{
    const float kx = gameK[0] * A.exposure;
    const int2 padded = int2(gid.y, gid.x);
    if (any(padded >= A.padDims)) return;
    const int2 inD = A.inDims;
    const float2 invIn = 1.0f / float2(inD);
    const int2 ref = int2(reflectIdx(padded.x, inD.x), reflectIdx(padded.y, inD.y));
    const float2 uv = (float2(ref) + 0.5f) * invIn;
    const float2 uvPad = (float2(padded) + 0.5f) / float2(A.padDims);

    // FindNearestDepth_4x4 (forward depth: smaller is nearer)
    const int2 so[16] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(0, -1), int2(-1, 0), int2(-1, 1), int2(1, 1), int2(-1, -1),
                         int2(1, -1), int2(-1, 2), int2(0, 2), int2(1, 2), int2(2, 2), int2(2, 1), int2(2, 0), int2(2, -1)};
    const int2 pxPos = ref;
    float nearest = LD(depth, inD, 0, clamp(pxPos, int2(0), inD - 1));
    int2 nearestCoord = pxPos, nearestOff = int2(0);
    for (int i = 1; i < 16; ++i)
    {
        const int2 p = pxPos + so[i];
        const float dn = LD(depth, inD, 0, clamp(p, int2(0), inD - 1));
        if (onScreen(p, inD) && dn < nearest) nearestCoord = p, nearest = dn, nearestOff = so[i];
    }
    const float2 nearestUv = (float2(nearestCoord) + 0.5f) / float2(inD);
    float2 mv = sampleM(motion, inD, nearestUv);
    mv *= float(length(mv) > 0.1f);

    const float2 reprojUv = uv - mv * invIn;
    const float2 unjitterUv = uv - A.jitter * invIn;
    const float2 reproj270 = ((float2(ref / 2) + 0.5f) / float2(A.depthDims)) - mv * invIn;
    const float2 reprojPad = uvPad - mv / float2(A.padDims);

    const float dis = depthClip(depthTm1, A.depthDims, reproj270, A.renderSize, nearest, A.deviceToView);
    const float3 unjittered = karis(sampleGameColor(color, unjitterUv, inD, A.colorTexInv) * kx);
    const float3 warped = karis(history.sample(kLinZero, reprojUv.yx).rgb * kx);
    const float4 dTm1 = derivTm1.sample(kLinZero, reprojUv.yx);
    float instability = 0;
    const LowMid noLm = {false, derivTm1, inD, 0, 0, 0};
    const float4 luma = lumaDerivative(color, inD, ref, kx, dTm1, dis, instability, noLm);
    float4 fb = sampleView(feedbackView, A.padDims, reprojPad, false);
    fb = mix(fb, 0.f, float(dis > 0.01f));
    // motion detector
    const float kMin = length(1.0f / A.renderSize), kMax = length(200.0f / A.renderSize);
    const float motionDet = sqrt((clamp(length(mv / A.renderSize), kMin, kMax) - kMin) / (kMax - kMin));

    const float vals[12] = {warped.r, warped.g, warped.b, unjittered.r, unjittered.g, unjittered.b, motionDet,
                            fb.r, fb.g, fb.b, fb.a, instability};
    device half* dstIn = inputTensor + (((long)padded.x + 1) * A.inWpad + padded.y + 1) * 12;
    for (int c = 0; c < 12; ++c) dstIn[c] = half(vals[c]);

    if (any(padded >= inD)) return;
    derivOut.write(luma, uint2(padded.y, padded.x));
    ST(disOut, inD, 0, padded, dis);
    const int2 cl = clamp(nearestOff, int2(-2), int2(2));
    ST(nearestOut, inD, 0, padded, float(((cl.y + 2) << 3) | (cl.x + 2)) / 255.0f);   // EncodeNearestDepthCoord
}

// ------------------------------------------------------------------------------------------------ mid/low: half-res inputs
kernel void nss_depth_scatter_q(constant ScatterArgs& A [[buffer(0)]],
                                const device float2* motion [[buffer(1)]], const device float* depth [[buffer(2)]],
                                device atomic_int* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    const int2 idx = int2(gid.y, gid.x);
    if (any(idx >= A.outDims)) return;
    const float2 invOut = 1.0f / float2(A.outDims);
    const float2 uv = (float2(idx) + 0.5f) * invOut;
    const float2 scale = float2(A.inDims) / float2(A.outDims), invScale = 1.0f / scale;
    const int2 srcSize = int2(round(float2(A.outDims) * scale));
    const int2 srcBase = int2(float2(idx) * scale);
    const float2 invSrc = invOut * invScale;
    const float2 uvs[4] = {(float2(srcBase) + 0.5f) * invSrc, (float2(srcBase + int2(0, 2)) + 0.5f) * invSrc,
                           (float2(srcBase + int2(2, 0)) + 0.5f) * invSrc, (float2(srcBase + int2(2, 2)) + 0.5f) * invSrc};
    const int2 qoff[4] = {int2(0, 0), int2(0, 2), int2(2, 0), int2(2, 2)};
    const int2 lane[4] = {int2(0, 0), int2(0, 1), int2(1, 0), int2(1, 1)};   // gather order tl, tr, bl, br
    float3 r = 0;
    for (int q = 0; q < 4; ++q)
    {
        const float4 d = gatherT(depth, A.inDims, 0, uvs[q]);
        float4 my, mx;
        gatherM(motion, A.inDims, uvs[q], my, mx);
        for (int k = 0; k < 4; ++k)
        {
            const float3 cand = float3(my[k], mx[k], d[k]);
            if (q == 0 && k == 0) { r = cand; continue; }
            const int2 src = srcBase + qoff[q] + lane[k];
            const float take = float(all(src >= 0) && all(src < srcSize)) * step(d[k], r.z);
            r = mix(r, cand, take);
        }
    }
    float2 mv = r.xy * invScale;
    mv *= float(length(r.xy) > 0.1f);
    mv *= invOut;
    reconstructPrevDepth(out, A.outDims, uv - mv, r.z);
}

struct DisLqArgs { float4 d2v; float2 renderSize; int2 inDims; int2 outDims; };

// Mid/low, when the render size is 4x the quarter-res grid: game prep + depth scatter in one pass. Each thread owns
// one 4x4 block of game pixels: writes NSS's depth / motion for it, scatters it like nss_depth_scatter_q, and keeps
// what nss_disocclusion_lq needs from the block (closest depth, its motion, view-depth range) in `stats`.
// The scatter target was cleared by the previous frame's nss_disocclusion_lq_s (double-buffered).
struct PrepQArgs { GamePrepArgs g; float4 d2v; int2 outDims; };
static inline float rdDepth(depth2d<float> t, uint2 p) { return t.read(p); }
static inline float rdDepth(texture2d<float> t, uint2 p) { return t.read(p).r; }

template <typename DT>
kernel void nss_prep_q(constant PrepQArgs& A [[buffer(0)]], DT depth [[texture(1)]],
                       texture2d<float> motion [[texture(2)]], texture2d<float> exposure [[texture(3)]],
                       device float* oDepth [[buffer(2)]], device float2* oMotion [[buffer(3)]], device float* oK [[buffer(4)]],
                       device atomic_int* out [[buffer(5)]], device float4* stats [[buffer(6)]],
                       uint2 gid [[thread_position_in_grid]])
{
    const int2 idx = int2(gid.y, gid.x);
    if (any(idx >= A.outDims)) return;
    if (gid.x == 0 && gid.y == 0) oK[0] = exposureK(A.g, exposure);
    const int2 base = idx * 4, inD = A.g.dims;
    float d[4][4];
    float2 m[4][4];
    const float2 ms = A.g.mvScale * (1.0f - A.g.reset);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
        {
            const int2 p = base + int2(y, x);
            const uint2 t = uint2(p.y, p.x);
            const float dv = rdDepth(depth, t);
            const float dd = A.g.depthReversed ? 1.0f - dv : dv;
            const float2 mm = motion.read(t).rg * ms;
            d[y][x] = dd, m[y][x] = float2(-mm.y, -mm.x);
            oDepth[p.x * inD.y + p.y] = dd;
            oMotion[p.x * inD.y + p.y] = m[y][x];
        }
    // scatter: the same candidate order as nss_depth_scatter_q (quads, then gather order), later ties win
    float3 r = float3(m[0][0], d[0][0]);
    for (int q = 0; q < 4; ++q)
        for (int k = 0; k < 4; ++k)
        {
            if (q == 0 && k == 0) continue;
            const int y = (q >> 1) * 2 + (k >> 1), x = (q & 1) * 2 + (k & 1);
            r = mix(r, float3(m[y][x], d[y][x]), step(d[y][x], r.z));
        }
    const float2 invOut = 1.0f / float2(A.outDims);
    const float2 uv = (float2(idx) + 0.5f) * invOut;
    float2 mv = r.xy * 0.25f;
    mv *= float(length(r.xy) > 0.1f);
    mv *= invOut;
    reconstructPrevDepth(out, A.outDims, uv - mv, r.z);
    // disocclusion inputs: nss_disocclusion_lq's scan order (row offset fastest), first minimum wins
    float closest = d[0][0];
    float2 cmv = m[0][0];
    float vmin = viewDepth(closest, A.d2v), vmax = vmin;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
        {
            const float dv = d[x][y], vd = viewDepth(dv, A.d2v);
            vmin = min(vmin, vd), vmax = max(vmax, vd);
            if (dv < closest) closest = dv, cmv = m[x][y];
        }
    stats[idx.x * A.outDims.y + idx.y] = float4(closest, cmv, vmax - vmin);
}
template [[host_name("nss_prep_q")]] kernel void nss_prep_q<texture2d<float>>(constant PrepQArgs&, texture2d<float>, texture2d<float>,
    texture2d<float>, device float*, device float2*, device float*, device atomic_int*, device float4*, uint2);
template [[host_name("nss_prep_q_d")]] kernel void nss_prep_q<depth2d<float>>(constant PrepQArgs&, depth2d<float>, texture2d<float>,
    texture2d<float>, device float*, device float2*, device float*, device atomic_int*, device float4*, uint2);

// nss_disocclusion_lq from nss_prep_q's block stats; also clears the other scatter buffer for next frame
kernel void nss_disocclusion_lq_s(constant DisLqArgs& A [[buffer(0)]], const device float4* stats [[buffer(1)]],
                                  device int* clearNext [[buffer(2)]], const device int* depthTm1 [[buffer(3)]],
                                  device float* out [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const int2 idx = int2(gid.y, gid.x);
    if (any(idx >= A.outDims)) return;
    const int i = idx.x * A.outDims.y + idx.y;
    clearNext[i] = 0x7fffffff;
    const float4 st = stats[i];
    const float2 uv = (float2(idx) + 0.5f) / float2(A.outDims);
    const float2 mvD = st.yz * (float2(A.outDims) / float2(A.inDims)) * float(length(st.yz) > 0.1f);
    const float2 reproj = uv - mvD / float2(A.outDims);
    ST(out, A.outDims, 0, idx, depthClip(depthTm1, A.outDims, reproj, A.renderSize, st.x, A.d2v, st.w));
}
kernel void nss_disocclusion_lq(constant DisLqArgs& A [[buffer(0)]],
                                const device float2* motion [[buffer(1)]], const device float* depth [[buffer(2)]],
                                const device int* depthTm1 [[buffer(3)]], device float* out [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const int2 idx = int2(gid.y, gid.x);
    if (any(idx >= A.outDims)) return;
    const float2 uv = (float2(idx) + 0.5f) / float2(A.outDims);
    const int2 base = int2(float2(idx) * (float2(A.inDims) / float2(A.outDims)));
    const int2 sp0 = clamp(base, int2(0), A.inDims - 1);
    float closest = LD(depth, A.inDims, 0, sp0);
    float vmin = viewDepth(closest, A.d2v), vmax = vmin;
    float2 mv = motion[(sp0).x * (A.inDims).y + (sp0).y];
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
        {
            const int2 sp = base + int2(x, y);
            if (!onScreen(sp, A.inDims)) continue;
            const float d = LD(depth, A.inDims, 0, sp);
            const float vd = viewDepth(d, A.d2v);
            vmin = min(vmin, vd), vmax = max(vmax, vd);
            if (d < closest) closest = d, mv = motion[(sp).x * (A.inDims).y + (sp).y];
        }
    const float2 mvD = mv * (float2(A.outDims) / float2(A.inDims)) * float(length(mv) > 0.1f);
    const float2 reproj = uv - mvD / float2(A.outDims);
    ST(out, A.outDims, 0, idx, depthClip(depthTm1, A.outDims, reproj, A.renderSize, closest, A.d2v, vmax - vmin));
}

static void nearestLowMid(const device float* depth, int2 inD, int2 px, thread float& nd, thread int2& off)
{
    const int2 so[16] = {int2(0, 0), int2(0, -1), int2(-1, 0), int2(-1, -1), int2(0, 1), int2(-1, 1), int2(-1, 2), int2(0, 2),
                         int2(1, 0), int2(1, -1), int2(2, 0), int2(2, -1), int2(1, 1), int2(1, 2), int2(2, 1), int2(2, 2)};
    nd = LD(depth, inD, 0, clamp(px, int2(0), inD - 1));
    off = int2(0);
    for (int i = 1; i < 16; ++i)
    {
        const int2 p = px + so[i];
        const float d = LD(depth, inD, 0, clamp(p, int2(0), inD - 1));
        if (onScreen(p, inD) && d <= nd) nd = d, off = so[i];
    }
}
static inline float2 motionAt(const device float2* motion, int2 inD, int2 p)
{
    p = clamp(p, int2(0), inD - 1);
    float2 v = motion[(p).x * (inD).y + (p).y];
    return v * float(length(v) > 0.1f);
}
static inline uint nibble(int2 o)
{
    const int2 c = clamp(o, int2(-1), int2(2));
    return uint(((c.y + 1) << 2) | (c.x + 1));
}

struct PreHalfArgs
{
    float4 deviceToView;
    float2 jitter, renderSize;
    int2 inDims, padDims, procDims, depthDims;
    float exposure;
    int inWpad, tempWpad;
    float2 colorTexInv;     // 1 / the game's color texture size; Metal puts it at byte 80 (the host pads)
};


// Mid/low pre-process specialization: the engine's fixed sizes as function constants (60+); jitter stays runtime.
constant int2 fcH_inDims [[function_constant(60)]];
constant int2 fcH_padDims [[function_constant(61)]];
constant int2 fcH_procDims [[function_constant(62)]];
constant int2 fcH_depthDims [[function_constant(63)]];
constant float2 fcH_renderSize [[function_constant(64)]];
constant float fcH_exposure [[function_constant(65)]];
constant int fcH_inWpad [[function_constant(66)]];
constant int fcH_tempWpad [[function_constant(67)]];
constant bool fcHOn = is_function_constant_defined(fcH_inDims);
constant int fcPreAblate [[function_constant(68)]];   // diagnostics: 1 no moire detector, 2 one history tap, 4 no packed-offset searches
#define H_inDims (fcHOn ? fcH_inDims : A.inDims)
#define H_padDims (fcHOn ? fcH_padDims : A.padDims)
#define H_procDims (fcHOn ? fcH_procDims : A.procDims)
#define H_depthDims (fcHOn ? fcH_depthDims : A.depthDims)
#define H_renderSize (fcHOn ? fcH_renderSize : A.renderSize)
#define H_exposure (fcHOn ? fcH_exposure : A.exposure)
#define H_inWpad (fcHOn ? fcH_inWpad : A.inWpad)
#define H_tempWpad (fcHOn ? fcH_tempWpad : A.tempWpad)

kernel void nss_pre_half(constant PreHalfArgs& A [[buffer(0)]],
                         texture2d<float> color [[texture(0)]], const device float* depth [[buffer(2)]],
                         const device float2* motion [[buffer(3)]], texture2d<float> history [[texture(1)]],
                         const device float* disLq [[buffer(5)]], texture2d<float> derivTm1 [[texture(3)]],
                         texture2d<float> feedbackView [[texture(2)]],
                         device half* inputTensor [[buffer(8)]], texture2d<float, access::write> derivOut [[texture(4)]],
                         device float* disOut [[buffer(10)]], device float* nearestOut [[buffer(11)]],
                         const device float* gameK [[buffer(12)]], device float2* dilMotion [[buffer(13)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const float kx = gameK[0] * H_exposure;
    const int2 padded = int2(gid.y, gid.x);
    if (any(padded >= H_padDims)) return;
    const int2 inD = H_inDims;
    const float2 invIn = 1.0f / float2(inD);
    const float2 scale = float2(H_procDims) / float2(inD), invScale = 1.0f / scale;
    const int2 proc = int2(reflectIdx(padded.x, H_procDims.x), reflectIdx(padded.y, H_procDims.y));
    const int2 ref = clamp(int2(floor((float2(proc) + 0.5f) * invScale)), int2(0), inD - 1);
    const float2 uv = (float2(ref) + 0.5f) * invIn;
    const float2 uvPad = (float2(padded) + 0.5f) / float2(H_padDims);

    float nd;
    int2 off;
    nearestLowMid(depth, inD, clamp(int2(uv * float2(inD)), int2(0), inD - 1), nd, off);
    const float2 mv = motionAt(motion, inD, clamp(ref + off, int2(0), inD - 1));

    const float2 reprojPad = uvPad - mv * scale / float2(H_padDims);
    const float2 unjitterUv = uv - A.jitter * invIn;
    const float dis = sampleT(disLq, H_depthDims, 0, (float2(proc) + 0.5f) / float2(H_procDims), true);
    const float3 unjittered = karis(sampleGameColor(color, unjitterUv, inD, A.colorTexInv) * kx);

    // history: average of the four input pixels under this process pixel, each warped by its own dilated motion
    const int2 hb = clamp(proc * 2, int2(0), inD - 1);
    const int2 lanes[4] = {hb, min(hb + int2(1, 0), inD - 1), min(hb + int2(0, 1), inD - 1), min(hb + int2(1, 1), inD - 1)};
    float3 warped = 0;
    for (int l = 0; l < ((fcPreAblate & 2) ? 1 : 4); ++l)
    {
        const int2 ic = clamp(lanes[l], int2(0), inD - 1);
        float d;
        int2 o;
        nearestLowMid(depth, inD, ic, d, o);
        const float2 m = motionAt(motion, inD, clamp(ic + o, int2(0), inD - 1));
        const float2 r = (float2(ic) + 0.5f) * invIn - m * invIn;
        warped += karis(history.sample(kLinZero, r.yx).rgb * kx);
    }
    warped *= 0.25f;

    const float2 dStep = 1.0f / float2(H_padDims);
    const float4 dTm1 = derivTm1.sample(kLinZero, reprojPad.yx);
    const LowMid lm = {!(fcPreAblate & 1), derivTm1, H_padDims, reprojPad, dStep, max(int2(invScale + 0.5f), int2(1))};
    float instability = 0;
    const float4 luma = lumaDerivative(color, inD, ref, kx, dTm1, dis, instability, lm);
    float4 fb = sampleView(feedbackView, H_padDims, reprojPad, false);
    fb = mix(fb, 0.f, float(dis > 0.01f));
    const float kMin = length(1.0f / H_renderSize), kMax = length(200.0f / H_renderSize);
    const float motionDet = sqrt((clamp(length(mv / H_renderSize), kMin, kMax) - kMin) / (kMax - kMin));

    const float vals[12] = {warped.r, warped.g, warped.b, unjittered.r, unjittered.g, unjittered.b, motionDet,
                            fb.r, fb.g, fb.b, fb.a, instability};
    device half* dstIn = inputTensor + (((long)padded.x + 1) * H_inWpad + padded.y + 1) * 12;
    for (int c = 0; c < 12; ++c) dstIn[c] = half(vals[c]);

    if (any(padded >= H_procDims)) return;
    derivOut.write(luma, uint2(padded.y, padded.x));
    ST(disOut, H_procDims, 0, padded, dis);
    // packed nearest offsets of the 2x2 input quad under this process pixel
    const int2 qb = clamp(padded * 2, int2(0), inD - 1);
    const int2 q[4] = {qb, min(qb + int2(0, 1), inD - 1), min(qb + int2(1, 0), inD - 1), min(qb + int2(1, 1), inD - 1)};
    for (int l = 0; l < 4; ++l)
    {
        float d;
        int2 o;
        if (fcPreAblate & 4) o = int2(0); else nearestLowMid(depth, inD, q[l], d, o);
        // post reads the motion at the nearest-depth neighbour straight from here (one load instead of two)
        const int2 icq = padded * 2 + int2(l >> 1, l & 1);
        const int2 dc = clamp(icq + o, int2(0), inD - 1);
        if (all(icq < inD)) dilMotion[icq.x * inD.y + icq.y] = motion[dc.x * inD.y + dc.y];
    }
}

// ------------------------------------------------------------------------------------------------ network
// 3x3 conv, zero padding 1, stride 1 or 2. Input = concat(srcA [cA ch], srcB [cB ch]); either may be read
// through a 2x nearest upsample. Weights [oc][ic][3][3] then bias [oc]. act: 0 relu, 1 sigmoid.
struct ConvArgs
{
    int2 outDims;
    int2 aDims, bDims;    // stored dims of the sources
    int cA, cB, oc, stride;
    int upA, upB;         // 1: source is read at half resolution (nearest upsample)
    int act;
    int ocPerThread;
};

kernel void nss_conv(constant ConvArgs& A [[buffer(0)]], const device float* srcA [[buffer(1)]],
                     const device float* srcB [[buffer(2)]], const device float* weights [[buffer(3)]],
                     device float* dst [[buffer(4)]], uint3 gid [[thread_position_in_grid]])
{
    const int2 o = int2(gid.y, gid.x);
    if (any(o >= A.outDims)) return;
    const int oc0 = gid.z * A.ocPerThread;
    const int ic = A.cA + A.cB;
    const int2 inDims = A.outDims * A.stride;   // logical input dims
    float acc[16];
    for (int k = 0; k < A.ocPerThread; ++k) acc[k] = weights[A.oc * ic * 9 + oc0 + k];
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
        {
            const int2 p = o * A.stride + int2(ky - 1, kx - 1);
            if (any(p < 0) || any(p >= inDims)) continue;
            for (int c = 0; c < ic; ++c)
            {
                float v;
                if (c < A.cA) v = LD(srcA, A.aDims, c, A.upA ? p / 2 : p);
                else v = LD(srcB, A.bDims, c - A.cA, A.upB ? p / 2 : p);
                for (int k = 0; k < A.ocPerThread; ++k) acc[k] += v * weights[((oc0 + k) * ic + c) * 9 + ky * 3 + kx];
            }
        }
    for (int k = 0; k < A.ocPerThread; ++k)
    {
        const float r = A.act == 0 ? max(acc[k], 0.0f) : 1.0f / (1.0f + exp(-acc[k]));
        ST(dst, A.outDims, oc0 + k, o, r);
    }
}

// ------------------------------------------------------------------------------------------------ post-process
struct PostArgs
{
    int2 inDims, outDims, kpnDims, tempDims;
    int2 idxMod;
    int taps;
    int kpnChannels;
    float exposure;
    float reset;
    int kpnWpad, kpnC, tempWpad;
    int2 procDims;       // preprocess (process) dims: input dims for high, half for mid/low
    int2 nearestDims;    // nearest-offset tensor dims
    int halfRes, sparse, catmullHist;
    int writeGame;       // also write the game's output (divided by k) from here, instead of a separate pass
    int ablate;          // diagnostics only: 1 no filter taps, 2 no history sample, 4 no temporal sample, 8 no game write, 16 no history write
};

constant int kSparseMap[36] = {-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1,
                               -1, 8, 9, 10, 11, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1};

// floor(a / b) for non-negative ints without the GPU's slow integer divide: a float estimate, then an exact fix-up.
static inline int2 fastDiv(int2 a, int2 b, float2 invB)
{
    int2 q = int2(float2(a) * invB);
    q -= int2(q * b > a);
    q += int2((q + 1) * b <= a);
    return q;
}

static float3 catmull(texture2d<float> t, int2 dims, float2 uv)
{
    const float2 size = float2(dims), inv = 1.0f / size;
    float2 u = uv * size;
    const float2 tc = floor(u - 0.5f) + 0.5f;
    const float2 f = u - tc, f2 = f * f, f3 = f2 * f;
    const float2 w0 = f2 - 0.5f * (f3 + f);
    const float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
    const float2 w3 = 0.5f * (f3 - f2);
    const float2 w2 = 1.0f - w0 - w1 - w3;
    const float2 W0 = w0, W1 = w1 + w2, W2 = w3;
    const float2 s0 = (tc - 1.0f) * inv, s1 = (tc + w2 / W1) * inv, s2 = (tc + 2.0f) * inv;
    const float2 uvs[5] = {float2(s1.x, s0.y), float2(s0.x, s1.y), float2(s1.x, s1.y), float2(s2.x, s1.y), float2(s1.x, s2.y)};
    const float ws[5] = {W1.x * W0.y, W0.x * W1.y, W1.x * W1.y, W2.x * W1.y, W1.x * W2.y};
    float3 col = 0, lo = MAX_HALF, hi = -MAX_HALF;
    float wsum = 0;
    for (int i = 0; i < 5; ++i)
    {
        const float3 s = t.sample(kLinClamp, uvs[i].yx).rgb;
        lo = min(lo, s), hi = max(hi, s);
        col += s * ws[i];
        wsum += ws[i];
    }
    col *= 1.0f / wsum;
    return any(col < 0.f) ? clamp(col, lo, hi) : col;
}

static inline float sharpenTheta(float t, float g)
{
    t = clamp(t, 0.f, 1.f);
    const float a = pow(t, g), b = pow(1.0f - t, g);
    return a / max(a + b, 1e-6f);
}

// Post-process specialization: the engine's fixed shape and quality switches as function constants (indices 30+);
// only `reset` stays a runtime argument.
constant int2 fcP_inDims [[function_constant(30)]];
constant int2 fcP_outDims [[function_constant(31)]];
constant int2 fcP_kpnDims [[function_constant(32)]];
constant int2 fcP_tempDims [[function_constant(33)]];
constant int2 fcP_idxMod [[function_constant(34)]];
constant int fcP_taps [[function_constant(35)]];
constant int fcP_kpnChannels [[function_constant(36)]];
constant float fcP_exposure [[function_constant(37)]];
constant int fcP_kpnWpad [[function_constant(38)]];
constant int fcP_kpnC [[function_constant(39)]];
constant int fcP_tempWpad [[function_constant(40)]];
constant int2 fcP_procDims [[function_constant(41)]];
constant int2 fcP_nearestDims [[function_constant(42)]];
constant int fcP_halfRes [[function_constant(43)]];
constant int fcP_sparse [[function_constant(44)]];
constant int fcP_catmullHist [[function_constant(45)]];
constant int fcP_writeGame [[function_constant(46)]];
constant int fcP_ablate [[function_constant(47)]];
constant bool fcPOn = is_function_constant_defined(fcP_inDims);
#define P_inDims (fcPOn ? fcP_inDims : A.inDims)
#define P_outDims (fcPOn ? fcP_outDims : A.outDims)
#define P_kpnDims (fcPOn ? fcP_kpnDims : A.kpnDims)
#define P_tempDims (fcPOn ? fcP_tempDims : A.tempDims)
#define P_idxMod (fcPOn ? fcP_idxMod : A.idxMod)
#define P_taps (fcPOn ? fcP_taps : A.taps)
#define P_kpnChannels (fcPOn ? fcP_kpnChannels : A.kpnChannels)
#define P_exposure (fcPOn ? fcP_exposure : A.exposure)
#define P_kpnWpad (fcPOn ? fcP_kpnWpad : A.kpnWpad)
#define P_kpnC (fcPOn ? fcP_kpnC : A.kpnC)
#define P_tempWpad (fcPOn ? fcP_tempWpad : A.tempWpad)
#define P_procDims (fcPOn ? fcP_procDims : A.procDims)
#define P_nearestDims (fcPOn ? fcP_nearestDims : A.nearestDims)
#define P_halfRes (fcPOn ? fcP_halfRes : A.halfRes)
#define P_sparse (fcPOn ? fcP_sparse : A.sparse)
#define P_catmullHist (fcPOn ? fcP_catmullHist : A.catmullHist)
#define P_writeGame (fcPOn ? fcP_writeGame : A.writeGame)
#define P_ablate (fcPOn ? fcP_ablate : A.ablate)

static inline void postPixel(constant PostArgs& A, texture2d<float> color, texture2d<float> history,
                             const device half* kpn, texture2d<float> temporal, const device float2* motion,
                             const device float* nearest, const device float2* dilMotion, const device uint* lut, texture2d<float, access::write> outTex,
                             texture2d<float, access::write> gameOut, const device float* gameK, uint2 gid)
{
    const float kx = gameK[0] * P_exposure;   // the game's exposure / preExposure: NSS's exposure input
    const int2 px = int2(gid.y, gid.x);
    if (any(px >= P_outDims)) return;
    const float2 scale = float2(P_outDims) / float2(P_inDims), invScale = 1.0f / scale;
    const float2 uv = (float2(px) + 0.5f) / float2(P_outDims);
    const float2 kpnScale = float2(P_kpnDims) / float2(P_tempDims);
    const HWC kh = {P_kpnDims, P_kpnWpad, P_kpnC};

    // FilterColor
    const float2 invMod = 1.0f / float2(P_idxMod), invIn = 1.0f / float2(P_inDims);
    const int2 tileIdx = px - fastDiv(px, P_idxMod, invMod) * P_idxMod;   // px % idxMod (px >= 0)
    const int lutIdx = tileIdx.x * P_idxMod.y + tileIdx.y;
    const int tiles = P_idxMod.x * P_idxMod.y;
    float3 m1 = 0, m2 = 0;
    float4 center = 0;
    float wsum = 0;
    for (int i = 0; i < ((P_ablate & 1) ? 1 : P_taps); ++i)
    {
        const uint w32 = lut[lutIdx * P_taps + i];
        const float3 tile = float3(float(int(w32 & 15u) - 8), float(int((w32 >> 4) & 15u) - 8), float((w32 >> 8) & 1u));
        const float3 tapPos = float3(float(int((w32 >> 9) & 15u) - 8), float(int((w32 >> 13) & 15u) - 8), float(w32 >> 17));
        const int2 tap = px + int2(tapPos.xy);
        const int2 lr = clamp(int2(floor((float2(tap) + 0.5f) * invScale + 1e-3f)) + int2(tile.xy), int2(0), P_inDims - 1);
        const float isCenter = float(tapPos.x == 0.f && tapPos.y == 0.f);
        const float3 c = min(colorAt(color, lr) * kx, MAX_HALF);
        center = mix(center, float4(c, 1.f), isCenter);
        int2 kt;
        if (P_sparse || P_halfRes) kt = clamp(min(fastDiv(lr * P_kpnDims, P_inDims, invIn), P_kpnDims - 1), int2(0), P_kpnDims - 1);
        else kt = clamp(int2(floor((float2(lr) + 0.5f + 1e-3f) * kpnScale)), int2(0), P_kpnDims - 1);
        float w;
        if (P_sparse)
        {
            const int tz = int(tapPos.z);
            const int ch = (tz >= 0 && tz < 36) ? kSparseMap[tz] : -1;
            w = ch >= 0 ? max(LDH(kpn, kh, ch, kt), EPS) : 0.f;
        }
        else w = max(LDH(kpn, kh, clamp(int(tapPos.z), 0, P_kpnChannels - 1), kt), EPS);
        w *= tile.z;
        m1 += c * w;
        m2 += c * c * w;
        wsum += w;
    }
    m1 /= max(wsum, EPS);
    m2 /= max(wsum, EPS);

    // temporal params (padded tensor, sampled in preprocess space)
    const float2 uvT = uv * (float2(P_procDims) / float2(P_tempDims));
    const float3 tp = (P_ablate & 4) ? float3(0.5f) : sampleView(temporal, P_tempDims, uvT, true).xyz;
    const float gamma = tp.z * 2.f;
    const float theta = sharpenTheta(tp.x, 2.f);
    const float alpha = tp.y * 0.35f + 0.05f;

    // warp history with the dilated motion
    const int2 ic = int2(float2(px) * invScale);
    int2 dil;
    float2 v;
    if (P_halfRes)
    {
        const int2 icl = min(ic, P_inDims - 1);
        v = dilMotion[icl.x * P_inDims.y + icl.y] * scale;   // nss_pre_half: motion at the nearest-depth neighbour
    }
    else
    {
        const int2 icl = clamp(ic, int2(0), P_inDims - 1);
        const int code = int(LD(nearest, P_nearestDims, 0, icl) * 255.0f + 0.5f);
        dil = int2((code & 0x7) - 2, ((code >> 3) & 0x7) - 2);
        const int2 dc = clamp(ic + dil, int2(0), P_inDims - 1);
        v = motion[(dc).x * (P_inDims).y + (dc).y] * scale;
    }
    v *= float(length(v) > 0.1f);
    const float2 reproj = uv - v / float2(P_outDims);
    const float onscreen = float(all(reproj >= 0.f) && all(reproj <= 1.f));
    const float3 warped = (P_ablate & 2) ? m1 : min((P_catmullHist ? catmull(history, P_outDims, reproj) : history.sample(kLinZero, reproj.yx).rgb) * kx, MAX_HALF);

    // clamp history to neighbourhood stats
    const float3 sigma = sqrt(max(abs(m2 - m1 * m1), EPS)) * gamma;
    const float3 hc = mix(m1, clamp(warped, m1 - sigma, m1 + sigma), A.reset);
    const float3 rect = mix(hc, warped, theta * onscreen * A.reset);

    // accumulate in tonemapped space
    const float3 rectTm = karis(rect);
    const float a = alpha * center.a * A.reset;
    const float3 acc = mix(rectTm, karis(center.rgb), a);
    const float3 outL = karisInverse(clamp(acc, 0.f, 1.f - EPS)) * (1.0f / kx);
    if (!(P_ablate & 16)) outTex.write(float4(outL, 1.0f), uint2(gid));
    if (P_writeGame && !(P_ablate & 8)) gameOut.write(float4(outL, 1.0f), uint2(gid));
}

#define POST_ARGS constant PostArgs& A [[buffer(0)]], \
    texture2d<float> color [[texture(1)]], texture2d<float> history [[texture(2)]], \
    const device half* kpn [[buffer(3)]], texture2d<float> temporal [[texture(4)]], \
    const device float2* motion [[buffer(5)]], const device float* nearest [[buffer(6)]], const device float2* dilMotion [[buffer(8)]], \
    const device uint* lut [[buffer(7)]], texture2d<float, access::write> outTex [[texture(0)]], \
    texture2d<float, access::write> gameOut [[texture(3)]], const device float* gameK [[buffer(9)]]

kernel void nss_post(POST_ARGS, uint2 gid [[thread_position_in_grid]])
{
    postPixel(A, color, history, kpn, temporal, motion, nearest, dilMotion, lut, outTex, gameOut, gameK, gid);
}


// ------------------------------------------------------------------------------------------------ network (matrix units)
// Implicit-GEMM convolutions on simdgroup 8x8 matrices: half inputs and weights, float accumulation.
// One simdgroup = 8 output pixels of one row x all OC. Tensors are channels-last with a zero border, so every
// tap is a plain strided 8x8 load.
struct MmaArgs
{
    int hOut, wOut;          // conv: output dims. upconv: low-res (source) dims
    int inWpad, inC, inCoff;
    int outWpad, outC, outCoff;
    int IC, stride, act;
};

// Per-layer specialization: the host compiles each layer's chosen conv variant with its shape as function constants
// (fixed loop counts and strides); without them the kernels read the shape from MmaArgs.
constant int fcIC [[function_constant(10)]];
constant int fcInC [[function_constant(11)]];
constant int fcInCoff [[function_constant(12)]];
constant int fcOutC [[function_constant(13)]];
constant int fcOutCoff [[function_constant(14)]];
constant int fcStride [[function_constant(15)]];
constant int fcAct [[function_constant(16)]];
constant int fcInWpad [[function_constant(17)]];
constant int fcOutWpad [[function_constant(18)]];
// packed taps: the 3 horizontal taps x C channels are contiguous in a channels-last row, so one K run of L_IC
// (3*C rounded up to 8) per kernel row instead of 3 runs of C rounded up (conv0: 5 blocks per row instead of 6)
constant bool fcPackKx [[function_constant(19)]];
constant bool kPackKx = is_function_constant_defined(fcPackKx) && fcPackKx;
// output-channel split: each simdgroup computes OC/split channels and threadgroup z picks the group, so small
// low-res layers get `split` times the simdgroups (more in flight to hide latency)
constant int fcOCSplit [[function_constant(20)]];
constant int kOCSplit = is_function_constant_defined(fcOCSplit) ? fcOCSplit : 1;
constant bool fcOn = is_function_constant_defined(fcIC);
#define L_IC (fcOn ? fcIC : A.IC)
#define L_INC (fcOn ? fcInC : A.inC)
#define L_INCOFF (fcOn ? fcInCoff : A.inCoff)
#define L_OUTC (fcOn ? fcOutC : A.outC)
#define L_OUTCOFF (fcOn ? fcOutCoff : A.outCoff)
#define L_STRIDE (fcOn ? fcStride : A.stride)
#define L_ACT (fcOn ? fcAct : A.act)
#define L_INWPAD (fcOn ? fcInWpad : A.inWpad)
#define L_OUTWPAD (fcOn ? fcOutWpad : A.outWpad)

// Accumulator precision for the conv matrix ops (NSS_HALF_ACC=1: half; default float).
#if NSS_HALF_ACC
typedef simdgroup_half8x8 nss_acc8x8;
typedef half nss_acc_t;
#else
typedef simdgroup_float8x8 nss_acc8x8;
typedef float nss_acc_t;
#endif

template <int kOC>
static void epilogue(thread nss_acc8x8* acc, threadgroup nss_acc_t* t, const device float* bias, device half* dst,
                     int outC, int outStep, int valid, int act, ushort lane)
{
    constexpr int NOB = kOC / 8;
    for (int ob = 0; ob < NOB; ++ob) simdgroup_store(acc[ob], t + ob * 8, kOC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 8 * kOC; e += 32)
    {
        const int j = e / kOC, o = e % kOC;
        if (j >= valid || o >= outC) continue;
        float v = t[e] + bias[o];
        v = act == 0 ? max(v, 0.0f) : 1.0f / (1.0f + exp(-v));
        dst[(long)j * outStep * outC + o] = half(v);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <int NOBG>
static void epilogueG(thread nss_acc8x8* acc, threadgroup nss_acc_t* t, const device float* bias, device half* dst,
                      int ocBase, int outC, int valid, int act, ushort lane)
{
    constexpr int W = NOBG * 8;
    for (int ob = 0; ob < NOBG; ++ob) simdgroup_store(acc[ob], t + ob * 8, W);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 8 * W; e += 32)
    {
        const int j = e / W, o = ocBase + e % W;
        if (j >= valid || o >= outC) continue;
        float v = t[e] + bias[o];
        v = act == 0 ? max(v, 0.0f) : 1.0f / (1.0f + exp(-v));
        dst[(long)j * outC + o] = half(v);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// 3x3 conv, stride 1 or 2. Weights [tap][IC][OC] half, bias [OC] float.
// One simdgroup computes PX x 8 consecutive output pixels of one row (all OC), so each weight tile loaded serves PX
// pixel tiles. More PX means fewer weight loads but fewer simdgroups: the host times 1, 2 and 4 per layer at setup.
// Reads past the row end stay inside the padded buffer and only feed masked-off outputs.
template <int kOC, int PX>
kernel void nss_conv_mma(constant MmaArgs& A [[buffer(0)]], const device half* in [[buffer(1)]],
                         const device half* w [[buffer(2)]], const device float* bias [[buffer(3)]],
                         device half* out [[buffer(4)]], threadgroup nss_acc_t* tile [[threadgroup(0)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]])
{
    static_assert(kOC % 8 == 0, "OC must be a multiple of 8");
    constexpr int NOB = kOC / 8;
    const int NOBG = NOB / kOCSplit, ob0 = int(tg.z) * NOBG;
    const int row = tg.y;
    const int col0 = (int(tg.x) * 4 + sg) * 8 * PX;
    if (col0 >= A.wOut) return;
    nss_acc8x8 acc[PX][NOB];
    for (int p = 0; p < PX; ++p)
        for (int i = 0; i < NOB; ++i) acc[p][i] = nss_acc8x8(0);
    const ulong aStride = (ulong)L_INC * L_STRIDE;
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < (kPackKx ? 1 : 3); ++kx)
        {
            const device half* base = in + ((long)(row * L_STRIDE + ky) * L_INWPAD + col0 * L_STRIDE + kx) * L_INC + L_INCOFF;
            const device half* wt = w + (kPackKx ? ky : ky * 3 + kx) * L_IC * kOC;
            for (int cb = 0; cb < L_IC; cb += 8)
            {
                simdgroup_half8x8 a[PX];
                for (int p = 0; p < PX; ++p) simdgroup_load(a[p], base + (long)p * 8 * aStride + cb, aStride);
                for (int ob = 0; ob < NOBG; ++ob)
                {
                    simdgroup_half8x8 b;
                    simdgroup_load(b, wt + cb * kOC + (ob0 + ob) * 8, kOC);
                    for (int p = 0; p < PX; ++p) simdgroup_multiply_accumulate(acc[p][ob], a[p], b, acc[p][ob]);
                }
            }
        }
    for (int p = 0; p < PX; ++p)
    {
        const int c = col0 + p * 8;
        if (c >= A.wOut) break;
        device half* dst = out + ((long)(row + 1) * L_OUTWPAD + c + 1) * L_OUTC + L_OUTCOFF;
        if (kOCSplit == 1) epilogue<kOC>(acc[p], tile + sg * 8 * kOC, bias, dst, L_OUTC, 1, min(8, A.wOut - c), L_ACT, lane);
        else if (NOBG == 1) epilogueG<1>(acc[p], tile + sg * 8 * kOC, bias, dst, ob0 * 8, L_OUTC, min(8, A.wOut - c), L_ACT, lane);
        else if (NOBG == 2) epilogueG<2>(acc[p], tile + sg * 8 * kOC, bias, dst, ob0 * 8, L_OUTC, min(8, A.wOut - c), L_ACT, lane);
        else epilogueG<4>(acc[p], tile + sg * 8 * kOC, bias, dst, ob0 * 8, L_OUTC, min(8, A.wOut - c), L_ACT, lane);
    }
}

// 2x nearest upsample followed by a 3x3 conv, computed on the low-res source as four 2x2 phase convolutions.
// Weights [phase(py,px)][ty][tx][IC][OC] half (taps pre-summed on the host), bias [OC] float.
template <int kOC, int PX>
kernel void nss_upconv_mma(constant MmaArgs& A [[buffer(0)]], const device half* in [[buffer(1)]],
                           const device half* w [[buffer(2)]], const device float* bias [[buffer(3)]],
                           device half* out [[buffer(4)]], threadgroup nss_acc_t* tile [[threadgroup(0)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int NOB = kOC / 8;
    const int row = tg.y;                               // low-res row
    const int col0 = (int(tg.x) * 4 + sg) * 8 * PX;     // low-res col
    if (col0 >= A.wOut) return;
    for (int py = 0; py < 2; ++py)
        for (int px = 0; px < 2; ++px)
        {
            nss_acc8x8 acc[PX][NOB];
            for (int p = 0; p < PX; ++p)
                for (int i = 0; i < NOB; ++i) acc[p][i] = nss_acc8x8(0);
            for (int ty = 0; ty < 2; ++ty)
                for (int tx = 0; tx < 2; ++tx)
                {
                    const int sr = row + py - 1 + ty, sc = col0 + px - 1 + tx;   // source (unpadded), may be -1 / W
                    const device half* base = in + ((long)(sr + 1) * L_INWPAD + sc + 1) * L_INC + L_INCOFF;
                    const device half* wt = w + (((py * 2 + px) * 2 + ty) * 2 + tx) * L_IC * kOC;
                    for (int cb = 0; cb < L_IC; cb += 8)
                    {
                        simdgroup_half8x8 a[PX];
                        for (int p = 0; p < PX; ++p) simdgroup_load(a[p], base + (long)p * 8 * L_INC + cb, (ulong)L_INC);
                        for (int ob = 0; ob < NOB; ++ob)
                        {
                            simdgroup_half8x8 b;
                            simdgroup_load(b, wt + cb * kOC + ob * 8, kOC);
                            for (int p = 0; p < PX; ++p) simdgroup_multiply_accumulate(acc[p][ob], a[p], b, acc[p][ob]);
                        }
                    }
                }
            for (int p = 0; p < PX; ++p)
            {
                const int c = col0 + p * 8;
                if (c >= A.wOut) break;
                device half* dst = out + ((long)(2 * row + py + 1) * L_OUTWPAD + 2 * c + px + 1) * L_OUTC + L_OUTCOFF;
                epilogue<kOC>(acc[p], tile + sg * 8 * kOC, bias, dst, L_OUTC, 2, min(8, A.wOut - c), L_ACT, lane);
            }
        }
}

#define CONV_INST_PX(N, P) \
    template [[host_name("nss_conv_mma_" #N "_p" #P)]] kernel void nss_conv_mma<N, P>(constant MmaArgs&, const device half*, const device half*, const device float*, device half*, threadgroup nss_acc_t*, uint3, ushort, ushort);
#define UPCONV_INST_PX(N, P) \
    template [[host_name("nss_upconv_mma_" #N "_p" #P)]] kernel void nss_upconv_mma<N, P>(constant MmaArgs&, const device half*, const device half*, const device float*, device half*, threadgroup nss_acc_t*, uint2, ushort, ushort);
#define CONV_INST(N) \
    CONV_INST_PX(N, 1) CONV_INST_PX(N, 2) CONV_INST_PX(N, 4) UPCONV_INST_PX(N, 1) UPCONV_INST_PX(N, 2) UPCONV_INST_PX(N, 4)
CONV_INST(8)
CONV_INST(16)
CONV_INST(32)
CONV_INST(40)
CONV_INST(64)
