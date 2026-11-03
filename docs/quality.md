# NSS vs MetalFX: quality, and how the port was checked

How Arm's Neural Super Sampling compares with MetalFX's temporal upscaler, how the Metal port was verified against
Arm's PyTorch reference, and what it did in a real game (The Last of Us Part II under Apple's Game Porting Toolkit,
where the game's DLSS calls become MetalFX and NSS was swapped in). Notes from November 2026; the later speedups are
in [optimization.md](optimization.md).

Arm publishes NSS v1, a temporal upscaler: a small U-Net (14 conv layers, ~160K parameters) that predicts per-pixel
filter weights and blend parameters, wrapped in shader pre/post-processing. Weights are on Hugging Face
(`Arm/neural-super-sampling`); the model code and the reference Slang shaders are Apache-2.0
(`arm/neural-graphics-model-gym`).

License of the weights (Arm AI Model Community License v1.0, read in full): use, modify, make derivatives and
distribute commercially, royalty-free. Conditions: ship a copy of the license and Arm's notices, mark modifications,
and carry its acceptable-use clause (no weapons, surveillance, unlawful use, etc.) into our own terms. So NSS can ship
inside an app. Note: the GLSL deployment shaders in the Hugging Face repo carry a "proprietary,
confidential" header, so the Metal port below is translated from the Apache-2.0 Slang reference instead.

### Step 1: Arm's PyTorch reference on the bench frames

The bench dumps each frame's inputs and MetalFX's output; NSS runs on the same frames through Arm's PyTorch pipeline.
NSS wants motion = current − previous (opposite of MetalFX), jitter as is. MetalFX re-checked at 2×: its calibrated
signs are still its best.

| PSNR / SSIM vs 8×8 supersampled truth | 1707×960 → 2560×1440 (1.5×) | 1280×720 → 2560×1440 (2×) |
|---|---|---|
| NSS high | **28.8 dB / 0.970** | **25.0 dB / 0.932** |
| NSS mid | – | 24.4 dB / 0.937 |
| NSS low | – | 24.1 dB / 0.932 |
| MetalFX | 24.9 dB / 0.949 | 22.6 dB / 0.896 |
| Bilinear | 24.0 dB / 0.933 | 21.7 dB / 0.885 |

NSS resolves the static chain-link fence that MetalFX blurs, and the moving wheel is sharper.

### Step 2: native Metal port ("high" quality)

All passes as Metal compute kernels: input conversion, depth scatter, pre-process, the network, post-process, plus the
per-frame offset LUT on the CPU. Checked stage by stage against the PyTorch reference on the same frame: every
intermediate matched to 4e-4 in float32 (one bug found that way: the nearest-depth offset was packed row/col swapped).

Network kernels, in order of what they bought (2×, whole upscale, M4 Max):

| Version | GPU time | Notes |
|---|---|---|
| naive per-pixel conv, float | 86 ms | correctness baseline |
| `simdgroup_matrix` implicit GEMM, channels-last with zero borders, float | 5.5 ms | |
| half activations/weights (float accumulate), upsample folded into the next conv as four 2×2 phase convs | 4.7 ms | max output change 0.02 |
| color and history as textures, hardware bilinear in post/pre-process | **3.8 ms** | −0.05 dB |

Final, all on the GPU, measured in the bench:

| | quality | GPU time (median) |
|---|---|---|
| 2×: NSS Metal | 25.0 dB / 0.932 | 3.8 ms (network ~2.7, post 0.7, pre 0.3) |
| 2×: MetalFX | 22.6 dB / 0.895 | 0.37 ms |
| 1.5×: NSS Metal | 28.5 dB / 0.966 | 6.6 ms |
| 1.5×: MetalFX | 24.9 dB / 0.946 | 0.48 ms |

The big conv layers run at 6–10 TFLOPS, so at "high" quality the network's floor on M4 Max is about 2 ms at 2×.
The remaining large lever is NSS's "mid"/"low" modes (network at half the input resolution: ~4× less work, −0.6 dB
in step 1), not ported yet.

Metal 4 tensor ops (Metal Performance Primitives `convolution2d`, NHWC/HWIO, matches our layout): compiles and ran
~25% faster than our kernel on the heaviest layer (0.49 vs 0.63 ms), but the output is wrong. Neither offsets nor
sliced tensors fix it; the op is undocumented beta (Xcode 27.2 beta 2). The large speedup from tensor ops is the M5's
GPU neural accelerators, not available on M4.

Conclusion: NSS on Metal beats MetalFX by 2.4 dB (2×) to 3.7 dB (1.5×) and could legally ship, at about 10× MetalFX's
GPU cost for "high". Next: port "mid"/"low", then try it in a real game.

### Step 3: "mid" and "low" modes on Metal

Ported the half-resolution path from the same Apache-2.0 Slang: quarter-resolution depth scatter, a separate low-res
disocclusion pass, half-resolution pre-process (four warped history taps averaged, plus the extra moiré / flat-flicker
detector), 4-bit packed nearest-depth offsets, and the sparse 2×2 filter in post-process (16 KPN channels, 4 taps).
Mid uses Catmull-Rom history sampling, low bilinear. Weights: the published int8 QAT export, run as float.
Checked against the PyTorch reference: pre-process exact; full output within 0.07 (mean 0.0002) once a struct
alignment bug was fixed (an `int2` after an odd number of ints is 8-byte aligned in Metal, not in C++).

| M4 Max, 64 frames | PSNR / SSIM | GPU time (median) | of which network |
|---|---|---|---|
| 2×: NSS high | 25.0 dB / 0.932 | 4.0 ms | 2.7 ms |
| 2×: NSS mid | 24.3 dB / 0.935 | **1.65 ms** | 0.8 ms |
| 2×: NSS low | 24.0 dB / 0.930 | 1.6 ms | 0.8 ms |
| 2×: MetalFX | 22.6 dB / 0.895 | 0.37 ms | |
| 1.5×: NSS high | 28.5 dB / 0.966 | 6.8 ms | 4.7 ms |
| 1.5×: NSS mid | **28.5 dB / 0.974** | **2.4 ms** | 1.3 ms |
| 1.5×: NSS low | 27.8 dB / 0.970 | 2.4 ms | 1.3 ms |
| 1.5×: MetalFX | 24.9 dB / 0.946 | 0.48 ms | |

Mid is the mode to use: +1.7 dB over MetalFX at 2× and +3.6 dB at 1.5×, for about 1.3–1.9 ms more GPU time.
At 1.5× it matches high's PSNR with better SSIM at a third of the cost. Low saves nothing over mid here (they differ
only in history sampling) and loses quality.

Next: try mid in a real game through the hook, which also shows whether the synthetic-scene gains hold up.

### Step 4: NSS in TLOU2, in place of MetalFX

The hook catches `-[<MetalFX temporal scaler> encodeToCommandBuffer:]` (D3DMetal turns the game's DLSS calls into
MetalFX temporal upscaling) and, when the game's profile says `"upscaler": "nss"`, encodes NSS into the same command
buffer instead (`hook/unix/lsfg_nss.m`, kernels in `hook/unix/nss/nss.metal`). Both GPTK paths:

- GPTK 3: Metal 3 command buffer, a plain compute encoder.
- GPTK 4 (what TLOU2 uses here): Metal 4 command buffer. Bindings through an argument table, constants through an
  upload buffer, a barrier before every dispatch, our resources and the game's textures in residency sets, and the
  scaler's fence waited on and updated like MetalFX does. Output is pixel-identical to the Metal 3 path.

What TLOU2 hands MetalFX (GPTK 4.0b3, DLSS): a scaler made at 2560×1600 → 2560×1600 with "input content properties"
(the real render size comes per frame: 1712×1072 at Quality, 1488×928 at Balanced), color RG11B10Float, depth
Depth32Float_Stencil8 (forward, not reversed), motion RG16Float in UV units (scale = render size), a 1×1 R32Float
exposure texture. The DLSS preset (e.g. "Transformer") and sharpness never reach MetalFX, so they don't matter.

Upscale GPU time in game, from Metal 4 command-buffer timestamps around the upscale call (median / p95 per second),
DLSS Balanced, 1488×928 → 2560×1600, ~50 s per mode, game capped at 60 fps:

| Upscaler | Upscale GPU time | Game fps |
|---|---|---|
| MetalFX | 1.34 / 1.37 ms | 60 |
| NSS low | 2.19 / 2.20 ms | 60 |
| NSS mid | 2.23 / 2.24 ms | 60 |
| NSS high | 6.13 / 6.17 ms | 51 |

- NSS mid costs +0.9 ms over MetalFX and keeps the cap; the user's verdict on the image: "looks really good".
- Low costs the same as mid (as in the bench), so mid is the setting to use. High needs GPU headroom.
- The profile setting, quality and the app's "Upscaling" card switch live while the game runs.
