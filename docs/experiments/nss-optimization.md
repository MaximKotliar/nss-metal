# Experiment: making NSS cheaper in the hook

Branch: `experiment/helixsr-vs-metalfx`. Started 2026-11-02. Follows [helixsr-vs-metalfx.md](helixsr-vs-metalfx.md),
which ported Arm Neural Super Sampling (NSS) v1 to Metal and put it in place of MetalFX in TLOU2.

Goal: cut NSS mid's per-frame cost, which in TLOU2 (GPTK 4, DLSS Balanced, 1488×928 → 2560×1600) was 2.23 ms of GPU
time per upscale against MetalFX's 1.34 ms.

## Method

Measured without the game: `hook/test/nss_perf.sh` drives a real Metal 4 MetalFX scaler (`hook/build/nss_hook_test`,
the way GPTK 4 does) with the hook loaded, at TLOU2's Balanced size, 900 frames:

- **Upscale GPU time:** Metal 4 command-buffer timestamps around the upscale call (as in the game), median of the
  per-second medians.
- **Per-pass breakdown:** `LSFG_NSS_PROFILE=10`, a precise GPU timestamp after every dispatch on every 10th frame.
  Pass names are kernel names; a kernel used by several layers (e.g. `conv_mma_32`) is summed.
- **CPU:** NSS's time on the thread that calls the upscale (the game's render thread), per frame.
- **Correctness:** the last frame's PNG against the baseline's.

M4 Max, macOS 27.2, Xcode 27.2 beta 2.

## Baseline

| Mode | Upscale GPU (median / p95) | CPU per frame |
|---|---|---|
| MetalFX | 1.65 / 1.98 ms | – |
| NSS mid | 2.27 / 4.65 ms | 1.35 ms |
| NSS high | 6.44 / 6.89 ms | 2.20 ms |

(In the game: MetalFX 1.34, mid 2.23, high 6.13 ms. The harness agrees for NSS; MetalFX costs a bit more here.)

NSS mid by pass (2.17 ms): post 0.596, `conv_mma_32` 0.410 (six layers), pre_half 0.293, `conv_mma_16` 0.252,
`upconv_mma_16` 0.155, `upconv_mma_8` 0.140, game_output 0.126, `conv_mma_64` 0.088, game_prep_d 0.065, depth scatter,
disocclusion and clear 0.05 together.

NSS high by pass (6.70 ms): `conv_mma_16` 1.478, post 1.457, `conv_mma_32` 1.435, `upconv_mma_8` 0.531,
`upconv_mma_16` 0.520, pre 0.517, `conv_mma_64` 0.274, `conv_mma_40` 0.200, game_output 0.126, game_prep_d 0.124.

The CPU time is mostly the offset LUT, rebuilt every frame (16,000 tiles at 1.72×) under Rosetta.

## Experiments

### 1. Offset LUT on the GPU — kept

The LUT (which low-res taps and KPN channels each output pixel's filter uses) depends on the frame's jitter, so it was
rebuilt on the CPU every frame: at 1.72× the reduced tile is 100×160 high-res pixels, 36 candidate taps each, under
Rosetta, on the game's render thread. A cache per jitter value would need 1.5–3.5 MB per entry (DLSS cycles ~24
jitter values at this ratio), so instead three small kernels build it in the same command buffer: clear, scatter
(which low-res pixel lands where), and per-tile top-k selection.

| | GPU | CPU per frame | Output |
|---|---|---|---|
| before | 2.27 ms | 1.35 ms | |
| after | 2.28 ms (+0.03 ms of LUT passes) | **0.02 ms** | identical (0 pixels differ) |

Doesn't show in GPU numbers, but it frees ~1.3 ms of the game's render thread per frame, which matters whenever a
game is CPU-bound.

### 2. Post-process writes the game's output — kept

A separate pass read NSS's result back and wrote the game's output texture (undoing the exposure scaling). Post-process
now writes both from registers.

| | GPU | Output |
|---|---|---|
| before | 2.28 ms | |
| after | **2.15 ms** (the 0.126 ms pass is gone; post-process unchanged at 0.60) | identical |

### 3. Nearest depth computed once per input pixel — rejected

Mid's pre-process runs the 4×4 nearest-depth search up to nine times per process pixel (its own pixel, four history
taps, four packed offsets). A separate pass computed each input pixel's nearest depth and offset once for lookup.

| | pre_half | extra pass | GPU total |
|---|---|---|---|
| before | 0.294 ms | – | 2.15 ms |
| after | 0.296 ms | 0.041 ms | 2.20 ms |

Identical output, but no gain: the repeated searches read the same few cache lines and weren't what pre-process waits
on. Reverted.

### 4. Interleaved derivative state — kept

Pre-process bilinearly samples last frame's 4-channel derivative state (the moiré/flicker detector) at 1 position
(high) or 5 (mid/low). Stored as four planes, each sample was 16 scattered loads. Stored as one float4 per pixel, it's 4.

| | pre-process | GPU total | Output |
|---|---|---|---|
| mid before | 0.296 ms | 2.15 ms | |
| mid after | 0.259 ms | **2.09 ms** | identical |
| high before | 0.517 ms | 6.44 ms | |
| high after | 0.516 ms | 6.10 ms | 3.3% of pixels differ by 1/255 (summation order) |

(High's gain is mostly experiments 1–2; its pre-process samples the state only once.)

### 5. Offset LUT packed into one uint per tap — kept

Post-process read 6 floats per filter tap from 6 separate planes of the LUT (24 loads per pixel for mid's 4 taps, 54
for high's 9). They're small integers, so one 32-bit word per tap holds them all.

| | post-process | GPU total | Output |
|---|---|---|---|
| mid before | 0.600 ms | 2.09 ms | |
| mid after | 0.627 ms (noise) | 2.14 ms (noise) | identical |
| high before | 1.402 ms | 6.10 ms | |
| high after | **1.091 ms** | **5.92 ms** | unchanged |

### 6. Temporal and feedback tensors sampled with half4 loads — rejected

Both are 8-channel half tensors sampled channel by channel (4 loads per channel per sample). One half4 load per tap
covers 4 channels. No measurable change (post 0.625 ms, pre 0.275 ms), and a 1/255 rounding difference on 0.7% of
pixels: the extra loads hit the same cache lines and were already almost free. Reverted.

Not tried: indirect command buffers to cut encoding cost. After experiment 1, NSS's CPU time per frame is 0.02 ms (about
20 dispatches encoded into the game's command buffer), so there's nothing left there for them to save.

### 7. Several pixel tiles per simdgroup, tuned per layer — kept

Each conv simdgroup computed 8 output pixels and reloaded every weight tile for them. Computing PX × 8 pixels per
simdgroup lets each weight load serve PX tiles, but leaves fewer simdgroups to fill the GPU. A fixed PX (4 for ≤32
output channels, 2 for 64) helped the big layers and hurt the small low-res ones:

| | `conv_mma_16` | `conv_mma_32` | `conv_mma_64` | total |
|---|---|---|---|---|
| mid, PX 1 | 0.244 | 0.405 | 0.089 | 2.17 ms |
| mid, fixed PX | 0.156 | 0.509 | 0.188 | 2.16 ms |
| high, PX 1 | 1.480 | 1.443 | 0.274 | 5.92 ms |
| high, fixed PX | 0.940 | 1.213 | 0.514 | 5.69 ms |

So the engine now times every conv layer with PX = 1, 2 and 4 when it's set up (8 dispatches each, after a warm-up
round) and keeps the fastest. Setup takes ~0.3 s instead of ~10 ms, once per size/quality.

| | picked PX (layers 0–13, upsample convs excluded) | GPU total | Output |
|---|---|---|---|
| mid | 4 4 1 1 1 1 1 1 2 4 4 | **2.00 ms** | identical |
| high | 4 4 4 4 1 1 1 4 1 4 2 | **5.42 ms** | unchanged |

### 8. Tuned pixel tiles for the upsample-convs too — kept

The three convolutions that read a 2× upsample (computed as four 2×2 phase convolutions on the low-res input) got the
same PX template and per-layer timing.

| | upsample-convs | GPU total | Output |
|---|---|---|---|
| mid before | 0.295 ms | 2.00 ms | |
| mid after | 0.182 ms | **1.99 ms** (other layers' timings shifted) | identical |
| high before | 1.050 ms | 5.42 ms | |
| high after | 0.567 ms | **5.12 ms** | unchanged |

### 9. Post-process: 2×2 output pixels per thread — rejected

Neighbouring output pixels share most of their low-res inputs at 1.5–2×, so one thread computed a 2×2 block. Post
went 0.625 → 0.634 ms (mid), 1.09 → 1.10 ms (high): the shared loads already hit the cache. Reverted. Post-process
(0.63 ms mid, 1.09 ms high) stays the largest single pass; its cost isn't in the loads tried here.

### 10. Independent dispatches share a barrier — kept

On Metal 4 every dispatch waited on a barrier for the one before it. The offset-LUT chain (clear, scatter, select)
and the depth chain (clear, scatter, low-res disocclusion) don't depend on each other, so they now go in pairs, and
the KPN head and the next upsample-conv (both only read the same tensor) run together: 4 fewer barriers per frame.

| | GPU total | Output |
|---|---|---|
| mid | 1.99 → **1.94 ms** | identical |
| high | 5.12 → **5.02 ms** | unchanged |

Metal 3 path checked too: still pixel-identical to Metal 4.

## Round 2: Apple-GPU specifics, target 1.5 ms for mid

### Neural Engine — not pursued

The upscale is serial inside a frame: pre-process → network → post-process. Moving the network to the Neural Engine
means the GPU signals an event, a CPU thread runs the Core ML prediction, and the GPU waits for it. In the game's
Metal 4 queue that waiting stalls everything behind it, so the network would cost ~1 ms less GPU time but add its
latency (Core ML on the Neural Engine took ~1.7 ms for the mid network at a similar size in the bench, copies
included) to the frame, with the GPU idle meanwhile. The hook also runs under Rosetta (x86_64), where Neural Engine
access is uncertain. Kept on the GPU.

### 11. Threadgroup shape for post- and pre-process — rejected

16×16, 32×8, 8×8, 32×4, 64×2 and 8×32 for post-process: 0.624–0.628 ms, no difference. Every pipeline allows 1024
threads per threadgroup, so none is register-limited.

### 12. Post-process ablation — finding

Switching parts off one at a time (diagnostic constant, mid, 0.627 ms in all): filter taps 4 → 1 saves 0.281 ms,
history sampling 0.037, temporal parameters 0.057, each output write ~0.008. With no writes at all the compiler
removes the whole pass, so the rest can't be isolated that way. The filter loop did two integer divisions per tap
(KPN coordinates) and two integer modulos per pixel (LUT tile).

### 13. No integer division in post-process — kept

GPUs have no integer divide: it's a long instruction sequence. Replaced with a float reciprocal estimate plus an exact
integer fix-up (`fastDiv`): post 0.627 → 0.545 ms, mid **1.84 ms**. 1.3% of pixels differ by 1/255 from the
baseline (the compiler reorders nearby float math once the code changes shape).

### 14. Half-precision accumulation in the convs — rejected

`simdgroup_half8x8` accumulators instead of float: 1.88 vs 1.87 ms. No gain, kept float.

### 15. Conv layers specialized with function constants — kept

Each layer's chosen kernel compiled with its shape (input channels, channel strides and offsets, stride, activation,
row lengths) as Metal function constants: fixed loop counts, constant address math. Convs 0.69 → 0.54 ms, mid 1.87 →
**1.77 ms**, output unchanged. Kernels that declare function constants must be made through
`newFunctionWithName:constantValues:` even for the generic variant.

### 16. Upsample-convs specialized too — kept

0.18 → 0.10 ms, mid **1.62 ms**.

### 17. Post-process specialized — kept

Its sizes and quality switches (taps, sparse filter, half-res motion decode, Catmull-Rom history) as function
constants; only `reset` stays a runtime argument. The tap loop unrolls, the branches go. Post 0.54 → 0.42 ms, mid
**1.57 ms**; 0.7% of pixels differ by 1/255 from the baseline.

### 18. Mid pre-process specialized — kept

Sizes, render size and padding as constants (jitter stays runtime): pre 0.277 → 0.267 ms, mid **1.52 ms**.

### 19. Tuning pixel tiles on the specialized kernels — kept (for setup time)

All three variants of every layer are compiled specialized (in parallel) and timed; previously the generic variants
were timed and only the winner specialized. GPU time unchanged within noise (1.50–1.55 ms), setup 0.1–0.2 s.

### 20. Post-process ablation again — finding

On the specialized pass (0.420 ms): filter loop 0.116, history 0.049, temporal 0.033, each write 0.008. The rest
(~0.2 ms) is fixed per-pixel work.

### 21. Interleaved motion vectors — kept (neutral)

Motion stored as one float2 per pixel instead of two planes (read by depth scatter, disocclusion, pre and post).
No measurable change for mid; same output on both Metal paths.

### 22. Offset LUT cached per jitter value — kept

Packed (experiment 5), a mid LUT is 256 KB, so up to 64 of them (≤32 MB) are kept on the GPU, keyed by the jitter
value; DLSS cycles through a fixed set, so after the first cycle the three LUT passes (~0.03 ms) never run. Within
the run-to-run noise of short runs.

### 23. Reading the game's color directly — kept

Pre- and post-process now read the game's color texture instead of a copy: the input pass no longer writes color
(0.06 → 0.026 ms). The exposure factor goes in as NSS's own exposure input (as in Arm's reference), so history stays in
the game's color space and the output needs no division. Colour sampling is clamped to the content region, since
TLOU2 renders into the top-left of a larger texture. (A missing 4-byte pad in the host struct, the same Metal float2
alignment trap as before, briefly produced wrong output that measured *faster*: always check the image.)

### 24. Mid pre-process ablation — finding

0.264 ms total: moiré detector 0.024, three extra history taps 0.033, packed-offset searches 0.016, the rest 0.216
(mostly its tensor reads and writes). Nothing large left to cut.

The network's big layers now run at ~9–13 TFLOPS (e.g. conv1, 1.6 GFLOP in 0.139 ms), close to this GPU's peak. The
remaining lever there is less arithmetic (Winograd convolution, ~2.25× fewer multiplies for 3×3 layers).

### 25. Winograd F(2×2, 3×3) for the stride-1 layers — rejected

The big convs run at 9–13 TFLOPS, so the remaining lever looked like less arithmetic: Winograd computes each 2×2
output tile from 16 transformed positions instead of 36 taps (2.25× fewer matrix multiplies). Weights transformed on
the host (U = G g Gᵀ), one simdgroup per 8 tiles, offered to the tuner as a fourth variant next to 1/2/4 pixel tiles
for the eight stride-1 layers. Output was correct (within 1/255). Per-layer times (µs per dispatch, direct px1 vs
Winograd), through four versions:

| Version | layer 1 (32→32) | layer 11 (48→16) | layer 12 (16→16) |
|---|---|---|---|
| inputs staged in threadgroup memory | 139 vs 20,963 | 92 vs 662 | 37 vs 129 |
| no staging, one value per lane, ~1 KB threadgroup memory | 138 vs 25,880 | 92 vs 158 | 38 vs 83 |
| output channels in groups of 16 | 137 vs 264 | 92 vs 160 | 39 vs 84 |
| transformed blocks built in matrix registers (no barriers) | 204 vs 435 | 92 vs 185 | 39 vs 84 |

- Staging all input channels took ~18 KB of threadgroup memory: one threadgroup per GPU core, too little in flight.
- Keeping all 32/64 output channels' accumulators (4 positions × 4 blocks + 4) spilled registers: 50–150× slower.
- Even fixed, Winograd stays ~2× slower. The direct kernel isn't bound by the multiplies at these sizes (16–64
  channels) but by the loads and instruction issue around them; Winograd needs four input-block loads per transformed
  block, for 16 positions, repeated per output-channel group, which costs more than the multiplies it saves.

Removed (it would only add compile and tuning time at setup).

## Round 3: parity with MetalFX

Measured in the harness's game-like mode (`TEST_GAMELIKE=1`: the game's formats — RG11B10 color and output,
Depth32Float_Stencil8 depth, a 1×1 exposure texture, motion in UV units, an input-content region), 2000 frames,
1488×928 → 2560×1600. Same-mode MetalFX: **1.340–1.350 ms**. NSS mid at the start of the round: **1.445 ms**.
Image checks are against the previous step's final frame at the same frame count.

### 26. Temporal and feedback tensors read as hardware-filtered textures — kept

The last conv's 4 output channels were stored in an 8-channel tensor and read with software bilinear (4 taps per
channel: 12 loads in post, 16 in pre). The tensor now has 4 channels (the conv epilogue skips the padding
channels), and an RGBA16F linear texture view over the same buffer is sampled with one hardware-filtered read; the
tensor's zero border gives the zero-outside behaviour pre needs, and a clamped coordinate gives post's
clamp-to-edge.

- post 0.418 → 0.393 ms; total **1.445 → 1.420 ms (−0.025)**.
- Not bit-exact: the filter unit's weight precision. Max 4/255 on 2.2% of pixels, 65 dB against the exact output.

### 27. Game prep and depth scatter fused (mid/low) — kept

game_prep read the game's depth and motion, wrote NSS's copies, and three more passes followed: clear of the
scatter target, the scatter, and the low-res disocclusion, which re-read 16 depths and motions per thread. Now one
pass per 4×4 block (when the render size is 4× the quarter-res grid, which DLSS's sizes are) reads the game's
textures, writes the copies, scatters, and keeps the block's closest depth, its motion and its view-depth range for
the disocclusion, which reads just that. The scatter target is double-buffered and the disocclusion clears the other
one for the next frame, so the clear pass is gone too (first frame only). Same candidate order as the separate
passes, so ties resolve the same way.

- game_prep 0.027 + depth_clear 0.003 + depth_scatter_q 0.027 + disocclusion_lq 0.017 → prep_q 0.030 + disocclusion 0.007,
  and two fewer barriers.
- Total **1.420 → 1.380 ms (−0.040)**. Bit-exact (also against `LSFG_NSS_NOFUSE=1`, the old path, in both
  harness modes).

### 28. Derivative state as a hardware-filtered texture — kept (RGBA16F)

pre reads the previous derivative state five times per pixel (reprojected, plus 4 neighbours for mid/low), each a
software bilinear over float4. It's now written and sampled through a linear texture over the same buffer.

| Format | Total | pre_half | vs exact |
|---|---|---|---|
| RGBA32F (data unchanged, filter-weight precision only) | 1.350 ms | 0.221 | 72 dB, max 4/255 on 0.7% |
| **RGBA16F** | **1.340 ms** | 0.211 | 71 dB, max 4/255 on 0.9% |

- Total **1.380 → 1.340 ms (−0.040)**, RGBA16F. `LSFG_NSS_DERIV32=1` selects RGBA32F.

### 29. Dilated motion written by pre, read by post (mid/low) — kept

post decoded a packed nearest-depth offset, then read the motion at that neighbour, then sampled history: three
dependent loads per output pixel. pre already finds the offset for each render pixel, so it now writes that
neighbour's motion directly (float2, exact) and post reads it in one load. The packed offsets are no longer
written for mid/low.

- post 0.390 → 0.365, pre_half 0.211 → 0.230 (the extra writes).
- Total **1.340 → 1.330 ms (−0.010)**. Bit-exact.

### Round 3 total

| After | NSS mid | NSS high | MetalFX (same mode) |
|---|---|---|---|
| start of round (game-like mode) | 1.445 ms | – | 1.340 |
| 26. texture views for temporal/feedback | 1.420 | – | |
| 27. fused prep + scatter | 1.380 | – | |
| 28. derivative state as RGBA16F texture | 1.340 | – | |
| 29. dilated motion from pre | **1.330** | 3.55 | 1.340–1.350 |

Mid reaches MetalFX's GPU time in the harness, −0.115 ms (−8%) this round. Low: 1.30 ms. High: 3.76 → 3.55 ms
(it shares 26 and 28). Combined drift from the exact output: under 0.004/255 mean, no visible change.

## Round 4: closing the in-game gap

In TLOU2 with the round-3 build: NSS mid 1.33 ms vs MetalFX 1.17 ms at DLSS Balanced, 1.73–1.78 vs 1.26 ms at
Quality (1712×1072). NSS's cost grows with the render size (the network runs at half of it); MetalFX's barely does.

Finding: the conv layers are bound by math, not memory. Counting multiply-adds per layer (mid at Balanced: 7.2 GFLOP
for the 14 layers), the big ones run at 12–13 TFLOPS (conv1, conv11), roughly 70–80% of the M4 Max GPU's peak; the
small low-res ones at 6–8. So fusing layers to save memory traffic, the plan going in, has little to win; less math
and better use of the GPU in the small layers do.

The machine was noisy during this round (WindowServer using 17–27% of the GPU with nothing running), so results are
from alternating runs of both settings: `LSFG_NSS_TILES` pins each layer's kernel choice ("PX" or "PXxSPLIT"),
`LSFG_NSS_NOPACK0` / `LSFG_NSS_NOSPLIT` turn the experiments off.

### 30. conv0 with packed taps — kept

conv0 has 12 input channels, padded to 16 for the 8×8 matrix ops: a quarter of its multiplies were by zero. The
network input tensor now has 12 channels (pre writes 25% less), and for conv0 the 3 horizontal taps × 12 channels,
which are contiguous in a channels-last row, are one run of 36 rounded up to 40: 5 matrix blocks per kernel row
instead of 6 (the 4 extra values hit zero weights). Function constant 19 selects it; the weights are packed again
for it.

- conv0 0.080 → 0.063 ms at Balanced, 0.129 → 0.089 at Quality; pre_half 0.229 → 0.222. Bit-exact.

### 31. Output-channel split for the small layers — kept

The 1/4- and 1/8-res layers have too few pixels to fill 40 cores (conv4–6: 5,400 pixels, under 700 simdgroups).
Each simdgroup can now compute OC/2 or OC/4 of the output channels, with the threadgroup's z picking the group, so
these layers get 2–4× the simdgroups. The setup tuner picks per layer among (pixel tiles, split): (1,1) (2,1) (4,1)
(1,2) (2,2) (1,4).

- The tuner's isolated timings were noisy enough to pick variants that are slower in the frame; it now takes the best
  of three rounds and leaves the simplest variant only for one ≥3% faster.
- Typical pick at Balanced: `4 4 1x4 1x4 1x2 1x2 1x4 1 1x4 1x2 1 2x2 4 4`. Bit-exact.

### Round 4 total

Alternating runs (game-like mode, 1500 frames), round-3 settings vs 30 + 31:

| Size | Round 3 | 30 + 31 | Gain |
|---|---|---|---|
| Balanced 1488×928 | 1.414 / 1.421 ms | 1.354 / 1.324 ms | **−0.08 ms** |
| Quality 1712×1072 | 1.743 / 1.769 ms | 1.636 / 1.641 ms | **−0.12 ms** |

(Absolute numbers include the background GPU load; the quiet-machine Balanced figure for round 3 was 1.33 ms, so
expect about 1.25 ms there.)

## Running total

Upscale GPU time at 1488×928 → 2560×1600 (Metal 4, harness), after each kept experiment:

| After | NSS mid | NSS high | NSS CPU / frame |
|---|---|---|---|
| baseline | 2.27 ms | 6.44 ms | 1.35 ms |
| 1. LUT on the GPU | 2.28 | – | 0.02 |
| 2. post writes game output | 2.15 | – | 0.02 |
| 4. interleaved derivative state | 2.09 | 6.10 | 0.02 |
| 5. packed LUT | 2.14 (noise) | 5.92 | 0.02 |
| 7. tuned pixel tiles | 2.00 | 5.42 | 0.02 |
| 8. tuned upsample-convs | 1.99 | 5.12 | 0.02 |
| 10. shared barriers | 1.94 | 5.02 | 0.02 |
| 13. no integer division | 1.84 | – | 0.02 |
| 15. specialized convs | 1.77 | – | 0.02 |
| 16. specialized upsample-convs | 1.62 | – | 0.02 |
| 17. specialized post-process | 1.57 | – | 0.02 |
| 18. specialized pre-process | 1.52 | 3.82 | 0.02 |
| 21–23. motion, LUT cache, game color | 1.50 | 3.76 | 0.02 |
| long runs (3000 frames ×2) | **1.48** (p95 1.53) | – | 0.02 |
| after 25 (Winograd removed), 3000 frames | **1.46** | – | 0.02 |

Overall: mid 2.27 → 1.48 ms (−35%), high 6.44 → 3.76 ms (−42%), CPU 1.35 → 0.02 ms. Mid's output is within 1/255 of
the baseline on 0.7% of pixels; high's within 1/255 on 3.3%. MetalFX in the same long runs: 1.67 ms.

MetalFX in the same harness measured 1.65 ms early on and 2.57 ms at the end (p95 3.9): the harness's MetalFX runs
vary more than NSS's, which stayed within ±0.05 ms between repeats. The game is the fair comparison.


## In game: TLOU2 after the optimizations

GPTK 4.0b3, DLSS Balanced (1488×928 → 2560×1600), game capped at 60 fps, same spot, switching live in LSFG Metal.app.
Per-second medians from the hook's log (GPU: timestamps around the upscale call; CPU: the call's time on the game's
render thread, under Rosetta):

| Upscaler | GPU per upscale (median / p95) | CPU per call | frames/s | before (first in-game run) |
|---|---|---|---|---|
| MetalFX | 1.18 / 1.21 ms | 0.128 ms | 58 | 1.34 ms GPU |
| NSS mid | 1.43–1.51 / 1.58–1.71 ms | 0.07–0.12 ms | 52–58 | 2.23 ms GPU |
| NSS high | 3.67–3.94 / 4.02–4.12 ms | 0.07–0.12 ms | 52–58 | 6.13 ms GPU |

- NSS mid now costs +0.25–0.33 ms of GPU time over MetalFX (it was +0.89), and less CPU than MetalFX's own encode.
- The game held ~58 frames/s on MetalFX and on NSS high early in the run, ~52 later on both NSS modes; with the game
  near its cap and the scene/thermals moving, that difference isn't attributable to NSS from this run.
- Setup when switching NSS on or changing quality: 0.14–0.3 s of compiling and tuning, once; the very first setup
  with a cold Metal shader cache took 1.3 s (a visible hitch).
