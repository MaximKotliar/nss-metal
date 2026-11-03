# NSSMetal

Arm's **Neural Super Sampling** (NSS v1) as a Metal temporal upscaler for Apple silicon: an Objective-C library
shaped like MetalFX's temporal scaler, Metal 3 and Metal 4 encoding, no Core ML or Neural Engine.

NSS is a small neural network (14 convolution layers) that predicts per-pixel filter weights and blending
parameters from the jittered low-resolution frame, its depth and motion, and the previous output. Compared with
MetalFX's temporal upscaler it keeps more fine detail and flickers less, at about the same GPU cost in its mid mode.

| Quality | What changes | GPU time, 1488×928 → 2560×1600, M4 Max |
|---|---|---|
| low | half-resolution network, bilinear history | ~1.25 ms |
| mid | half-resolution network, Catmull-Rom history | ~1.3 ms |
| high | full-resolution network, 9-tap filter | ~3.5 ms |

MetalFX's cost depends on how the scaler is set up: 0.5 ms created at the exact render size, 1.2–1.3 ms created at
output size with input-content properties (what D3DMetal games do, to allow dynamic resolution). See
[docs/optimization.md](docs/optimization.md) for how the time was brought down from 6.4 / 2.3 ms, experiment by
experiment, and [docs/quality.md](docs/quality.md) for image quality against MetalFX and how the port was checked.

## Get the weights

The weights are Arm's, under the [Arm AI Model Community License](https://huggingface.co/Arm/neural-super-sampling),
and are not in this repository. Download the release and convert it (needs Python with PyTorch):

```bash
git clone https://huggingface.co/Arm/neural-super-sampling
python3 tools/export_weights.py neural-super-sampling nss
```

This writes `nss/nss_midlow.bin` (low and mid) and `nss/nss_high.bin`. If you ship them, ship the license and its
notices with them.

## Use it

Swift Package Manager:

```swift
.package(url: "https://github.com/MaximKotliar/nss-metal", branch: "main")
// target dependency: .product(name: "NSSMetal", package: "NSSMetal")
```

Without SwiftPM, compile `Sources/NSSMetal/NSSUpscaler.m` (ARC) with `Sources/NSSMetal/include` on the header path,
link Metal and QuartzCore, and build the kernels with `scripts/build-metallib.sh nss/nss.metallib`.

```objc
#import <NSSMetal/NSSMetal.h>

NSSUpscalerDescriptor *d = [NSSUpscalerDescriptor new];
d.inputWidth = 1488, d.inputHeight = 928, d.outputWidth = 2560, d.outputHeight = 1600;
d.quality = NSSQualityMid;
d.resourceDirectory = [NSURL fileURLWithPath:@"/path/to/nss"];   // weights (+ nss.metallib, optional)
NSError *error;
NSSUpscaler *upscaler = [[NSSUpscaler alloc] initWithDevice:device descriptor:d error:&error];

// every frame, like MTLFXTemporalScaler
upscaler.colorTexture = color;          // render size
upscaler.depthTexture = depth;          // depth format or float
upscaler.motionTexture = motion;        // RG: (previous - current) * scale, in input pixels
upscaler.outputTexture = output;        // output size, shader-writable
upscaler.jitterOffsetX = jitterX;       // MetalFX sign convention
upscaler.jitterOffsetY = jitterY;
upscaler.motionVectorScaleX = scaleX;
upscaler.motionVectorScaleY = scaleY;
upscaler.reset = cameraCut;
[upscaler encodeToCommandBuffer:commandBuffer];
// Metal 4: [upscaler encodeToMTL4CommandBuffer:commandBuffer fence:fence];
```

Notes:

- Setup compiles kernel variants specialized for the size and times them on the GPU (0.15–0.3 s; longer on the
  first run while Metal's shader cache is cold, ~2 s without a prebuilt `nss.metallib`). Make one upscaler per size
  and quality.
- Optional: `exposureTexture` (1×1) and `preExposure`, as in MetalFX; `depthReversed`.
- `zNear`, `zFar`, `verticalFieldOfView` on the descriptor: NSS compares view-space depths. Rough values are fine.
- The input region is the top-left `inputWidth × inputHeight` of the input textures, so oversized textures work.

## Try it

```bash
scripts/build-metallib.sh nss/nss.metallib
swift run -c release nss-bench --resources nss --game-like --png out.png
swift run -c release nss-bench --metalfx --game-like          # MetalFX on the same scene
```

`nss-bench` renders a moving, jittered test scene, upscales it, prints the median and p95 GPU time of the upscale and
writes the last frame. `--quality low|mid|high`, `--size 1712x1072:2560x1600`, `--frames N`.

## Debug switches

Environment variables, read at setup:

| Variable | Effect |
|---|---|
| `NSS_PROFILE=N` | Metal 4: time every pass of every Nth frame; the breakdown goes to the log handler |
| `NSS_TUNE_LOG=1` | log the timing of every kernel variant tried at setup |
| `NSS_TILES="4 4 1x4 …"` | fix each conv layer's variant (pixel tiles × channel split) instead of timing them |
| `NSS_SPECIALIZE=0` | generic kernels, no function-constant specialization |
| `NSS_NOFUSE`, `NSS_NOPACK0`, `NSS_NOSPLIT`, `NSS_DERIV32` | turn individual optimizations off (see docs/optimization.md) |

Set a log handler with `+[NSSUpscaler setLogHandler:]` to see them.

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The pass schedule, offset table and pre/post-processing
are ported from Arm's Apache-2.0 [neural-graphics-model-gym](https://github.com/arm/neural-graphics-model-gym).
The weights are separate and under Arm's own license.
