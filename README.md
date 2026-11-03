# NSSMetal

Arm's **Neural Super Sampling** (NSS v1) as a Metal temporal upscaler for Apple silicon: an Objective-C library
shaped like MetalFX's temporal scaler, Metal 3 and Metal 4 encoding, no Core ML or Neural Engine.

NSS is a small neural network (14 convolution layers) that predicts per-pixel filter weights and blending
parameters from the jittered low-resolution frame, its depth and motion, and the previous output. Compared with
MetalFX's temporal upscaler it keeps more fine detail and flickers less, at about the same GPU cost in its mid mode.

| Quality | What changes | GPU time, 1488×928 → 2560×1600, M4 Max |
|---|---|---|
| fast | quarter-resolution network (mid's weights), Catmull-Rom history; as good as mid in motion, fine static detail at MetalFX's level | ~0.8 ms |
| low | half-resolution network, bilinear history | ~1.25 ms |
| mid | half-resolution network, Catmull-Rom history | ~1.3 ms |
| high | full-resolution network, 9-tap filter | ~3.5 ms |

MetalFX's cost depends on how the scaler is set up: 0.5 ms created at the exact render size, 1.2–1.3 ms created at
output size with input-content properties (what D3DMetal games do, to allow dynamic resolution). See
[docs/optimization.md](docs/optimization.md) for how the time was brought down from 6.4 / 2.3 ms, experiment by
experiment, and [docs/quality.md](docs/quality.md) for image quality against MetalFX and how the port was checked.

## Arm's NSS

NSS is Arm's model; this is an independent Metal port of it, not an Arm product.

- Model card and weights: [huggingface.co/Arm/neural-super-sampling](https://huggingface.co/Arm/neural-super-sampling)
- Paper: [Neural Super Sampling (2025)](https://huggingface.co/Arm/neural-super-sampling/blob/main/2025-neural-super-sampling.pdf)
- Reference implementation and training (Apache-2.0): [arm/neural-graphics-model-gym](https://github.com/arm/neural-graphics-model-gym), `usecases/nss`
- [NSS guide](https://developer.arm.com/documentation/111009/latest/) and Arm's [Neural Graphics Development Kit](https://developer.arm.com/mobile-graphics-and-gaming/neural-technology)

## Weights

Arm's trained weights come with the package, in [`Sources/NSSMetal/Weights`](Sources/NSSMetal/Weights) (about 1.2 MB),
converted to the format NSSMetal loads. **They are not under this repository's Apache-2.0 license:** they're Arm's,
under the [Arm AI Model Community License](Sources/NSSMetal/Weights/Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf).
It allows commercial use and redistribution, provided that whoever distributes them:

- ships the license and Arm's notices with them,
- marks changes (the conversion is described in the folder's README),
- binds their users to the license's acceptable-use clause, and tells them their product is covered by it
  ([ACCEPTABLE_USE.md](Sources/NSSMetal/Weights/ACCEPTABLE_USE.md)).

To convert Arm's release yourself (needs Python with PyTorch):

```bash
git clone https://huggingface.co/Arm/neural-super-sampling
python3 tools/export_weights.py neural-super-sampling nss     # then set resourceDirectory to nss/
```

## Use it

Swift Package Manager:

```swift
.package(url: "https://github.com/MaximKotliar/nss-metal", branch: "main")
// target dependency: .product(name: "NSSMetal", package: "NSSMetal")
```

Without SwiftPM, compile `Sources/NSSMetal/NSSUpscaler.m` (ARC) with `Sources/NSSMetal/include` on the header path,
link Metal and QuartzCore, build the kernels with `scripts/build-metallib.sh nss/nss.metallib`, copy the two `.bin`
files from `Sources/NSSMetal/Weights` next to it, and set `resourceDirectory` to that folder.

```objc
#import <NSSMetal/NSSMetal.h>

NSSUpscalerDescriptor *d = [NSSUpscalerDescriptor new];
d.inputWidth = 1488, d.inputHeight = 928, d.outputWidth = 2560, d.outputHeight = 1600;
d.quality = NSSQualityMid;
// weights: bundled (Swift package); or d.resourceDirectory = a folder with them (+ nss.metallib, optional)
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
swift run -c release nss-bench --game-like --png out.png     # bundled weights, kernels compiled at startup
swift run -c release nss-bench --metalfx --game-like          # MetalFX on the same scene
```

`nss-bench` renders a moving, jittered test scene, upscales it, prints the median and p95 GPU time of the upscale and
writes the last frame. `--quality fast|low|mid|high`, `--size 1712x1072:2560x1600`, `--frames N`.

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

The code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The pass schedule, offset table and
pre/post-processing are ported from Arm's Apache-2.0
[neural-graphics-model-gym](https://github.com/arm/neural-graphics-model-gym).

The weights in `Sources/NSSMetal/Weights` are Arm's, under the Arm AI Model Community License, with its acceptable-use
terms: see that folder.
