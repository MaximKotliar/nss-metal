# NSS v1 weights (Arm)

These files are Arm's Neural Super Sampling model weights in the format NSSMetal loads. They are **not** covered by
this repository's Apache-2.0 license. They are licensed by Arm Limited under the **Arm AI Model Community License**
(`Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf`, included here). By using, copying, modifying or distributing
them you agree to that license.

## Source

[huggingface.co/Arm/neural-super-sampling](https://huggingface.co/Arm/neural-super-sampling), commit
`39ed48a9beef9dc7250c0fbea5a4c8d6d49666b8` (model version 1.0.1):

| File here | From |
|---|---|
| `nss_midlow.bin` | `nss_v1_0_1_mid_low_int8.pt` (used for quality low and mid) |
| `nss_high.bin` | `nss_v1_0_1_high_fp32.pt` (used for quality high) |

## Changes made (Arm AI Model Community License, clause 3.2)

These are modified versions of Arm's files:

- Converted from PyTorch checkpoints to raw little-endian float32: for each of the 14 convolution layers in network
  order, the weights `[out][in][3][3]` and then the biases `[out]`. No other data is kept.
- `nss_midlow.bin` comes from Arm's 8-bit quantized (int8, quantization-aware) checkpoint, stored here as float32.
- The values themselves are unchanged. `tools/export_weights.py` in this repository reproduces both files from Arm's
  release.

## Arm's notices

Kept as Arm provides them:

- `Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf`: the license.
- `ARM_LICENSE_POINTER.txt`: Arm's `LICENSE` file from the release.
- `third_party_licenses_and_copyright_notices.txt`: Arm's third-party notices from the release.
- From Arm's model card: the model source code is under Apache-2.0
  ([neural-graphics-model-gym](https://github.com/arm/neural-graphics-model-gym)); Arm® is a registered trademark of
  Arm Limited (or its subsidiaries) in the US and/or elsewhere.

## Acceptable use

Use of these weights, and of anything derived from them or built with them (including apps that ship them), is
subject to the acceptable-use terms in [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md).
