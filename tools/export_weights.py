"""Turn Arm's NSS v1 checkpoints into the weight files NSSMetal loads.

    python3 tools/export_weights.py HF_DIR OUT_DIR

HF_DIR: a download of https://huggingface.co/Arm/neural-super-sampling (Arm AI Model Community License; read it
before using or shipping the weights). Writes:
    OUT_DIR/nss_midlow.bin   from nss_v1_0_1_mid_low_int8.pt   (quality low and mid)
    OUT_DIR/nss_high.bin     from nss_v1_0_1_high_fp32.pt      (quality high)
Format: raw float32; for each of the 14 conv layers in network order, weight [oc][ic][3][3] then bias [oc].

Needs PyTorch. Checkpoints are loaded with weights_only=True (no code from them runs).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ORDER = ["conv2d_0", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6", "conv2d_7",
         "conv2d_8", "kpn_params", "conv2d_9", "conv2d_10", "conv2d_11", "temporal_params_out_conv"]
FILES = {"nss_midlow.bin": "nss_v1_0_1_mid_low_int8.pt", "nss_high.bin": "nss_v1_0_1_high_fp32.pt"}


def export(checkpoint: Path, out: Path) -> None:
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    if "autoencoder.conv2d_0.conv2d.weight" in sd:
        get = lambda n, k: sd[f"autoencoder.{n}.conv2d.{k}"]
    else:  # quantization-aware graph export: constants in execution order
        get = lambda n, k: sd[f"autoencoder._param_constant{2 * ORDER.index(n) + (k == 'bias')}"]
    parts = []
    for n in ORDER:
        w, b = get(n, "weight").float().numpy(), get(n, "bias").float().numpy()
        assert w.ndim == 4 and w.shape[2:] == (3, 3) and b.shape == (w.shape[0],), (n, w.shape, b.shape)
        parts += [w.ravel(), b.ravel()]
    np.concatenate(parts).astype(np.float32).tofile(out)
    print(f"{out.name}: {sum(p.size for p in parts)} values from {checkpoint.name}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for out, ckpt in FILES.items():
        if not (src / ckpt).exists():
            sys.exit(f"missing {src / ckpt}")
        export(src / ckpt, dst / out)


if __name__ == "__main__":
    main()
