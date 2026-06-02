"""PyTorch 2.x inference port of DMENet's defocus-map network.

This module ports the TensorFlow/TensorLayer inference graph from the original
DMENet repository:
  Deep Defocus Map Estimation Using Domain Adaptation, CVPR 2019.

Only inference is implemented. Training, domain adaptation losses, VGG
classification heads, and the MATLAB deconvolution code are intentionally not
ported for ComfyUI usage.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# TensorFlow/TensorLayer used BGR channel order and ImageNet mean in 0..255 space.
_VGG_MEAN_BGR = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float32)


def _same_pad_3x3(x: Tensor) -> Tensor:
    # TensorFlow PadLayer(..., "Symmetric") is closer to edge replication than
    # PyTorch ReflectionPad2d for a one-pixel pad because TF symmetric includes
    # the boundary value. ReplicationPad2d gives the same boundary inclusion.
    return F.pad(x, (1, 1, 1, 1), mode="replicate")


class TLConv2d(nn.Module):
    """Conv2d with an original TensorFlow variable scope name."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, tf_name: str, padding: str = "none") -> None:
        super().__init__()
        self.tf_name = tf_name
        self.padding = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == "symm1":
            x = _same_pad_3x3(x)
        return self.conv(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tf_conv: str, tf_bn: str, act: str = "lrelu") -> None:
        super().__init__()
        self.conv = TLConv2d(in_ch, out_ch, 3, tf_conv, padding="symm1")
        self.bn = nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True)
        self.bn.tf_name = tf_bn  # type: ignore[attr-defined]
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        if self.act == "lrelu":
            return F.leaky_relu(x, negative_slope=0.2, inplace=False)
        if self.act == "sigmoid":
            return torch.sigmoid(x)
        if self.act == "relu":
            return F.relu(x, inplace=False)
        if self.act == "none":
            return x
        raise ValueError(f"Unsupported activation: {self.act}")


class VGGEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # VGG19-style feature extractor used by DMENet. Names match TensorLayer
        # scopes under main_net/defocus_net/encoder.
        self.conv1_1 = TLConv2d(3, 64, 3, "main_net/defocus_net/encoder/conv1_1", "symm1")
        self.conv1_2 = TLConv2d(64, 64, 3, "main_net/defocus_net/encoder/conv1_2", "symm1")
        self.conv2_1 = TLConv2d(64, 128, 3, "main_net/defocus_net/encoder/conv2_1", "symm1")
        self.conv2_2 = TLConv2d(128, 128, 3, "main_net/defocus_net/encoder/conv2_2", "symm1")
        self.conv3_1 = TLConv2d(128, 256, 3, "main_net/defocus_net/encoder/conv3_1", "symm1")
        self.conv3_2 = TLConv2d(256, 256, 3, "main_net/defocus_net/encoder/conv3_2", "symm1")
        self.conv3_3 = TLConv2d(256, 256, 3, "main_net/defocus_net/encoder/conv3_3", "symm1")
        self.conv3_4 = TLConv2d(256, 256, 3, "main_net/defocus_net/encoder/conv3_4", "symm1")
        self.conv4_1 = TLConv2d(256, 512, 3, "main_net/defocus_net/encoder/conv4_1", "symm1")
        self.conv4_2 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv4_2", "symm1")
        self.conv4_3 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv4_3", "symm1")
        self.conv4_4 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv4_4", "symm1")
        self.conv5_1 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv5_1", "symm1")
        self.conv5_2 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv5_2", "symm1")
        self.conv5_3 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv5_3", "symm1")
        self.conv5_4 = TLConv2d(512, 512, 3, "main_net/defocus_net/encoder/conv5_4", "symm1")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

    @staticmethod
    def preprocess(rgb: Tensor) -> Tensor:
        # Input Comfy/PyTorch tensor is RGB, range [0, 1]. Original TF code:
        # rgb_scaled = rgb * 255; split RGB; concat BGR - VGG_MEAN.
        mean = _VGG_MEAN_BGR.to(device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        bgr = rgb[:, [2, 1, 0], :, :] * 255.0 - mean
        return bgr

    def forward(self, rgb: Tensor) -> List[Tensor]:
        x = self.preprocess(rgb)
        x = F.relu(self.conv1_1(x), inplace=False)
        x = F.relu(self.conv1_2(x), inplace=False)
        d0 = x
        x = self.pool(x)

        x = F.relu(self.conv2_1(x), inplace=False)
        x = F.relu(self.conv2_2(x), inplace=False)
        d1 = x
        x = self.pool(x)

        x = F.relu(self.conv3_1(x), inplace=False)
        x = F.relu(self.conv3_2(x), inplace=False)
        x = F.relu(self.conv3_3(x), inplace=False)
        x = F.relu(self.conv3_4(x), inplace=False)
        d2 = x
        x = self.pool(x)

        x = F.relu(self.conv4_1(x), inplace=False)
        x = F.relu(self.conv4_2(x), inplace=False)
        x = F.relu(self.conv4_3(x), inplace=False)
        x = F.relu(self.conv4_4(x), inplace=False)
        d3 = x
        x = self.pool(x)

        x = F.relu(self.conv5_1(x), inplace=False)
        x = F.relu(self.conv5_2(x), inplace=False)
        x = F.relu(self.conv5_3(x), inplace=False)
        x = F.relu(self.conv5_4(x), inplace=False)
        d4 = x
        return [d0, d1, d2, d3, d4]


class ResidualRefineBlock(nn.Module):
    def __init__(self, idx: int) -> None:
        super().__init__()
        base = "main_net/defocus_net/decoder/u0"
        self.res = nn.Sequential(OrderedDict([
            ("conv", TLConv2d(64, 64, 1, f"{base}/c_res{idx}", padding="none")),
            ("bn", nn.BatchNorm2d(64, affine=True, track_running_stats=True)),
        ]))
        self.res.bn.tf_name = f"{base}/b_res{idx}"  # type: ignore[attr-defined]
        self.c1 = ConvBNAct(64, 64, f"{base}/c{idx}_1", f"{base}/b{idx}_1", act="lrelu")
        self.c2 = ConvBNAct(64, 64, f"{base}/c{idx}_2", f"{base}/b{idx}_2", act="lrelu")

    def forward(self, x: Tensor) -> Tensor:
        r = self.res.bn(self.res.conv(x))
        r = F.leaky_relu(r, negative_slope=0.2, inplace=False)
        y = self.c2(self.c1(x))
        return y + r


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        p = "main_net/defocus_net/decoder"

        self.u4_aux_1 = ConvBNAct(512, 256, f"{p}/u4_aux/c1", f"{p}/u4_aux/b1", act="lrelu")
        self.u4_aux_2 = ConvBNAct(256, 1, f"{p}/u4_aux/c2", f"{p}/u4_aux/b2", act="sigmoid")

        self.u3_1 = ConvBNAct(512 + 512, 256, f"{p}/u3/c1", f"{p}/u3/b1", act="lrelu")
        self.u3_2 = ConvBNAct(256, 256, f"{p}/u3/c2", f"{p}/u3/b2", act="lrelu")
        self.u3_3 = ConvBNAct(256, 256, f"{p}/u3/c3", f"{p}/u3/b3", act="lrelu")
        self.u3_aux_1 = ConvBNAct(256, 128, f"{p}/u3_aux/c1", f"{p}/u3_aux/b1", act="lrelu")
        self.u3_aux_2 = ConvBNAct(128, 1, f"{p}/u3_aux/c2", f"{p}/u3_aux/b2", act="sigmoid")

        self.u2_1 = ConvBNAct(256 + 256, 128, f"{p}/u2/c1", f"{p}/u2/b1", act="lrelu")
        self.u2_2 = ConvBNAct(128, 128, f"{p}/u2/c2", f"{p}/u2/b2", act="lrelu")
        self.u2_3 = ConvBNAct(128, 128, f"{p}/u2/c3", f"{p}/u2/b3", act="lrelu")
        self.u2_aux_1 = ConvBNAct(128, 64, f"{p}/u2_aux/c1", f"{p}/u2_aux/b1", act="lrelu")
        self.u2_aux_2 = ConvBNAct(64, 1, f"{p}/u2_aux/c2", f"{p}/u2_aux/b2", act="sigmoid")

        self.u1_1 = ConvBNAct(128 + 128, 64, f"{p}/u1/c1", f"{p}/u1/b1", act="lrelu")
        self.u1_2 = ConvBNAct(64, 64, f"{p}/u1/c2", f"{p}/u1/b2", act="lrelu")
        self.u1_3 = ConvBNAct(64, 64, f"{p}/u1/c3", f"{p}/u1/b3", act="lrelu")
        self.u1_aux_1 = ConvBNAct(64, 32, f"{p}/u1_aux/c1", f"{p}/u1_aux/b1", act="lrelu")
        self.u1_aux_2 = ConvBNAct(32, 1, f"{p}/u1_aux/c2", f"{p}/u1_aux/b2", act="sigmoid")

        self.u0_init = ConvBNAct(64 + 64, 64, f"{p}/u0/c_init", f"{p}/u0/b_init", act="lrelu")
        self.u0_aux_1 = ConvBNAct(64, 32, f"{p}/u0_aux/c1", f"{p}/u0_aux/b1", act="lrelu")
        self.u0_aux_2 = ConvBNAct(32, 1, f"{p}/u0_aux/c2", f"{p}/u0_aux/b2", act="sigmoid")

        self.refine = nn.ModuleList([ResidualRefineBlock(i) for i in range(7)])
        self.uf1 = ConvBNAct(64, 64, f"{p}/uf/c1", f"{p}/uf/b1", act="lrelu")
        self.uf2 = ConvBNAct(64, 32, f"{p}/uf/c2", f"{p}/uf/b2", act="lrelu")
        self.uf3 = TLConv2d(32, 1, 3, f"{p}/uf/c3", padding="symm1")

    @staticmethod
    def _up_to(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        d0, d1, d2, d3, d4 = feats
        u4 = self.u4_aux_2(self.u4_aux_1(d4))

        x = self._up_to(d4, d3)
        x = torch.cat([x, d3], dim=1)
        x = self.u3_3(self.u3_2(self.u3_1(x)))
        u3 = self.u3_aux_2(self.u3_aux_1(x))

        x = self._up_to(x, d2)
        x = torch.cat([x, d2], dim=1)
        x = self.u2_3(self.u2_2(self.u2_1(x)))
        u2 = self.u2_aux_2(self.u2_aux_1(x))

        x = self._up_to(x, d1)
        x = torch.cat([x, d1], dim=1)
        x = self.u1_3(self.u1_2(self.u1_1(x)))
        u1 = self.u1_aux_2(self.u1_aux_1(x))

        x = self._up_to(x, d0)
        x = torch.cat([x, d0], dim=1)
        x = self.u0_init(x)
        u0 = self.u0_aux_2(self.u0_aux_1(x))

        for block in self.refine:
            x = block(x)

        x = self.uf3(self.uf2(self.uf1(x)))
        return torch.sigmoid(x), [u4, u3, u2, u1, u0]


class DMENet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = UNetDecoder()

    def forward(self, rgb: Tensor) -> Tensor:
        feats = self.encoder(rgb)
        out, _ = self.decoder(feats)
        return out


@dataclass
class LoadReport:
    loaded: int
    missing: List[str]
    unexpected: List[str]


class DMENetCheckpointError(RuntimeError):
    pass


def _read_npz(path: str | Path) -> Dict[str, np.ndarray]:
    raw = np.load(str(path), allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        data = {k: raw[k] for k in raw.files}
    else:
        raise DMENetCheckpointError(f"Unsupported checkpoint container: {type(raw)!r}")

    # Some older utilities save a single object array containing a dict.
    if len(data) == 1:
        only = next(iter(data.values()))
        if only.shape == () and isinstance(only.item(), dict):
            data = only.item()
    return {str(k): v for k, v in data.items()}


def _find_key(data: Mapping[str, np.ndarray], candidates: Iterable[str]) -> Optional[str]:
    keys = list(data.keys())
    for c in candidates:
        if c in data:
            return c
    # Be permissive about :0 suffixes and possible leading scopes.
    for c in candidates:
        c0 = c[:-2] if c.endswith(":0") else c
        for k in keys:
            k0 = k[:-2] if k.endswith(":0") else k
            if k0 == c0 or k0.endswith("/" + c0):
                return k
    return None


def _load_conv_from_npz(module: TLConv2d, data: Mapping[str, np.ndarray], missing: List[str]) -> int:
    base = module.tf_name
    w_key = _find_key(data, [f"{base}/W:0", f"{base}/weights:0", f"{base}/kernel:0", f"{base}.weight"])
    b_key = _find_key(data, [f"{base}/b:0", f"{base}/biases:0", f"{base}/bias:0", f"{base}.bias"])
    loaded = 0
    if w_key is None:
        missing.append(f"{base}/W:0")
    else:
        w = data[w_key]
        # TensorFlow conv kernels are [H, W, in, out]; PyTorch uses [out, in, H, W].
        if w.ndim == 4:
            w = np.transpose(w, (3, 2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(w)).to(dtype=module.conv.weight.dtype)
        if tuple(t.shape) != tuple(module.conv.weight.shape):
            raise DMENetCheckpointError(f"Shape mismatch for {base}/W: {tuple(t.shape)} vs {tuple(module.conv.weight.shape)}")
        module.conv.weight.data.copy_(t)
        loaded += 1
    if b_key is None:
        # TensorLayer Conv2d normally has a bias, but allow no-bias checkpoints.
        module.conv.bias.data.zero_()
    else:
        b = torch.from_numpy(np.ascontiguousarray(data[b_key])).to(dtype=module.conv.bias.dtype)
        if tuple(b.shape) != tuple(module.conv.bias.shape):
            raise DMENetCheckpointError(f"Shape mismatch for {base}/b: {tuple(b.shape)} vs {tuple(module.conv.bias.shape)}")
        module.conv.bias.data.copy_(b)
        loaded += 1
    return loaded


def _load_bn_from_npz(module: nn.BatchNorm2d, data: Mapping[str, np.ndarray], missing: List[str]) -> int:
    base = getattr(module, "tf_name", None)
    if base is None:
        return 0
    aliases = {
        "gamma": [f"{base}/gamma:0", f"{base}/Gamma:0", f"{base}.weight"],
        "beta": [f"{base}/beta:0", f"{base}/Beta:0", f"{base}.bias"],
        "moving_mean": [f"{base}/moving_mean:0", f"{base}/mean/EMA:0", f"{base}/moving_mean/EMA:0", f"{base}.running_mean"],
        "moving_variance": [f"{base}/moving_variance:0", f"{base}/variance/EMA:0", f"{base}/moving_variance/EMA:0", f"{base}.running_var"],
    }
    loaded = 0
    for name, candidates in aliases.items():
        key = _find_key(data, candidates)
        if key is None:
            if name == "gamma":
                module.weight.data.fill_(1.0)
            elif name == "beta":
                module.bias.data.zero_()
            elif name == "moving_mean":
                module.running_mean.zero_()
            elif name == "moving_variance":
                module.running_var.fill_(1.0)
            missing.append(candidates[0])
            continue
        arr = torch.from_numpy(np.ascontiguousarray(data[key])).to(dtype=module.weight.dtype)
        if name == "gamma":
            module.weight.data.copy_(arr)
        elif name == "beta":
            module.bias.data.copy_(arr)
        elif name == "moving_mean":
            module.running_mean.copy_(arr)
        elif name == "moving_variance":
            module.running_var.copy_(arr)
        loaded += 1
    return loaded


def load_dmenet_checkpoint(model: DMENet, checkpoint: str | Path, strict: bool = False) -> LoadReport:
    """Load either a converted PyTorch checkpoint or original TensorLayer npz.

    Supported formats:
      - .pt/.pth with a regular PyTorch state_dict or {"state_dict": ...}
      - original DMENet .npz saved by TensorLayer save_npz_dict
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))

    missing: List[str] = []
    unexpected: List[str] = []
    loaded = 0

    if checkpoint.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(str(checkpoint), map_location="cpu")
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        report = model.load_state_dict(sd, strict=strict)
        missing = list(report.missing_keys)
        unexpected = list(report.unexpected_keys)
        loaded = len(sd) if isinstance(sd, Mapping) else 0
        return LoadReport(loaded=loaded, missing=missing, unexpected=unexpected)

    if checkpoint.suffix.lower() != ".npz":
        raise DMENetCheckpointError(f"Unsupported checkpoint extension: {checkpoint.suffix}")

    data = _read_npz(checkpoint)
    used_keys = set()

    for m in model.modules():
        if isinstance(m, TLConv2d):
            before = len(missing)
            loaded += _load_conv_from_npz(m, data, missing)
            # approximate used key accounting is not critical for inference.
        elif isinstance(m, nn.BatchNorm2d):
            loaded += _load_bn_from_npz(m, data, missing)

    if strict and missing:
        raise DMENetCheckpointError("Missing checkpoint variables:\n" + "\n".join(missing[:100]))

    # Report variables that look like model variables but were not consumed is hard
    # with fuzzy matching. Keep unexpected empty for .npz to avoid false alarms.
    return LoadReport(loaded=loaded, missing=missing, unexpected=unexpected)


def convert_npz_to_pytorch(npz_path: str | Path, output_path: str | Path, strict: bool = False) -> LoadReport:
    model = DMENet()
    report = load_dmenet_checkpoint(model, npz_path, strict=strict)
    torch.save({"state_dict": model.state_dict(), "source_npz": str(npz_path), "load_report": report.__dict__}, str(output_path))
    return report
