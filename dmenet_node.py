from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:  # Allows import outside ComfyUI for syntax tests.
    class _FolderPathsFallback:
        models_dir = os.path.join(os.getcwd(), "models")
    folder_paths = _FolderPathsFallback()  # type: ignore

import comfy.model_management as model_management

from .dmenet_pytorch import DMENet, load_dmenet_checkpoint


MODEL_DIR = Path(folder_paths.models_dir) / "dmenet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_CACHE: Dict[Tuple[str, str], DMENet] = {}


def list_checkpoints():
    files = []
    if MODEL_DIR.exists():
        for p in MODEL_DIR.iterdir():
            if p.suffix.lower() in {".npz", ".pt", ".pth"}:
                files.append(p.name)
    return sorted(files)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def _resize_by_max_side(x: torch.Tensor, max_side: int):
    if max_side <= 0:
        return x, x.shape[-2:]
    h, w = x.shape[-2:]
    side = max(h, w)
    if side <= max_side:
        return x, (h, w)
    scale = max_side / float(side)
    nh = max(16, int(round(h * scale / 16.0)) * 16)
    nw = max(16, int(round(w * scale / 16.0)) * 16)
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False), (h, w)


def get_model(checkpoint: str, device: torch.device) -> DMENet:
    ckpt_path = MODEL_DIR / checkpoint
    key = (str(ckpt_path.resolve()), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = DMENet()
    report = load_dmenet_checkpoint(model, ckpt_path, strict=False)
    if report.loaded == 0:
        raise RuntimeError(f"Failed to load model from {ckpt_path}. Please ensure that checkpoint is DMENet.npz/.pt/.pth.")
    model.eval().to(device=device, dtype=torch.float32)
    _MODEL_CACHE[key] = model
    return model


class DMENetFocusMapNode:
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("focus_map", "defocus_map", "sigma_map_7_norm")
    FUNCTION = "estimate"
    CATEGORY = "image/analysis"

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = list_checkpoints()
        if not ckpts:
            ckpts = ["put_DMENet_BDCS.npz_or_converted_pt_in_ComfyUI_models_dmenet"]
        return {
            "required": {
                "image": ("IMAGE",),
                "checkpoint": (ckpts,),
                "normalize": (["raw", "minmax_per_image"], {"default": "raw"}),
                "focus_gamma": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.05}),
                "defocus_gamma": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.05}),
                "smooth": ("INT", {"default": 0, "min": 0, "max": 63, "step": 2}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64})
            }
        }

    def estimate(self, image, checkpoint, normalize, focus_gamma, defocus_gamma, smooth, max_side):
        if checkpoint.startswith("put_"):
            raise FileNotFoundError(f"Please put DMENet_BDCS.npz or converted .pt in {MODEL_DIR}")

        device = model_management.get_torch_device()
        model = get_model(checkpoint, device)

        # ComfyUI IMAGE is [B,H,W,C], RGB, float32 in [0,1].
        x = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        orig_hw = x.shape[-2:]
        x_small, before_resize_hw = _resize_by_max_side(x, int(max_side))
        x_pad, valid_hw = _pad_to_multiple(x_small, 16)

        with torch.inference_mode():
            defocus = model(x_pad).clamp(0.0, 1.0)

        defocus = defocus[..., :valid_hw[0], :valid_hw[1]]
        if defocus.shape[-2:] != orig_hw:
            defocus = F.interpolate(defocus, size=orig_hw, mode="bilinear", align_corners=False)

        if normalize == "minmax_per_image":
            b = defocus.shape[0]
            flat = defocus.reshape(b, -1)
            mn = flat.min(dim=1)[0].reshape(b, 1, 1)
            mx = flat.max(dim=1)[0].reshape(b, 1, 1)
            defocus = (defocus - mn) / (mx - mn + 1e-6)

        if smooth > 0:
            k = int(smooth)
            if k % 2 == 0:
                k += 1
            defocus = F.avg_pool2d(defocus, kernel_size=k, stride=1, padding=k // 2).clamp(0.0, 1.0)

        defocus_map = defocus[:, 0]
        if defocus_gamma != 1.0:
            defocus_map = torch.pow(defocus_map.clamp(0, 1), float(defocus_gamma))

        focus_map = (1.0 - defocus[:, 0]).clamp(0.0, 1.0)
        if focus_gamma != 1.0:
            focus_map = torch.pow(focus_map, float(focus_gamma))

        # Original evaluation script maps defocus output to normalized sigma map as:
        sigma = (((defocus[:, 0] * 15.0) - 1.0) / 2.0).clamp(min=0.0) / 7.0
        sigma = sigma.clamp(0.0, 1.0)

        return (focus_map, defocus_map, sigma)


NODE_CLASS_MAPPINGS = {
    "DMENetFocusMap": DMENetFocusMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DMENetFocusMap": "DMENet Focus/Defocus Map",
}
