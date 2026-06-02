# ComfyUI-DMENet

ComfyUI custom node for DMENet-style single-image defocus-map estimation.

This is a PyTorch 2.x inference-only port of the original TensorFlow 1.15 / TensorLayer DMENet network from:

> [Deep Defocus Map Estimation Using Domain Adaptation, CVPR 2019.](https://github.com/codeslake/DMENet)

## What the node outputs

`DMENet Focus/Defocus Map` returns:

- `focus_map` (`MASK`): `1 - defocus_map`, where 1 means likely in focus and 0 means likely out of focus.
- `defocus_map` (`MASK`): DMENet direct output, where higher values mean stronger estimated defocus.
- `sigma_map_7_norm` (`MASK`): matches the original evaluation script's normalized sigma visualization.

## Install

Clone this repository to:

```text
ComfyUI/custom_nodes/ComfyUI-DMENet/
```

Place the original checkpoint or a converted PyTorch checkpoint under:

[Download converted checkpoint here.](https://huggingface.co/MoRanYue/ComfyUI-DMENet)

```text
ComfyUI/models/dmenet/DMENet_BDCS.npz
```

or:

```text
ComfyUI/models/dmenet/DMENet_BDCS.pt
```

The node can load the original TensorLayer `.npz` directly. For faster startup, convert once:

```bash
cd ComfyUI/custom_nodes/ComfyUI-DMENet
python convert_npz.py ../../models/dmenet/DMENet_BDCS.npz ../../models/dmenet/DMENet_BDCS.pt
```

## Notes

- This port implements inference only. Training, GAN/domain-adaptation losses, TensorBoard logging, MATLAB deconvolution, and dataset code are not ported.
- The original DMENet preprocessing is preserved: RGB image in `[0,1]` is converted to BGR `0..255` and ImageNet VGG mean is subtracted.
- The original code crops inputs to multiples of 16. This node instead pads to a multiple of 16 and crops back, so it can preserve ComfyUI image size.
- `normalize=minmax_per_image` is useful for control masks; `normalize=raw` is closer to the original network output.
- Use `smooth` and avoid early thresholding when the mask controls super-resolution or diffusion strength; soft maps reduce halos.

## License

The original DMENet repository is licensed under GNU AGPLv3 and marked non-commercial in its README/license notice. This derived port should be treated as GNU AGPLv3 and non-commercial unless you obtain a separate license from the original authors.
