# Porting notes

## Original repository inspected

The uploaded DMENet archive contains a TensorFlow 1.15 / TensorLayer 1.11 implementation with these core files:

- `model.py`: VGG19-style encoder, UNet-style decoder, discriminator, binary net.
- `main.py`: training/evaluation graph construction, checkpoint load via TensorLayer `.npz`.
- `utils.py`: image I/O and preprocessing utilities.
- `config.py`: dataset/log/checkpoint paths.

## What is ported

- Inference graph for `VGG19_down(..., is_test=True)`.
- Inference graph for `UNet_up(...)`.
- Original RGB -> BGR/255/ImageNet-mean preprocessing.
- Original final defocus map output in `[0,1]`.
- Original sigma visualization formula from `main.py`.
- ComfyUI node wrapper returning focus/defocus/sigma masks.

## What is not ported

- Training.
- Discriminator and domain adaptation losses.
- Binary loss head.
- TensorBoard logging.
- Dataset generation/loading pipelines.
- MATLAB deconvolution scripts.

## Weight loading

The node can load original TensorLayer `.npz` checkpoints directly by mapping TensorFlow kernels `[H,W,in,out]` to PyTorch kernels `[out,in,H,W]` and loading BatchNorm statistics where names match.

If the original `.npz` variable names differ, run `convert_npz.py --strict` first; it will print missing names. The loader deliberately keeps `strict=False` in ComfyUI so minor naming differences are visible during conversion but do not hard-crash the UI unless no tensors are loaded.

## PyTorch compatibility

The implementation uses stable PyTorch 2.x APIs only: `nn.Conv2d`, `nn.BatchNorm2d`, `F.interpolate`, `F.pad`, `torch.inference_mode`, and `torch.load`/`torch.save`. It was syntax-checked and forward-tested on the sandbox's PyTorch build. It should run on PyTorch 2.11.x and newer.
