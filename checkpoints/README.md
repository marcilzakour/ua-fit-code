# Checkpoints

The released UA-Fit checkpoint is not stored in git (240 MB). Download it and
place it here so that the evaluation can find it:

```
checkpoints/
  uafit_w40/
    DOVFManoMVEpi.pth.tar     # HRNet-W40 DOVF backbone + uncertainty head + shape head
```

Then run:

```bash
python scripts/eval.py --ckpt checkpoints/uafit_w40 --config configs/uafit_eval.yaml
```

The checkpoint file name must be `DOVFManoMVEpi.pth.tar` (the loader keys the
file by the model class name). The state dict is loaded with `strict=False`; a
small refiner sub-module is present in the file but disabled at evaluation.

> Download link: see the project page,
> https://marcilzakour.github.io/ua-fit/ (release coming soon).

## For training from scratch

Stage A needs the ImageNet-pretrained HRNet-W40 backbone at
`checkpoints/hrnetv2_w40_imagenet_pretrained.pth` (HRNetV2-W40 ImageNet weights,
as referenced by `configs/uafit_stageA.yaml`). See [../TRAIN.md](../TRAIN.md).
