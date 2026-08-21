# UA-Fit (code)

**Learn 2D, Solve 3D: UA-Fit, an Uncertainty-Weighted Analytical Solver for
Multi-view Hand Mesh Recovery.** ECCV 2026 &mdash; HANDS Workshop.

Marsil Zakour, Constantin Patsch, Martin Piccolrovazzi, Yuankai Wu, Eckehard
Steinbach &middot; Technical University of Munich (Chair of Media Technology, MIRMI).

**[Project page](https://marcilzakour.github.io/ua-fit/)** &middot;
[Solver core (parafit)](https://github.com/marcilzakour/parafit) &middot; Paper (coming soon)

UA-Fit learns only per-view 2D evidence (a dense oriented vote field with
per-vote anisotropic precision); a parameter-free analytical Levenberg&ndash;Marquardt
solver lifts it to a MANO-consistent 3D hand. The reconstruction stage is only
2.0M parameters, and the analytic kinematic Jacobian makes each solve about
14&times; cheaper than automatic differentiation.

This repository reproduces **Table 1** of the paper: multi-view parametric hand
mesh recovery on five benchmarks. The reusable, model-agnostic solver core is
released separately as
**[parafit](https://github.com/marcilzakour/parafit)**.

## Install

```bash
# 1. Install torch and pytorch3d for your platform first (CUDA specific):
#    https://pytorch.org/get-started/locally/
#    https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# 2. Then the rest:
pip install -r requirements.txt
pip install "git+https://github.com/lixiny/manotorch.git"
```

MANO model files are license-restricted and not shipped here. Download
`MANO_LEFT.pkl` / `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de/ and place
them under `assets/mano_v1_2/models/`.

## Data and checkpoint

- **Checkpoint**: download the released UA-Fit checkpoint into
  `checkpoints/uafit_w40/DOVFManoMVEpi.pth.tar` (see
  [checkpoints/README.md](checkpoints/README.md)).
- **Datasets**: bring your own multi-view WebDataset shards (DexYCB, HO3D,
  ARCTIC, InterHand2.6M, OakInk) under `data/dataset_tars/`. The shard URLs are
  set in `configs/uafit_eval.yaml`; edit them to point at your copies.

## Reproduce Table 1

```bash
python scripts/eval.py --ckpt checkpoints/uafit_w40 --config configs/uafit_eval.yaml
```

This evaluates the full official split of every dataset at its fixed maximum
view count and prints MPVPE / MPJPE / PA-MPVPE (mm). Expected (full-set):

| dataset (views) | MPVPE | MPJPE | PA-MPVPE |
|---|---|---|---|
| DexYCB (8v)    | 7.9 | 7.5 | 4.9 |
| HO3D (5v)      | 8.0 | 7.4 | 5.1 |
| ARCTIC (8v)    | 7.5 | 6.6 | 5.7 |
| InterHand (8v) | 8.8 | 7.8 | 6.2 |
| OakInk (4v)    | 9.6 | 8.9 | 5.7 |

MPJPE is measured against mesh-regressed ground-truth joints (both methods use
the same joint regressor, because the raw joint annotations follow a different
convention). InterHand crop boxes are rebuilt from the visible ground-truth
joints, applied identically to every method. See
[REPRODUCE.md](REPRODUCE.md) for the exact protocol, per-dataset commands, and a
quick sanity subset.

## Training

The training code and full from-scratch recipe are released after the paper is
published. This repository currently provides evaluation / inference and the
released checkpoint. The reusable, model-agnostic solver core is already available
as **[parafit](https://github.com/marcilzakour/parafit)**.

## Citation

```bibtex
@inproceedings{zakour2026uafit,
  title     = {Learn 2D, Solve 3D: UA-Fit, an Uncertainty-Weighted Analytical
               Solver for Multi-view Hand Mesh Recovery},
  author    = {Zakour, Marsil and Patsch, Constantin and Piccolrovazzi, Martin
               and Wu, Yuankai and Steinbach, Eckehard},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops --- HANDS},
  year      = {2026}
}
```
