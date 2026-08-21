# Reproducing UA-Fit Table 1

## Protocol

Every number is measured through one identical harness at seed 1:

- **Full official split** of each dataset, at its fixed maximum view count
  (DexYCB 8, HO3D 5, ARCTIC 8, InterHand2.6M 8, OakInk 4).
- **Feed-forward shape ("fast") mode**: the compact shape head emits the MANO
  betas and the analytical Levenberg-Marquardt solver performs a single fit
  (15 iterations, analytic kinematic Jacobian, tuned 3D-anchor weight
  `lambda_3D = 3e5`, learned Huber radius scaled by 0.5, i.e. 3px -> 1.5px).
- **Metrics** (mm): MPVPE (mean per-vertex error), MPJPE (mean per-joint error
  against **mesh-regressed** ground-truth joints), and PA-MPVPE (Procrustes-aligned
  MPVPE). Joints are regressed from the predicted / ground-truth mesh with the
  same joint regressor for both, because the raw joint annotations use a
  different joint convention.
- **InterHand crop fix**: a subset of the released InterHand2.6M crop boxes is
  corrupt (out-of-frame joints inflate them by an order of magnitude). Each
  view's crop is rebuilt from the visible ground-truth joints
  (`RECROP_FROM_JOINTS`), applied identically to every method, so the crop
  tightly contains the evaluated hand.

## Full table

```bash
python scripts/eval.py --ckpt checkpoints/uafit_w40 --config configs/uafit_eval.yaml
```

Expected (full-set; small run-to-run variation of ~0.05 mm from shard ordering):

| dataset (views) | n     | MPVPE | MPJPE | PA-MPVPE |
|---|---|---|---|---|
| DexYCB (8v)    | 4931  | 7.86 | 7.55 | 4.92 |
| HO3D (5v)      | 2687  | 8.01 | 7.41 | 5.10 |
| ARCTIC (8v)    | 17359 | 7.54 | 6.62 | 5.66 |
| InterHand (8v) | 85235 | 8.77 | 7.79 | 6.23 |
| OakInk (4v)    | 21322 | 9.58 | 8.86 | 5.67 |

## Per-dataset / quick runs

```bash
# one dataset, full split
python scripts/eval.py --ckpt checkpoints/uafit_w40 --datasets HO3D

# quick sanity subset (2000 scenes) -- ARCTIC/InterHand are subset-sensitive,
# so a small subset reads lower than the full-split value above
python scripts/eval.py --ckpt checkpoints/uafit_w40 --datasets HO3D --limit 2000

# view-count study: cap every dataset at N views
python scripts/eval.py --ckpt checkpoints/uafit_w40 --views 4
```

Useful flags: `--iters` (LM iterations), `--jac analytic|autograd` (Jacobian
mode), `--rad-mult` (Huber-radius multiplier), `--gnscale` (3D-anchor weight),
`--batch-size`, `--workers`, `--device`.

## Notes

- **Zero-shot rows.** The paper additionally reports zero-shot transfer on
  GigaHands (public) and an internal multi-view set. The internal set is not
  public and is therefore not reproducible here. GigaHands can be evaluated by
  pointing a new test section at its shards with the same protocol (6 views).
- **POEM-v2-param column.** Table 1 also compares against POEM-v2's released
  parametric checkpoint, evaluated through the same harness. That baseline is
  external to this repository (POEM-v2 is a separate project); this repo
  reproduces the UA-Fit column.
- **Subset vs full split.** The default is the full official split. ARCTIC and
  InterHand2.6M are subset-sensitive; a 2000-scene subset flatters ARCTIC
  (~6.7 vs full 7.5). Always report the full split for headline numbers.
