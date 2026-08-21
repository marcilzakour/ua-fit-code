"""Reproduce UA-Fit Table 1 (parametric multi-view hand mesh recovery).

Runs the released UA-Fit checkpoint (HRNet-W40 DOVF backbone + analytic,
uncertainty-weighted Levenberg-Marquardt solver) over the multi-view test/val
splits and reports, per dataset, the mean per-vertex error (MPVPE), the
mesh-regressed per-joint error (MPJPE) and the Procrustes-aligned MPVPE
(PA-MPVPE), all in millimetres. These are the numbers in Table 1 of the paper.

Protocol (identical to the paper):
  * fixed maximum view count per dataset (DexYCB 8, HO3D 5, ARCTIC 8,
    InterHand 8, OakInk 4), seed 1;
  * joints are regressed from the predicted / ground-truth *mesh* for both the
    prediction and the target (the raw joint annotations use a different joint
    convention), so MPJPE is the mesh-regressed error;
  * InterHand2.6M crop boxes are rebuilt from the visible ground-truth joints
    (``RECROP_FROM_JOINTS``) because a subset of the released bounding boxes is
    corrupt; this is applied identically to every method;
  * the full official split is evaluated by default (pass ``--limit N`` for a
    quick sanity subset; the paper's headline table is the full split).

Feed-forward shape ("betahead") mode: the compact shape head emits ``beta_hat``
and the solver performs a single LM fit. This is the "ours (fast)" column.

Example (full Table 1, all datasets):
    python scripts/eval.py --ckpt checkpoints/uafit_w40 --config configs/uafit_eval.yaml

Quick check on one dataset:
    python scripts/eval.py --ckpt checkpoints/uafit_w40 --datasets HO3D --limit 200
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import webdataset as wds
from pytorch3d.transforms import matrix_to_axis_angle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.datasets import create_dataset
from lib.utils import builder
from lib.utils.collation import collation_random_n_views
from lib.utils.config import get_config
from lib.utils.io_utils import load_model
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.transform import mano_to_openpose
import lib.models  # noqa: F401  (registers DOVFManoMVEpi with the model registry)

# dataset key -> (config test-section, fixed max views, full-split size)
DATASETS = {
    "DexYCB":    ("DEXYCB_TEST", 8, 4931),
    "HO3D":      ("HO3D_TEST", 5, 2687),
    "ARCTIC":    ("ARCTIC_TEST", 8, 17359),
    "InterHand": ("INTERHAND_TEST", 8, 85235),
    "OakInk":    ("OAKINK_TEST", 4, 21322),
}


def build_args():
    p = argparse.ArgumentParser(description="Reproduce UA-Fit Table 1.")
    p.add_argument("--ckpt", required=True,
                   help="checkpoint directory containing DOVFManoMVEpi.pth.tar")
    p.add_argument("--config", default="configs/uafit_eval.yaml",
                   help="model/data config (the released betabone eval config)")
    p.add_argument("--backbone", default="HRNet")
    p.add_argument("--iters", type=int, default=15, help="LM iterations")
    p.add_argument("--gnscale", type=float, default=3e5,
                   help="3D-anchor weight lambda_3D (tuned)")
    p.add_argument("--jac", default="analytic", choices=["analytic", "autograd"],
                   help="kinematic Jacobian mode")
    p.add_argument("--rad-mult", type=float, default=0.5,
                   help="multiplier on the learned Huber radius (3px -> 1.5px)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--views", type=int, default=0,
                   help="override the max view count for every dataset (0 = per-dataset default)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap scenes per dataset (0 = full official split)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_model(args, device):
    cfg = get_config(args.config, arg=None, merge=True)
    cfg.defrost()
    cfg.MODEL.INIT_FROM = ""  # the eval checkpoint is self-contained
    # OakInk test split mirrors the ARCTIC test section + the OakInk shards.
    if "OAKINK_TEST" not in cfg.DATASET.TEST:
        ok = cfg.DATASET.TEST.ARCTIC_TEST.clone()
        ok.defrost()
        ok.DATA_SPLIT = "test"
        ok.URLS = "data/dataset_tars/Oakink_mv/Oakink_mv_test-{000000..000045}.tar"
        cfg.DATASET.TEST.OAKINK_TEST = ok
    # Efficiency inference path: raw robust triangulation, no trained refiner /
    # mesh decoder (the components disabled at eval for the paper numbers).
    cfg.MODEL.EPI.POSE_REFINE = False
    cfg.MODEL.EPI.MESH_DECODER = False
    cfg.MODEL.EPI.ROBUST_TRIANG = True
    cfg.MODEL.BACKBONE.TYPE = args.backbone
    cfg.MODEL.BACKBONE.PRETRAINED = ""
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    load_model(model, args.ckpt, map_location="cpu", strict=False)
    model.eval()
    # solver configuration
    model.predictor_only = False
    model.fit_iters = args.iters
    model.j3d_gn_scale = args.gnscale
    model.jac_mode_eval = args.jac
    model.beta_head_on = True  # feed-forward shape head + a single LM solve
    if args.rad_mult != 1.0:
        with torch.no_grad():
            model.fit_log_radius.data = (
                model.fit_log_radius.data.exp() * args.rad_mult).clamp_min(1e-4).log()
    return cfg, model


def cuda_batch(b, device):
    for k, v in list(b.items()):
        if k != "cam_view_num":
            try:
                b[k] = torch.as_tensor(np.asarray(v)).float().to(device)
            except Exception:
                pass
    return b


def kabsch_global(mu, mano, Jreg, device):
    """Initialise the global wrist rotation/translation by aligning the canonical
    MANO joints to the triangulated 3D mean ``mu`` (closed-form Kabsch)."""
    B = mu.shape[0]
    with torch.no_grad():
        Jc0 = mano_to_openpose(Jreg, mano(torch.zeros(B, 48, device=device),
                                          torch.zeros(B, 10, device=device)).verts)[:, :21]
        mc = Jc0.mean(1, keepdim=True)
        mm = mu.mean(1, keepdim=True)
        H = (Jc0 - mc).transpose(1, 2) @ (mu - mm)
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)
        Z = torch.eye(3, device=device).repeat(B, 1, 1)
        Z[:, 2, 2] = torch.sign(torch.det(V @ U.transpose(1, 2)))
        go = matrix_to_axis_angle(V @ Z @ U.transpose(1, 2))
        jc_go = mano_to_openpose(Jreg, mano(torch.cat([go, torch.zeros(B, 45, device=device)], 1),
                                            torch.zeros(B, 10, device=device)).verts)[:, :21]
        tr = (mu - jc_go).mean(1)
    return go, tr


def pa_mpvpe(pred, gt):
    """Procrustes-aligned per-vertex error (mm)."""
    mp = pred.mean(1, keepdim=True)
    mg = gt.mean(1, keepdim=True)
    X = pred - mp
    Y = gt - mg
    K = X.transpose(1, 2) @ Y
    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, device=pred.device).repeat(pred.shape[0], 1, 1)
    Z[:, 2, 2] = torch.sign(torch.det(V @ U.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    var = (X ** 2).sum(dim=(1, 2)).clamp(min=1e-8)
    sc = (S * torch.diagonal(Z, dim1=-2, dim2=-1)).sum(-1) / var
    return (sc[:, None, None] * torch.einsum("bij,bvj->bvi", R, X) + mg - gt).norm(dim=-1).mean(-1) * 1000


def eval_dataset(model, cfg, mano, Jreg, name, section, nv, neval, args, device):
    ce = cfg.DATASET.TEST.clone()
    ce.defrost()
    ce.DATASET_LIST = [section]
    ce.EPOCH_SIZE = neval
    s = getattr(ce, section)
    s.VIEW_RANGE = [nv, nv]
    s.EPOCH_SIZE = neval
    if section == "INTERHAND_TEST":
        s.RECROP_FROM_JOINTS = True  # fix corrupt released bboxes (applied to all methods)
    try:
        s.MIX_RATIO = 1.0
    except Exception:
        pass
    de = create_dataset(ce, data_preset=cfg.DATA_PRESET, is_train=False)

    MPVPE, MPJPE, PA = [], [], []
    seen = set()
    scored = 0
    torch.cuda.synchronize()
    t0 = time.time()
    loader = wds.WebLoader(de, batch_size=args.batch_size, num_workers=args.workers,
                           worker_init_fn=worker_init_fn, collate_fn=collation_random_n_views)
    for b in loader:
        b = cuda_batch(b, device)
        if "master_verts_3d" not in b:
            continue
        with torch.no_grad():
            p = model._forward_impl(b, mode="val", epoch_idx=0)
        B = p["mu3d"].shape[0]
        mu = p["mu3d"].float()
        beta = p["beta_hat"].float().detach()  # feed-forward shape head
        go, tr = kabsch_global(mu, mano, Jreg, device)
        pose = torch.cat([go, torch.zeros(B, 45, device=device)], 1)
        K_all = b["target_cam_intr"].view(-1, 3, 3)
        w2c_all = torch.linalg.inv(b["target_cam_extr"].view(-1, 4, 4))
        with torch.no_grad():
            pose, tr = model._run_fitter_analytic_unc(
                pose, beta, tr, p["dovf_field"].float(), p["chol_field"].float(), None,
                K_all.float(), w2c_all.float(), b["cam_view_num"],
                device=device, mode="val", mc_mu3d=mu)[:2]
            out = mano(pose, beta)
            verts = out.verts + tr[:, None]
            joints = mano_to_openpose(Jreg, out.verts)[:, :21] + tr[:, None]

        gtv = b["master_verts_3d"].view(B, 778, 3)
        gtj_mesh = mano_to_openpose(Jreg, gtv)[:, :21]  # mesh-regressed GT joints (POEM protocol)
        ok = (gtv.norm(dim=-1).mean(-1) > 1e-4)
        # score each scene exactly once (batch-size invariant)
        if "sample_idx" in b:
            first = torch.as_tensor(np.concatenate([[0], np.cumsum(b["cam_view_num"])[:-1]]),
                                    device=ok.device, dtype=torch.long)
            sid = b["sample_idx"].view(-1)[first].long()
            fresh = torch.tensor([int(x.item()) not in seen for x in sid],
                                 device=ok.device, dtype=torch.bool)
            ok = ok & fresh
            for x in sid[ok].tolist():
                seen.add(int(x))
        if not ok.any():
            if scored >= neval:
                break
            continue
        scored += int(ok.sum())
        MPVPE.append(((verts[ok] - gtv[ok]).norm(dim=-1).mean(-1) * 1000).cpu())
        MPJPE.append(((joints[ok] - gtj_mesh[ok]).norm(dim=-1).mean(-1) * 1000).cpu())
        PA.append(pa_mpvpe(verts[ok], gtv[ok]).cpu())
        if scored >= neval:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    mpvpe = torch.cat(MPVPE).mean().item()
    mpjpe = torch.cat(MPJPE).mean().item()
    pa = torch.cat(PA).mean().item()
    n = torch.cat(MPVPE).numel()
    return dict(name=name, nv=nv, n=n, mpvpe=mpvpe, mpjpe=mpjpe, pa=pa,
                fps=n / max(dt, 1e-6), dt=dt)


def main():
    args = build_args()
    setup_seed(1)
    device = torch.device(args.device)
    cfg, model = build_model(args, device)
    mano, Jreg = model.mano_layer, model.J_regressor
    print(f"UA-Fit eval | jac={args.jac} iters={args.iters} lambda3D={args.gnscale:g} "
          f"rad_mult={args.rad_mult} | ckpt={args.ckpt}", flush=True)
    print(f"{'dataset':<12}{'views':>6}{'n':>8}{'MPVPE':>8}{'MPJPE':>8}{'PA-MPVPE':>10}{'fit/s':>8}", flush=True)
    setup_seed(1)
    rows = []
    for name in args.datasets:
        section, nv, full_n = DATASETS[name]
        if args.views:
            nv = min(args.views, nv)
        neval = args.limit if args.limit else full_n
        try:
            r = eval_dataset(model, cfg, mano, Jreg, name, section, nv, neval, args, device)
            rows.append(r)
            print(f"{r['name']:<12}{r['nv']:>6}{r['n']:>8}{r['mpvpe']:>8.2f}{r['mpjpe']:>8.2f}"
                  f"{r['pa']:>10.2f}{r['fps']:>8.1f}", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"{name:<12}  SKIP ({type(ex).__name__}: {ex})", flush=True)
    if len(rows) > 1:
        mean_mpvpe = sum(r["mpvpe"] for r in rows) / len(rows)
        mean_mpjpe = sum(r["mpjpe"] for r in rows) / len(rows)
        print(f"{'mean':<12}{'':>6}{'':>8}{mean_mpvpe:>8.2f}{mean_mpjpe:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
