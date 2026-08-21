"""
DOVFManoMV — Dense Offset Vector Field guided multi-view MANO estimator.
========================================================================

Motivation (see ``report/`` and ``iterative_mano_fit_theseus_dovf/``)
---------------------------------------------------------------------
The stress-test in ``report/`` shows that an oracle *joint* DOVF fed to a
differentiable Levenberg-Marquardt MANO fitter reaches <4 mm MPJPE with as few
as 2–4 views, *provided the DOVF predictor error stays below ~3 px*.  A DOVF is
a dense, per-pixel field of 2D offsets pointing at each joint:

    dovf[y, x, j] = (jx - x, jy - y)

Because every pixel votes for the joint location, the representation is far more
redundant / occlusion-robust than a single heatmap peak, and — crucially — the
differentiable fitter can *sample the field at the current projected joint* each
LM iteration, turning the field into a reprojection-residual oracle that
converges from a poor initialisation.

Architecture
------------
1. Backbone (HRNet-W40 default; WiLoR ViT / DINOv2 pyramids selectable) runs per
   view and yields multi-resolution feature maps.
2. ``MultiResNeck`` (top-down FPN) fuses them into a feat_dim pyramid.
3. Two heads share the neck:
     • ``HeatmapHead``        — per-point 2D heatmaps (soft-argmax + softmax).
     • ``CoarseToFineDOVFHead`` — dense offset field via coarse-to-fine residual
       refinement across the pyramid (coarse global field + finer corrections).
   Each head's target (joints / vertices) is configurable; default: joints.
4. Dense-voting consensus turns (heatmap, DOVF) into a robust per-view 2D point.
5. ``ManoInitHead`` regresses a coarse MANO pose/shape; root translation is
   triangulated from the consensus joints (weighted DLT).
6. The DOVF field + init are handed to the Theseus ``DifferentiableManoOptimizer``
   (report pipeline, manotorch MANO) which fits world-space pose/translation by
   sampling the field at the projected joints (IMPLICIT backward, O(1) memory).
   Report-recommended defaults: iters=10, step=0.5, prior λ=1.0.

The whole stack is differentiable end-to-end: the 3D loss after the fitter trains
the DOVF predictor to drive the optimizer toward the GT — exactly the regime the
report identifies as the win at low view counts.
"""

import os
import sys
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.manolayer import ManoLayer
from termcolor import cprint

from lib.metrics.basic_metric import LossMetric
from lib.metrics.mean_epe import MeanEPE
from lib.metrics.pa_eval import PAEval
from lib.models.backbones import HRNet, build_backbone
from lib.models.model_abc import ModelABC
from lib.models.occ_vit_mv.mv_refiner import weighted_dlt
from lib.utils.builder import MODEL
from lib.utils.logger import logger
from lib.utils.recorder import Recorder
from lib.utils.transform import mano_to_openpose

from lib.models.dovf import (
    MultiResNeck,
    HeatmapHead,
    CoarseToFineDOVFHead,
    ManoInitHead,
    dovf_consensus_2d,
    build_dovf_target,
)
from lib.models.dovf.analytic_fitter import analytic_fit


@MODEL.register_module()
class DOVFManoMV(ModelABC):
    """Dense-Offset-Vector-Field guided multi-view MANO model."""

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.data_preset_cfg = getattr(cfg, "DATA_PRESET", cfg)
        self.num_joints = getattr(self.data_preset_cfg, "NUM_JOINTS", 21)
        self.num_verts = getattr(self.data_preset_cfg, "NUM_VERTS", 778)
        self.center_idx = getattr(self.data_preset_cfg, "CENTER_IDX", 0)
        self.image_size = tuple(getattr(self.data_preset_cfg, "IMAGE_SIZE", [256, 256]))
        self.heatmap_size = tuple(cfg.get("HEATMAP_SIZE", [64, 64]))
        self.feat_dim = cfg.get("FEAT_DIM", 128)

        # Heatmap supervision variant: mse (default, current behaviour) | kl | bce_unnorm.
        self.heatmap_loss = str(cfg.get("HEATMAP_LOSS", "mse")).lower()

        # ── Head targets (configurable; default joints for both) ─────────────
        self.heatmap_target = cfg.get("HEATMAP_TARGET", "joints")
        self.dovf_target = cfg.get("DOVF_TARGET", "joints")
        assert self.heatmap_target in ("joints", "verts")
        assert self.dovf_target in ("joints", "verts")
        self.n_hm_pts = self.num_joints if self.heatmap_target == "joints" else self.num_verts
        self.n_dovf_pts = self.num_joints if self.dovf_target == "joints" else self.num_verts

        # ── 3D back-end ───────────────────────────────────────────────────────
        # "theseus": DOVF -> DifferentiableManoOptimizer (report pipeline).
        # "dlt":     consensus 2D -> weighted DLT (no MANO fit, no theseus dep).
        self.backend = cfg.get("BACKEND", "theseus")
        assert self.backend in ("theseus", "dlt")
        # Fitter implementation for the theseus backend's reprojection optimization:
        #   "analytic" — batched torch GN with analytic projection+DOVF Jacobian
        #                (verified identical to theseus; batches ~linearly → ~100x).
        #   "theseus"  — the original per-scene Theseus DENSE-autograd solver.
        self.fitter = cfg.get("FITTER", "analytic")
        assert self.fitter in ("analytic", "theseus")
        if self.backend == "theseus":
            assert self.dovf_target == "joints", \
                "theseus backend requires DOVF_TARGET=joints (the optimizer is joint-based)"
        else:  # dlt — direct vertex triangulation needs a dense vertex field + matching heatmap
            assert self.dovf_target == "verts" and self.heatmap_target == "verts", \
                "dlt backend requires DOVF_TARGET=verts and HEATMAP_TARGET=verts"

        # ── Backbone ──────────────────────────────────────────────────────────
        if hasattr(cfg, "BACKBONE"):
            self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        else:
            cfg_bb = cfg.clone(); cfg_bb.defrost()
            if getattr(cfg_bb, "PRETRAINED", None) is None:
                cfg_bb.PRETRAINED = ""
            self.img_backbone = HRNet(cfg_bb)

        # Probe backbone to discover the multi-scale channel list.
        in_ch_list = self._probe_backbone_channels()
        cprint(f"[{self.name}] backbone scales (ch, high->low): {in_ch_list}", "cyan")

        # ── Neck + heads ────────────────────────────────────────────────────
        # NECK_UPSAMPLE adds a learned 2× level on top of the pyramid so the heads
        # produce a GENUINE higher-res field (e.g. 128² vs the backbone's 64²),
        # the σ_px lever (V0 probe: 2v error is 2D-localization-bound, r≈0.9–1.0).
        self.neck_upsample = bool(cfg.get("NECK_UPSAMPLE", False))
        self.neck = MultiResNeck(in_ch_list, feat_dim=self.feat_dim,
                                 extra_fine=self.neck_upsample)
        if self.neck_upsample:
            self._n_scales += 1
        self.heatmap_head = HeatmapHead(self.feat_dim, self.n_hm_pts, self.heatmap_size)
        self.dovf_head = CoarseToFineDOVFHead(self.feat_dim, self.n_dovf_pts,
                                              self.heatmap_size, n_levels=self._n_scales,
                                              context=cfg.get("DOVF_HEAD_CONTEXT", "none"))
        self.init_head = ManoInitHead(self.feat_dim, n_pose=48, n_betas=10)

        # ── MANO ───────────────────────────────────────────────────────────
        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang", use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=self.center_idx, flat_hand_mean=True,
        )
        self.register_buffer("J_regressor", self.mano_layer.th_J_regressor.clone())
        self.nominal_root_depth = float(cfg.get("NOMINAL_ROOT_DEPTH", 0.5))

        # ── Theseus DOVF optimizers (one per supported view count, tied σ) ──
        self.opt_cfg = cfg.get("OPTIMIZER_DOVF", {})
        self.min_views = int(self.opt_cfg.get("MIN_VIEWS", 2))
        self.max_views = int(self.opt_cfg.get("MAX_VIEWS", 4))
        self.fit_iters = int(self.opt_cfg.get("ITERS", 10))
        self.fit_step = float(self.opt_cfg.get("STEP_SIZE", 0.5))
        self.fit_prior = float(self.opt_cfg.get("PRIOR_WEIGHT", 1.0))
        # MANO Jacobian for the analytic fitter: "analytic" (closed-form kinematic
        # Jacobian, ~14x faster) or "autograd" (exact via batched vjp). Both batch.
        # MANO Jacobian: fast analytic kinematic Jac in TRAIN (bias absorbed by the
        # DOVF head), exact autograd Jac in EVAL (bit-matches the Theseus pipeline
        # that the official manolayer-feedforward evaluation expects).
        self.jac_mode_train = self.opt_cfg.get("JAC_MODE", "analytic")
        self.jac_mode_eval = self.opt_cfg.get("JAC_MODE_EVAL", "autograd")
        # Outer backward through the fitter: "implicit" (O(1) memory, fixed-point;
        # verified to match Theseus IMPLICIT to cos 0.9997) or "unroll".
        self.fit_backward = self.opt_cfg.get("BACKWARD", "implicit")
        if self.backend == "theseus":
            if self.fitter == "theseus":
                self._build_optimizers()
            else:  # analytic fitter: a single learnable Huber log-radius
                self.fit_log_radius = nn.Parameter(
                    torch.tensor([np.log(float(self.opt_cfg.get("LOSS_RADIUS_INIT", 3.0)))],
                                 dtype=torch.float32))

        # ── Pose prior for the fitter ────────────────────────────────────────
        # "init" (default): isotropic L2 toward the network init θ0.
        # "pca": Mahalanobis prior on the 45 finger DoF in MANO's PCA basis — pulls
        #        the pose toward the MEAN hand (0 PCA coeffs), penalising large /
        #        low-variance (implausible) deviations. MANO stays in axis-angle;
        #        we only use the PCA basis to define the prior. Global orient + trans
        #        keep a small L2 toward init (ANCHOR_WEIGHT) for placement.
        # per-(view,joint) confidence weighting of the reprojection residual
        # (heatmap peak prob): trust reliable joints, down-weight occluded ones.
        self.conf_weight = bool(self.opt_cfg.get("CONF_WEIGHT", False))
        self.prior_type = self.opt_cfg.get("PRIOR_TYPE", "init")
        if self.fitter == "analytic" and self.prior_type == "pca":
            _pca = ManoLayer(joint_rot_mode="pca", use_pca=True, ncomps=45,
                             mano_assets_root="assets/mano_v1_2",
                             center_idx=self.center_idx, flat_hand_mean=True)
            Bm = _pca.th_selected_comps.detach().float()          # (45,45) variance-scaled basis
            mu = _pca.th_hands_mean.detach().float().reshape(45)  # (45,) mean finger pose
            Sinv = torch.inverse(Bm.t() @ Bm)                     # (45,45) Mahalanobis precision
            Sinv = Sinv / Sinv.diag().mean()                      # unit mean-diag -> λ interpretable
                                                                  # (keeps Mahalanobis shape, isotropic scale)
            P = 48 + 3
            prec = torch.zeros(P, P); prec[3:48, 3:48] = self.fit_prior * Sinv
            ref = torch.zeros(P); ref[3:48] = mu
            aw = float(self.opt_cfg.get("ANCHOR_WEIGHT", 0.05))
            anchor = torch.zeros(P); anchor[0:3] = aw; anchor[48:51] = aw
            self.register_buffer("prior_prec", prec)
            self.register_buffer("prior_ref", ref)
            self.register_buffer("prior_anchor", anchor)
            cprint(f"[{self.name}] PCA-Mahalanobis pose prior (λ={self.fit_prior}, anchor={aw})", "cyan")

        # ── Losses ───────────────────────────────────────────────────────────
        loss_cfg = getattr(cfg, "LOSS", None) or {}
        self.w_heatmap = float(loss_cfg.get("HEATMAP_WEIGHT", 10.0))
        self.w_dovf = float(loss_cfg.get("DOVF_WEIGHT", 1.0))
        self.w_dovf_deep = float(loss_cfg.get("DOVF_DEEP_WEIGHT", 0.25))
        self.w_consensus = float(loss_cfg.get("CONSENSUS_2D_WEIGHT", 1.0))
        self.w_joints_3d = float(loss_cfg.get("JOINTS_3D_WEIGHT", 10.0))
        self.w_verts_3d = float(loss_cfg.get("VERTS_3D_WEIGHT", 10.0))
        self.w_pose_reg = float(loss_cfg.get("POSE_REG_WEIGHT", 0.001))
        self.w_betas_reg = float(loss_cfg.get("BETAS_REG_WEIGHT", 0.0005))
        # Option-1 init prior: supervise the MANO-init head's root-relative joints
        # against GT (camera-model-free; gives the fitter a real orientation basin,
        # esp. at 2 views). Active in BOTH the 2D-pretrain and 3D stages when > 0.
        # We supervise MANO(init) JOINTS (always present) not pose params (missing
        # GT pose on Oakink etc.). 0 -> off (init head stays frozen/unused in 2D).
        self.w_init_joints = float(loss_cfg.get("INIT_JOINTS_3D_WEIGHT", 0.0))
        # Near-joint DOVF weighting: weight the dense DOVF L1 by a Gaussian (this sigma,
        # in heatmap px) around each GT joint so accuracy concentrates where the votes
        # live (consensus = heatmap-weighted votes), not on far/easy pixels. 0 = uniform.
        self.dovf_loss_sigma = float(loss_cfg.get("DOVF_LOSS_SIGMA", 0.0))
        self.deep_supervision = bool(cfg.get("DEEP_SUPERVISION", True))

        # Warmup: train only the 2D heads before enabling the fitter's 3D loss.
        self.warmup_epochs = int(cfg.get("WARMUP_EPOCHS", 2))
        self.ramp_epochs = int(cfg.get("RAMP_EPOCHS", 3))

        # ── Stage-A 2D-only pretrain ────────────────────────────────────────
        # When PRETRAIN_2D is set the model trains ONLY the per-view 2D front-end
        # (backbone + neck + heatmap/DOVF heads + consensus) with the 2D losses,
        # skipping the fitter / init head / 3D losses entirely. Combined with
        # single-view sampling (VIEW_RANGE [1,1], large batch) this is the fast,
        # high-throughput way to drive the consensus-2D error under ~5px before
        # attaching the 3D solver (Stage B). Module names are unchanged so the
        # checkpoint loads straight into the full multi-view model.
        self.pretrain_2d = bool(cfg.get("PRETRAIN_2D", False))
        self.dark_decode = bool(cfg.get("DARK_DECODE", False))  # heatmap soft-argmax refine
        # channels_last (NHWC): ~2.1x faster bf16 convs on the HRNet backbone
        # (microbench). Layout-only -> numerically identical. Converted lazily on
        # first forward so it survives .cuda()/DDP wrap.
        self.channels_last = bool(cfg.get("CHANNELS_LAST", False))
        self._cl_done = False
        if self.pretrain_2d:
            # Freeze every module the 2D path never touches so ALL remaining
            # trainable params receive a gradient -> DDP runs with
            # FIND_UNUSED_PARAMETERS=false (no per-step autograd-graph walk).
            # Exception: with the Option-1 init prior on, the init head IS trained
            # in 2D-pretrain (it gets gradient from the init-joint loss), so leave
            # it unfrozen. MANO layer is never trained (template) -> always frozen.
            if self.w_init_joints <= 0:
                for p in self.init_head.parameters():
                    p.requires_grad_(False)
            for p in self.mano_layer.parameters():
                p.requires_grad_(False)
            if hasattr(self, "fit_log_radius"):
                self.fit_log_radius.requires_grad_(False)
            # HRNet's ImageNet classification head (incre/downsamp/final/classifier)
            # is unused for feature extraction — freeze it (this is what forced
            # find_unused=true in every prior run; dead weight in 2D mode).
            for attr in ("incre_modules", "downsamp_modules", "final_layer", "classifier"):
                sub = getattr(self.img_backbone, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad_(False)

        self.criterion_hm = nn.MSELoss()
        self.criterion_2d = nn.SmoothL1Loss()
        self.criterion_3d = nn.SmoothL1Loss()

        # ── Metrics ───────────────────────────────────────────────────────────
        self.MPJPE_3D = MeanEPE(cfg, "joints_3d")
        self.MPVPE_3D = MeanEPE(cfg, "vertices_3d")
        self.MPJPE_3D_REL = MeanEPE(cfg, "joints_3d_rel")
        self.MPVPE_3D_REL = MeanEPE(cfg, "vertices_3d_rel")
        self.PA = PAEval(cfg, mesh_score=True)
        self.loss_metric = LossMetric(cfg)
        self.MPJPE_3D_TRAIN = MeanEPE(cfg, "joints_3d")
        # 2D-pretrain metrics: image-pixel reprojection error of the predictor,
        # plus cheap multi-view DLT triangulation 3D MPJPE (NO Theseus solver) from
        # the heatmap-argmax 2D and from the DOVF-consensus 2D — the real 3D signal,
        # directly comparable to POEM's 2-view MPJPE.
        self.CONS2D_PX = MeanEPE(cfg, "consensus_2d_px")
        self.HM2D_PX = MeanEPE(cfg, "heatmap_2d_px")
        self.DLT_CONS_MPJPE = MeanEPE(cfg, "dlt_cons_mpjpe")
        self.DLT_HM_MPJPE = MeanEPE(cfg, "dlt_hm_mpjpe")
        # Per-dataset val buckets (one model / one checkpoint, evaluated on all). The
        # mixed val loader interleaves datasets; we bucket by the per-sample
        # `dataset_name` tag so each eval set's 2D px / DLT-3D is visible separately.
        # Datasets in `_agg3d_exclude` are kept out of the AGGREGATE DLT-3D proxy only
        # (their raw-DLT 3D is a coord-frame artefact, e.g. Interhand ~38mm) — they
        # still contribute to aggregate 2D px and to their own per-dataset buckets.
        self._metric_cfg = cfg
        self._per_ds_metrics = {}
        self._agg3d_exclude = set(cfg.get("EVAL_AGG3D_EXCLUDE", ["Interhand"]))

        # ── Visualisation (TensorBoard/WandB DOVF panels) ──────────────────────
        self.viz_enabled = bool(cfg.get("VIZ_ENABLED", True))
        self.viz_interval = int(cfg.get("VIZ_INTERVAL", 500))   # train steps between panels
        self._cur_epoch = 0
        self._last_val_viz_epoch = -1
        self._wheel_logged = False

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        train_cfg = getattr(cfg, "TRAIN", None)
        self.train_log_interval = getattr(train_cfg, "LOG_INTERVAL", 50) if train_cfg else 50

        # Stage-B init: load a Stage-A (2D-pretrained) checkpoint's weights (fresh
        # optimizer/epoch, unlike --resume). INIT_FROM is the checkpoint dir holding
        # DOVFManoMV.pth.tar. strict=False so frozen-then-unfrozen modules still map.
        init_from = cfg.get("INIT_FROM", "")
        if init_from:
            from lib.utils.io_utils import load_model
            load_model(self, init_from, strict=False, map_location="cpu")
            logger.info(f"{self.name}: initialized weights from {init_from}")

        logger.info(
            f"{self.name} ready | backend={self.backend} | "
            f"heatmap={self.heatmap_target}({self.n_hm_pts}) dovf={self.dovf_target}({self.n_dovf_pts}) | "
            f"views[{self.min_views}-{self.max_views}] | hm={self.heatmap_size} | warmup={self.warmup_epochs}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _probe_backbone_channels(self):
        """Return backbone output channels per scale, high-res first."""
        was_train = self.img_backbone.training
        self.img_backbone.eval()
        with torch.no_grad():
            probe = self.img_backbone(torch.zeros(1, 3, *self.image_size))
        self.img_backbone.train(was_train)
        if isinstance(probe, dict):
            probe = [v for v in probe.values() if isinstance(v, torch.Tensor) and v.dim() == 4]
        if isinstance(probe, torch.Tensor):
            probe = [probe]
        # Order high-res -> low-res by spatial size.
        probe = sorted(probe, key=lambda t: t.shape[-1], reverse=True)
        self._n_scales = len(probe)
        return [t.shape[1] for t in probe]

    def _build_optimizers(self):
        from lib.models.dovf.mano_optimizer import DifferentiableManoOptimizer
        hm_h, hm_w = self.heatmap_size
        iters = int(self.opt_cfg.get("ITERS", 10))
        step = float(self.opt_cfg.get("STEP_SIZE", 0.5))
        prior = float(self.opt_cfg.get("PRIOR_WEIGHT", 1.0))
        radius = float(self.opt_cfg.get("LOSS_RADIUS_INIT", 3.0))
        opts = {}
        for n in range(self.min_views, self.max_views + 1):
            opts[str(n)] = DifferentiableManoOptimizer(
                mano_layer=self.mano_layer, J_regressor=self.J_regressor,
                num_views=n, H=hm_h, W=hm_w, num_joints=self.num_joints,
                max_iterations=iters, step_size=step, pose_prior_weight=prior,
                loss_radius_init=radius, max_projection_views=n,
            )
        self.optimizers = nn.ModuleDict(opts)
        # Tie the learnable Huber radius across all view-count optimizers.
        shared = self.optimizers[str(self.min_views)].log_radius_param
        for n in range(self.min_views + 1, self.max_views + 1):
            self.optimizers[str(n)].log_radius_param = shared

    # ──────────────────────────────────────────────────────────────────────────
    # Small utilities
    # ──────────────────────────────────────────────────────────────────────────

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _log_metric_scalars(self, metrics, comment, epoch_idx):
        """Push per-epoch metric values to TB/WandB (Recorder only writes text)."""
        summ = getattr(self, "summary", None)
        if summ is None or getattr(summ, "rank", 0) != 0:
            return
        for M in metrics:
            try:
                for k, val in M.get_measures().items():
                    summ.add_scalar(f"{comment}/{k}", float(val), epoch_idx)
            except Exception:
                pass

    def _maybe_viz(self, batch, preds, mode, step_idx):
        """Render + log a DOVF panel (rank-0 only; train on cadence, val once/epoch)."""
        if not self.viz_enabled:
            return
        summ = getattr(self, "summary", None)
        if summ is None or getattr(summ, "rank", 0) != 0 or not hasattr(summ, "add_figure"):
            return
        if mode == "train":
            if self.viz_interval <= 0 or (step_idx % self.viz_interval) != 0:
                return
        else:  # val: only the first batch of each val pass
            if self._last_val_viz_epoch == self._cur_epoch:
                return
            self._last_val_viz_epoch = self._cur_epoch
        try:
            from lib.utils.dovf_viz import (build_dovf_panel, build_dovf_multires_panel,
                                            build_dovf_perjoint_panel, build_flow_wheel_key,
                                            build_multiview_panel, build_solver3d_panel)
            # colour-wheel key for the flow tiles — log once per run.
            if not self._wheel_logged:
                summ.add_figure("viz/dovf_flow_key", build_flow_wheel_key(), global_step=step_idx)
                self._wheel_logged = True
            fig = build_dovf_panel(self, batch, preds, mode=mode)
            if fig is not None:
                summ.add_figure(f"viz/{mode}", fig, global_step=step_idx)
            mr = build_dovf_multires_panel(self, batch, preds, mode=mode)
            if mr is not None:
                summ.add_figure(f"viz/{mode}_dovf_multires", mr, global_step=step_idx)
            fl = build_dovf_perjoint_panel(self, batch, preds, mode=mode)
            if fl is not None:
                summ.add_figure(f"viz/{mode}_dovf_flow", fl, global_step=step_idx)
            mv = build_multiview_panel(self, batch, preds, mode=mode)
            if mv is not None:
                summ.add_figure(f"viz/{mode}_multiview", mv, global_step=step_idx)
            s3 = build_solver3d_panel(self, batch, preds)      # 3D stage only
            if s3 is not None:
                summ.add_figure(f"viz/{mode}_solver3d", s3, global_step=step_idx)
        except Exception as e:
            logger.warning(f"[viz] panel render failed: {e}")

    def _renorm(self, img):
        return (img + 0.5 - self.imagenet_mean) / self.imagenet_std

    def _hm_scale(self, device, dtype):
        hm_h, hm_w = self.heatmap_size
        img_h, img_w = self.image_size
        return torch.tensor([hm_w / img_w, hm_h / img_h], device=device, dtype=dtype)

    def _backbone_forward(self, images):
        if self.channels_last:
            if not self._cl_done:  # convert conv weights once (after .cuda()/DDP)
                self.img_backbone.to(memory_format=torch.channels_last)
                self.neck.to(memory_format=torch.channels_last)
                self.heatmap_head.to(memory_format=torch.channels_last)
                self.dovf_head.to(memory_format=torch.channels_last)
                self._cl_done = True
            images = images.contiguous(memory_format=torch.channels_last)
        feats = self.img_backbone(self._renorm(images))
        if isinstance(feats, dict):
            feats = [v for v in feats.values() if isinstance(v, torch.Tensor) and v.dim() == 4]
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        return sorted(feats, key=lambda t: t.shape[-1], reverse=True)  # high->low

    def _project_to_views(self, points_world, batch, n_pts):
        """master-frame points (B, P, 3) -> per-view image-space 2D (BN, P, 2)."""
        extr = batch["target_cam_extr"].view(-1, 4, 4)           # T_c2m (cam->world)
        w2c = torch.linalg.inv(extr)                             # world->cam
        K = batch["target_cam_intr"].view(-1, 3, 3)
        cvn = batch["cam_view_num"]
        out = []
        for i in range(len(cvn)):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1]))
            R = w2c[s:e, :3, :3]; t = w2c[s:e, :3, 3]            # (N,3,3),(N,3)
            pw = points_world[i:i + 1, :n_pts].expand(e - s, -1, -1)  # (N,P,3)
            pc = torch.einsum("nij,npj->npi", R, pw) + t.unsqueeze(1)  # (N,P,3)
            z = pc[..., 2].clamp(min=1e-3)
            u = K[s:e, 0, 0:1] * pc[..., 0] / z + K[s:e, 0, 2:3]
            v = K[s:e, 1, 1:2] * pc[..., 1] / z + K[s:e, 1, 2:3]
            out.append(torch.stack([u, v], dim=-1))               # (N,P,2)
        return torch.cat(out, dim=0)                              # (BN,P,2)

    def _build_heatmap_gt(self, gt_2d_hm, normalize=True):
        """(BN, P, 2) in heatmap px -> (BN, P, h, w) Gaussian heatmaps.
        normalize=True: sum-to-1 per joint (for softmax-MSE / KL targets).
        normalize=False: peak-1 unnormalised Gaussian (for sigmoid-BCE regression)."""
        bn, p, _ = gt_2d_hm.shape
        h, w = self.heatmap_size
        xs = torch.linspace(0, w - 1, w, device=gt_2d_hm.device, dtype=gt_2d_hm.dtype)
        ys = torch.linspace(0, h - 1, h, device=gt_2d_hm.device, dtype=gt_2d_hm.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.view(1, 1, h, w); gy = gy.view(1, 1, h, w)
        sigma = float(getattr(self.data_preset_cfg, "HEATMAP_SIGMA", 2.0))
        cx = gt_2d_hm[..., 0].view(bn, p, 1, 1)
        cy = gt_2d_hm[..., 1].view(bn, p, 1, 1)
        hm = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma * sigma))
        if normalize:
            hm = hm / hm.flatten(2).sum(-1).clamp(min=1e-6)[..., None, None]
        return hm

    def _heatmap_loss(self, preds, gt_2d_hm):
        """Heatmap supervision; `HEATMAP_LOSS` selects the variant.
          mse        : MSE(softmax probs, sum-normalised Gaussian)  — distribution match,
                       numerically tiny (effective weight ~0; heatmap learns via consensus).
          kl         : KL(GT || softmax pred) on sum-normalised targets — soft cross-entropy,
                       monotonic from uniform, no softmax-MSE sharpening bump. O(1) scale.
          bce_unnorm : BCE-with-logits(logits, peak-1 Gaussian) — standard heatmap regression,
                       target peak=1 so confident peaks are rewarded, not penalised. O(0.1) scale.
          none/attn  : NO direct heatmap supervision. The softmax head over the spatial
                       grid becomes a pure LEARNABLE ATTENTION gate for the DOVF votes,
                       trained end-to-end only through the consensus-2D loss. Removes the
                       softmax-MSE "sharpening bump" (hm loss rising while the rest fall).
        NOTE: scales differ a lot -> retune HEATMAP_WEIGHT per mode (mse~10, kl~0.5-1, bce~1-5)."""
        mode = self.heatmap_loss
        if mode in ("none", "attn"):
            # gate is learned purely via consensus; return a graph-free zero.
            return preds["hm_probs"].sum() * 0.0
        if mode == "mse":
            return self.criterion_hm(preds["hm_probs"], self._build_heatmap_gt(gt_2d_hm))
        if mode == "kl":
            gt = self._build_heatmap_gt(gt_2d_hm).flatten(2)                 # (BN,P,HW) sums to 1
            logp = F.log_softmax(preds["hm_logits"].flatten(2), dim=-1)
            return (gt * (gt.clamp_min(1e-8).log() - logp)).sum(-1).mean()
        if mode == "bce_unnorm":
            gt = self._build_heatmap_gt(gt_2d_hm, normalize=False)           # peak-1 Gaussian
            return F.binary_cross_entropy_with_logits(preds["hm_logits"], gt)
        raise ValueError(f"unknown HEATMAP_LOSS '{mode}' (mse|kl|bce_unnorm)")

    def _dovf_l1(self, pred_field, gt_field, gt_2d_hm):
        """Final-scale DOVF L1. If DOVF_LOSS_SIGMA>0, Gaussian-weight by proximity to the
        GT joint (consensus error = heatmap-weighted DOVF error -> make the field accurate
        where the votes live). pred/gt_field: (BN,P,2,h,w); gt_2d_hm: (BN,P,2) heatmap px."""
        if self.dovf_loss_sigma <= 0:
            return F.l1_loss(pred_field, gt_field)
        bn, p, _ = gt_2d_hm.shape
        h, w = self.heatmap_size
        xs = torch.linspace(0, w - 1, w, device=pred_field.device, dtype=pred_field.dtype)
        ys = torch.linspace(0, h - 1, h, device=pred_field.device, dtype=pred_field.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        cx = gt_2d_hm[..., 0].view(bn, p, 1, 1); cy = gt_2d_hm[..., 1].view(bn, p, 1, 1)
        s = self.dovf_loss_sigma
        wgt = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s)).unsqueeze(2)  # (bn,p,1,h,w)
        return (wgt * (pred_field - gt_field).abs()).sum() / (wgt.sum() * 2 + 1e-6)

    def _decode_mano(self, pose, betas, trans):
        """pose (B,48) axisang, betas (B,10), trans (B,3) world wrist -> joints,verts (world)."""
        out = self.mano_layer(pose, betas)
        verts = out.verts                                          # (B,778,3) centered
        joints = mano_to_openpose(self.J_regressor, verts)[:, : self.num_joints]
        verts_w = verts + trans.unsqueeze(1)
        joints_w = joints + trans.unsqueeze(1)
        return joints_w, verts_w

    def _mano_init_joints_rel(self, init_pose, init_betas):
        """Root-relative joints (B,21,3) from the MANO-init head output (Option-1).
        Translation-free: only the pose/shape -> joint geometry is supervised."""
        zero_t = init_pose.new_zeros(init_pose.shape[0], 3)
        j, _ = self._decode_mano(init_pose, init_betas, zero_t)
        return j - j[:, self.center_idx:self.center_idx + 1]

    def _optimizer_active(self, mode, epoch_idx):
        if self.backend != "theseus":
            return False
        if mode != "train":
            return True
        return epoch_idx is not None and epoch_idx >= self.warmup_epochs

    def _ramp(self, epoch_idx):
        if epoch_idx is None or epoch_idx >= self.warmup_epochs + self.ramp_epochs:
            return 1.0
        if epoch_idx < self.warmup_epochs:
            return 0.0
        return (epoch_idx - self.warmup_epochs) / max(self.ramp_epochs, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Core forward
    # ──────────────────────────────────────────────────────────────────────────

    def _forward_impl(self, batch, mode="val", epoch_idx=0):
        images = batch["image"]
        if images.dim() == 5:
            images = images.view(-1, *images.shape[-3:])
        K_all = batch["target_cam_intr"].view(-1, 3, 3)
        extr_all = batch["target_cam_extr"].view(-1, 4, 4)         # T_c2m
        w2c_all = torch.linalg.inv(extr_all)                       # world->cam
        cvn = batch["cam_view_num"]
        device = images.device

        # ── Per-view front-end ─────────────────────────────────────────────
        feats = self._backbone_forward(images)                     # list high->low
        pyramid = self.neck(feats)                                 # list high->low
        hm_logits, hm_probs, hm_coords = self.heatmap_head(pyramid[0])   # (BN,P,*)
        dovf_field, dovf_per_scale = self.dovf_head(pyramid)       # (BN,Pd,2,h,w)
        consensus_2d, consensus_conf = dovf_consensus_2d(hm_probs, dovf_field) \
            if self.heatmap_target == self.dovf_target else (None, None)
        # Pooled global descriptor per view (coarsest level GAP).
        desc_view = pyramid[-1].mean(dim=(-2, -1))                 # (BN, feat_dim)

        hm_h, hm_w = self.heatmap_size
        scale_hm2img = (torch.tensor([self.image_size[1] / hm_w,
                                      self.image_size[0] / hm_h], device=device))

        B = len(cvn)
        opt_active = self._optimizer_active(mode, epoch_idx)

        # ── Per-scene init (cheap): pooled MANO init + DLT translation ────────
        init_pose_list, init_betas_list, trans0_list = [], [], []
        for i in range(B):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
            desc = desc_view[s:e].mean(0, keepdim=True)            # (1, feat_dim)
            pose0, betas0 = self.init_head(desc)                   # (1,48),(1,10)
            init_pose_list.append(pose0); init_betas_list.append(betas0)
            if self.backend == "theseus":
                trans0_list.append(self._init_translation(
                    consensus_2d, consensus_conf, hm_coords, dovf_field,
                    K_all, w2c_all, extr_all, s, e, N, scale_hm2img, device))
        init_pose = torch.cat(init_pose_list, 0)                   # (B,48)
        init_betas = torch.cat(init_betas_list, 0)                 # (B,10)

        if self.backend == "dlt":
            pred_joints_list, pred_verts_list = [], []
            for i in range(B):
                s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
                jw, vw = self._dlt_3d(consensus_2d, consensus_conf, K_all, w2c_all,
                                      s, e, N, scale_hm2img, device)
                pred_joints_list.append(jw); pred_verts_list.append(vw)
            pred_joints = torch.cat(pred_joints_list, 0)
            pred_verts = torch.cat(pred_verts_list, 0)
        else:
            trans0 = torch.cat(trans0_list, 0)                     # (B,3)
            # `opt_active` gates whether the 3D loss BACKPROPS through the solver
            # (False during warmup). The fitter itself ALWAYS runs so the reported
            # MPJPE reflects the (improving) DOVF field, not the static init decode.
            # During warmup we run it under no_grad (metric only; 3D loss ramp=0).
            if self.fitter == "analytic":
                grad_ctx = nullcontext() if opt_active else torch.no_grad()
                with grad_ctx:
                    pose_opt, trans_opt = self._run_fitter_analytic(
                        init_pose, init_betas, trans0, dovf_field, K_all, w2c_all, cvn, device,
                        mode, conf=consensus_conf)
            else:  # theseus: too expensive to run every warmup step -> gate fully
                if opt_active:
                    po, to = [], []
                    for i in range(B):
                        s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
                        if self.min_views <= N:
                            p, t = self._run_optimizer(init_pose[i:i+1], init_betas[i:i+1],
                                                       trans0[i:i+1], dovf_field, K_all, w2c_all, s, e, N, device)
                        else:
                            p, t = init_pose[i:i+1], trans0[i:i+1]
                        po.append(p); to.append(t)
                    pose_opt, trans_opt = torch.cat(po, 0), torch.cat(to, 0)
                else:
                    pose_opt, trans_opt = init_pose, trans0
            pred_joints, pred_verts = self._decode_mano(pose_opt, init_betas, trans_opt)

        preds = {
            "pred_joints_3d": pred_joints,                          # (B,21,3)
            "pred_verts_3d": pred_verts,                            # (B,778,3)
            "hm_logits": hm_logits, "hm_probs": hm_probs, "hm_coords": hm_coords,
            "dovf_field": dovf_field, "dovf_per_scale": dovf_per_scale,
            "consensus_2d": consensus_2d,
            "init_pose": init_pose,
            "init_betas": init_betas,
            "init_joints_rel": (self._mano_init_joints_rel(init_pose, init_betas)
                                if self.w_init_joints > 0 else None),
        }
        return preds

    # ──────────────────────────────────────────────────────────────────────────
    # Stage-A: 2D-only front-end (no fitter, no 3D) — per-view, single-frame
    # ──────────────────────────────────────────────────────────────────────────

    def _forward_2d(self, batch):
        """Run only the per-view 2D front-end (backbone+neck+heads+consensus).

        All inputs are treated as independent images (the multi-view grouping is
        irrelevant to the 2D losses); the fitter / init head / 3D decode are
        skipped. Returns the same 2D keys as ``_forward_impl`` so the loss code
        and metrics are shared.
        """
        images = batch["image"]
        if images.dim() == 5:
            images = images.view(-1, *images.shape[-3:])
        feats = self._backbone_forward(images)
        pyramid = self.neck(feats)
        hm_logits, hm_probs, hm_coords = self.heatmap_head(pyramid[0])
        dovf_field, dovf_per_scale = self.dovf_head(pyramid)
        consensus_2d, consensus_conf = dovf_consensus_2d(hm_probs, dovf_field) \
            if self.heatmap_target == self.dovf_target else (None, None)
        # Option-1: MANO-init prior. Per-scene pooled descriptor -> init head ->
        # MANO root-relative joints (fp32; MANO dislikes bf16). Also exposes
        # init pose/betas for the MANO viz panel in the 2D stage.
        init_joints_rel = init_pose = init_betas = None
        if self.w_init_joints > 0:
            cvn = batch["cam_view_num"]
            desc_view = pyramid[-1].mean(dim=(-2, -1))           # (BN, feat_dim)
            descs = [desc_view[int(np.sum(cvn[:i])):int(np.sum(cvn[:i + 1]))].mean(0, keepdim=True)
                     for i in range(len(cvn))]
            desc = torch.cat(descs, 0)                            # (B, feat_dim)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                init_pose, init_betas = self.init_head(desc.float())
                init_joints_rel = self._mano_init_joints_rel(init_pose, init_betas)
        return {
            "hm_logits": hm_logits, "hm_probs": hm_probs, "hm_coords": hm_coords,
            "dovf_field": dovf_field, "dovf_per_scale": dovf_per_scale,
            "consensus_2d": consensus_2d, "consensus_conf": consensus_conf,
            "init_joints_rel": init_joints_rel,
            "init_pose": init_pose, "init_betas": init_betas,
        }

    def _gt_2d_hm(self, batch, scale_hm):
        """GT 2D for the heatmap and DOVF targets, in heatmap px (per view)."""
        B = len(batch["cam_view_num"])
        gt_joints = batch["master_joints_3d"].view(B, self.num_joints, 3)
        gt_verts = batch["master_verts_3d"].view(B, self.num_verts, 3)
        gt_hm_pts = gt_joints if self.heatmap_target == "joints" else gt_verts
        gt_hm_2d = self._project_to_views(gt_hm_pts, batch, self.n_hm_pts) * scale_hm
        if self.dovf_target == self.heatmap_target:
            gt_dovf_2d = gt_hm_2d
        else:
            gt_dovf_pts = gt_joints if self.dovf_target == "joints" else gt_verts
            gt_dovf_2d = self._project_to_views(gt_dovf_pts, batch, self.n_dovf_pts) * scale_hm
        return gt_hm_2d, gt_dovf_2d

    def compute_loss_2d(self, preds, batch):
        """2D-only losses (heatmap + DOVF field + deep-sup + consensus). No 3D."""
        device = preds["hm_logits"].device
        scale_hm = self._hm_scale(device, preds["hm_logits"].dtype)        # img->hm
        gt_hm_2d, gt_dovf_2d = self._gt_2d_hm(batch, scale_hm)

        loss_hm = self._heatmap_loss(preds, gt_hm_2d)

        h, w = self.heatmap_size
        gt_field_full = build_dovf_target(gt_dovf_2d, h, w)
        loss_dovf = self._dovf_l1(preds["dovf_field"], gt_field_full, gt_dovf_2d)
        loss_dovf_deep = torch.zeros((), device=device)
        if self.deep_supervision and len(preds["dovf_per_scale"]) > 1:
            n_extra = 0
            bn = gt_field_full.shape[0]
            gt_flat = gt_field_full.view(bn, -1, h, w)
            for fld in preds["dovf_per_scale"][:-1]:
                fh, fw = fld.shape[-2:]
                tgt = F.interpolate(gt_flat, size=(fh, fw), mode="bilinear", align_corners=False)
                loss_dovf_deep = loss_dovf_deep + F.l1_loss(fld, tgt)
                n_extra += 1
            if n_extra:
                loss_dovf_deep = loss_dovf_deep / n_extra

        loss_cons = torch.zeros((), device=device)
        if preds["consensus_2d"] is not None:
            gt_cons = gt_hm_2d if self.dovf_target == self.heatmap_target else gt_dovf_2d
            loss_cons = self.criterion_2d(preds["consensus_2d"], gt_cons)

        loss_init = torch.zeros((), device=device)
        if self.w_init_joints > 0 and preds.get("init_joints_rel") is not None:
            B = len(batch["cam_view_num"])
            gt_j = batch["master_joints_3d"].view(B, self.num_joints, 3)
            gt_rel = gt_j - gt_j[:, self.center_idx:self.center_idx + 1]
            loss_init = self.criterion_3d(preds["init_joints_rel"], gt_rel)

        loss = (self.w_heatmap * loss_hm
                + self.w_dovf * loss_dovf
                + self.w_dovf_deep * loss_dovf_deep
                + self.w_consensus * loss_cons
                + self.w_init_joints * loss_init)
        loss_dict = {"loss": loss, "loss_hm": loss_hm, "loss_dovf": loss_dovf,
                     "loss_dovf_deep": loss_dovf_deep, "loss_cons": loss_cons,
                     "loss_init": loss_init}
        return loss, loss_dict, gt_hm_2d, gt_dovf_2d

    def _2d_px_err(self, preds, gt_hm_2d, gt_dovf_2d, scale_hm):
        """Reprojection error in IMAGE px (256) for consensus + heatmap argmax."""
        inv = 1.0 / scale_hm                                               # hm->img
        gt_cons = gt_dovf_2d if self.dovf_target != self.heatmap_target else gt_hm_2d
        cons_px = hm_px = float("nan")
        if preds["consensus_2d"] is not None:
            cons_px = ((preds["consensus_2d"] - gt_cons) * inv).norm(dim=-1).mean().item()
        hm_px = ((preds["hm_coords"] - gt_hm_2d) * inv).norm(dim=-1).mean().item()
        return cons_px, hm_px

    def _init_translation(self, consensus_2d, consensus_conf, hm_coords, dovf_field,
                          K_all, w2c_all, extr_all, s, e, N, scale_hm2img, device):
        """Triangulate the consensus joints to a world wrist position (DLT)."""
        # 2D joints in heatmap px -> image px.
        if consensus_2d is not None:
            j2d_hm = consensus_2d[s:e]                              # (N, P, 2)
            conf = consensus_conf[s:e]
        else:
            j2d_hm = hm_coords[s:e]
            conf = torch.ones(N, hm_coords.shape[1], device=device, dtype=hm_coords.dtype)
        j2d_img = j2d_hm * scale_hm2img                            # (N, P, 2) image px

        if N >= 2:
            u2d = j2d_img.unsqueeze(0)                             # (1,N,P,2)
            K = K_all[s:e].unsqueeze(0)                            # (1,N,3,3)
            w2c = w2c_all[s:e].unsqueeze(0)                        # (1,N,4,4) world->cam
            w = conf.unsqueeze(0).clamp(min=1e-3)                  # (1,N,P)
            pts3d = weighted_dlt(u2d, K, w2c, w)                   # (1,P,3) world
            # Detach: the 2-view DLT/SVD Jacobian is ill-conditioned (O(1/baseline));
            # the fitter refines translation and the 3D loss trains the DOVF field
            # through the optimizer, so the init need not carry gradient.
            wrist = pts3d[:, self.center_idx, :].detach()          # (1,3)
            return wrist
        # N == 1 fallback: unproject the wrist at nominal depth, lift to world.
        K1 = K_all[s]; extr1 = extr_all[s]                         # T_c2m
        uv = j2d_img[0, self.center_idx]                          # (2,)
        z = torch.tensor(self.nominal_root_depth, device=device, dtype=uv.dtype)
        x = (uv[0] - K1[0, 2]) * z / K1[0, 0]
        y = (uv[1] - K1[1, 2]) * z / K1[1, 1]
        cam_pt = torch.stack([x, y, z])
        world = (extr1[:3, :3] @ cam_pt) + extr1[:3, 3]
        return world.unsqueeze(0).detach()

    def _run_optimizer(self, pose0, betas0, trans0, dovf_field, K_all, w2c_all, s, e, N, device):
        """Run the Theseus DOVF optimizer for one scene (fp32, no autocast)."""
        import theseus as th
        n_key = min(N, self.max_views)
        opt = self.optimizers[str(n_key)]
        views = slice(s, s + n_key)

        # joint DOVF for these views: (n,Pd,2,h,w) -> (1,n,h,w,J,2)
        fld = dovf_field[views]                                    # (n,J,2,h,w)
        fld = fld.permute(0, 3, 4, 1, 2).unsqueeze(0).contiguous() # (1,n,h,w,J,2)

        # heatmap-space intrinsics for these views.
        K_hm = K_all[views].clone()
        sx, sy = self._hm_scale(device, K_hm.dtype)
        K_hm[:, 0, :] = K_hm[:, 0, :] * sx
        K_hm[:, 1, :] = K_hm[:, 1, :] * sy
        K_hm = K_hm.unsqueeze(0)                                   # (1,n,3,3)
        w2c = w2c_all[views].unsqueeze(0)                          # (1,n,4,4)

        with torch.cuda.amp.autocast(enabled=False):
            pose_opt, trans_opt = opt(
                pose0.float(), trans0.float(), betas0.float(),
                K_hm.float(), w2c.float(), fld.float(),
                backward_mode=th.BackwardMode.IMPLICIT,
            )
        return pose_opt, trans_opt

    def _run_fitter_analytic(self, pose0, betas, trans0, dovf_field, K_all, w2c_all, cvn, device,
                             mode="train", conf=None):
        """Single batched analytic GN fit over all scenes (pad to N_max, zero-weight pad).

        ~100x faster than the per-scene Theseus loop because the analytic fitter
        batches over scenes (verified identical outputs). Differentiable wrt the
        DOVF field, pose/trans init, and the learnable Huber radius.
        """
        B = len(cvn)
        Nmax = int(max(cvn))
        hm_h, hm_w = self.heatmap_size
        J = self.num_joints
        sx, sy = self._hm_scale(device, K_all.dtype)

        # heatmap-space intrinsics for all views
        K_hm_all = K_all.clone()
        K_hm_all[:, 0, :] = K_hm_all[:, 0, :] * sx
        K_hm_all[:, 1, :] = K_hm_all[:, 1, :] * sy

        K_pad = K_hm_all.new_zeros(B, Nmax, 3, 3)
        w2c_pad = w2c_all.new_zeros(B, Nmax, 4, 4)
        w2c_pad[..., :, :] = torch.eye(4, device=device, dtype=w2c_all.dtype)
        dovf_pad = dovf_field.new_zeros(B, Nmax, hm_h, hm_w, J, 2)
        vmask = torch.zeros(B, Nmax, device=device, dtype=dovf_field.dtype)
        # per-(view,joint) reliability weight from the consensus confidence (heatmap peak)
        use_conf = self.conf_weight and conf is not None
        conf_pad = dovf_field.new_zeros(B, Nmax, J) if use_conf else None
        for i in range(B):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
            K_pad[i, :N] = K_hm_all[s:e]
            w2c_pad[i, :N] = w2c_all[s:e]
            # (N,J,2,h,w) -> (N,h,w,J,2)
            dovf_pad[i, :N] = dovf_field[s:e].permute(0, 3, 4, 1, 2)
            vmask[i, :N] = 1.0
            if use_conf:
                conf_pad[i, :N] = conf[s:e]

        # fp32 (autocast off): the GN linear solve is unstable in bf16/fp16.
        with torch.cuda.amp.autocast(enabled=False):
            pose_opt, trans_opt = analytic_fit(
                self.mano_layer, self.J_regressor,
                pose0.float(), trans0.float(), betas.float(),
                K_pad.float(), w2c_pad.float(), dovf_pad.float(),
                num_joints=J, H=hm_h, W=hm_w,
                max_iterations=self.fit_iters, step_size=self.fit_step,
                pose_prior_weight=self.fit_prior, log_radius=self.fit_log_radius.float(),
                view_weight=vmask.float(),
                jac_mode=(self.jac_mode_train if mode == "train" else self.jac_mode_eval),
                center_idx=self.center_idx, backward=self.fit_backward,
                prior_prec=getattr(self, "prior_prec", None),
                prior_ref=getattr(self, "prior_ref", None),
                prior_anchor=getattr(self, "prior_anchor", None),
                joint_conf=(conf_pad.float() if use_conf else None),
            )
        return pose_opt, trans_opt

    def _dlt_3d(self, consensus_2d, consensus_conf, K_all, w2c_all, s, e, N, scale_hm2img, device):
        """dlt backend: triangulate the consensus vertex field -> world mesh + joints.

        2D coords are detached before the SVD (the 2-view DLT Jacobian is
        ill-conditioned); the vertex field is trained by the 2D heatmap / DOVF /
        consensus losses, mirroring the stable 2D-mode of HRNetVertexHeatmapOcc.
        """
        v2d_img = (consensus_2d[s:e] * scale_hm2img).detach()          # (N, 778, 2) img px
        if N >= 2:
            u2d = v2d_img.unsqueeze(0)
            K = K_all[s:e].unsqueeze(0)
            w2c = w2c_all[s:e].unsqueeze(0)
            w = consensus_conf[s:e].unsqueeze(0).clamp(min=1e-3)
            verts_w = weighted_dlt(u2d, K, w2c, w)                     # (1, 778, 3) world
        else:
            verts_w = torch.zeros(1, self.num_verts, 3, device=device, dtype=v2d_img.dtype)
        joints_w = mano_to_openpose(self.J_regressor, verts_w)[:, : self.num_joints]
        return joints_w, verts_w

    # ──────────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────────

    def compute_loss(self, preds, batch, epoch_idx=None):
        device = preds["hm_logits"].device
        B = len(batch["cam_view_num"])
        gt_joints = batch["master_joints_3d"].view(B, self.num_joints, 3)
        gt_verts = batch["master_verts_3d"].view(B, self.num_verts, 3)

        scale_hm = self._hm_scale(device, preds["hm_logits"].dtype)   # img->hm

        # GT 2D for heatmap target (joints or verts) in heatmap px.
        gt_hm_pts_world = gt_joints if self.heatmap_target == "joints" else gt_verts
        gt_hm_2d_img = self._project_to_views(gt_hm_pts_world, batch, self.n_hm_pts)
        gt_hm_2d = gt_hm_2d_img * scale_hm
        # GT 2D for dovf target.
        if self.dovf_target == self.heatmap_target:
            gt_dovf_2d = gt_hm_2d
        else:
            gt_dovf_pts = gt_joints if self.dovf_target == "joints" else gt_verts
            gt_dovf_2d = self._project_to_views(gt_dovf_pts, batch, self.n_dovf_pts) * scale_hm

        # ── 2D heatmap loss ──
        loss_hm = self._heatmap_loss(preds, gt_hm_2d)

        # ── DOVF field loss (final + deep supervision) ──
        h, w = self.heatmap_size
        gt_field_full = build_dovf_target(gt_dovf_2d, h, w)           # (BN,Pd,2,h,w)
        loss_dovf = self._dovf_l1(preds["dovf_field"], gt_field_full, gt_dovf_2d)
        loss_dovf_deep = torch.zeros((), device=device)
        if self.deep_supervision and len(preds["dovf_per_scale"]) > 1:
            n_extra = 0
            bn = gt_field_full.shape[0]
            gt_flat = gt_field_full.view(bn, -1, h, w)                # (bn, P*2, h, w)
            # per-scale fields are (bn, P*2, fh, fw) — compare in the same layout.
            for fld in preds["dovf_per_scale"][:-1]:                  # skip finest dup
                fh, fw = fld.shape[-2:]
                tgt = F.interpolate(gt_flat, size=(fh, fw), mode="bilinear", align_corners=False)
                loss_dovf_deep = loss_dovf_deep + F.l1_loss(fld, tgt)
                n_extra += 1
            if n_extra:
                loss_dovf_deep = loss_dovf_deep / n_extra

        # ── Consensus 2D loss ──
        loss_cons = torch.zeros((), device=device)
        if preds["consensus_2d"] is not None:
            gt_cons = gt_hm_2d if self.dovf_target == self.heatmap_target else gt_dovf_2d
            loss_cons = self.criterion_2d(preds["consensus_2d"], gt_cons)

        # ── Init pose / shape regularisation ──
        loss_pose = (preds["init_pose"] ** 2).mean()
        loss_betas = (preds["init_betas"] ** 2).mean()

        # ── Option-1: MANO-init root-relative joint prior (active from epoch 0,
        # NOT ramped — it shapes the optimizer's starting basin) ──
        loss_init = torch.zeros((), device=device)
        if self.w_init_joints > 0 and preds.get("init_joints_rel") is not None:
            gt_rel = gt_joints - gt_joints[:, self.center_idx:self.center_idx + 1]
            loss_init = self.criterion_3d(preds["init_joints_rel"], gt_rel)

        # ── 3D loss (ramped after warmup) ──
        ramp = self._ramp(epoch_idx)
        loss_j3d = self.criterion_3d(preds["pred_joints_3d"], gt_joints)
        loss_v3d = self.criterion_3d(preds["pred_verts_3d"], gt_verts)

        loss = (self.w_heatmap * loss_hm
                + self.w_dovf * loss_dovf
                + self.w_dovf_deep * loss_dovf_deep
                + self.w_consensus * loss_cons
                + self.w_pose_reg * loss_pose
                + self.w_betas_reg * loss_betas
                + self.w_init_joints * loss_init
                + ramp * (self.w_joints_3d * loss_j3d + self.w_verts_3d * loss_v3d))

        loss_dict = {
            "loss": loss, "loss_hm": loss_hm, "loss_dovf": loss_dovf,
            "loss_dovf_deep": loss_dovf_deep, "loss_cons": loss_cons,
            "loss_init": loss_init,
            "loss_j3d": loss_j3d, "loss_v3d": loss_v3d, "ramp": torch.tensor(ramp),
        }
        return loss, loss_dict

    # ──────────────────────────────────────────────────────────────────────────
    # Train / val / test
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, step_idx, epoch_idx=0, **kwargs):
        if self.pretrain_2d:
            return self._training_step_2d(batch, step_idx)
        B = len(batch["cam_view_num"])
        gt_joints = batch["master_joints_3d"].view(B, self.num_joints, 3)
        preds = self._forward_impl(batch, mode="train", epoch_idx=epoch_idx)
        loss, loss_dict = self.compute_loss(preds, batch, epoch_idx=epoch_idx)

        self.MPJPE_3D_TRAIN.feed(preds["pred_joints_3d"], gt_kp=gt_joints)
        # instantaneous (this-batch) MPJPE in mm — the real-time trend (the
        # MeanEPE above is the lagging cumulative epoch average)
        with torch.no_grad():
            self._last_mpjpe_mm = (preds["pred_joints_3d"] - gt_joints).norm(dim=-1).mean().item() * 1000.0
        self.loss_metric.feed(loss_dict, B)
        if step_idx % self.train_log_interval == 0 and getattr(self, "summary", None):
            for k, v in loss_dict.items():
                self.summary.add_scalar(k, float(v.item() if torch.is_tensor(v) else v), step_idx)
            self.summary.add_scalar("mpjpe_mm_inst", self._last_mpjpe_mm, step_idx)
        self._maybe_viz(batch, preds, "train", step_idx)
        return preds, loss_dict

    def _training_step_2d(self, batch, step_idx):
        """Stage-A: per-view 2D-only training step (no fitter / 3D)."""
        preds = self._forward_2d(batch)
        loss, loss_dict, gt_hm_2d, gt_dovf_2d = self.compute_loss_2d(preds, batch)
        bn = preds["hm_logits"].shape[0]
        self.loss_metric.feed(loss_dict, bn)
        # Only compute the px metric (which .item()-syncs the GPU) on log steps,
        # so the training step doesn't stall on a host sync every iteration.
        if step_idx % self.train_log_interval == 0:
            scale_hm = self._hm_scale(preds["hm_logits"].device, preds["hm_logits"].dtype)
            with torch.no_grad():
                self._last_cons_px, self._last_hm_px = self._2d_px_err(
                    preds, gt_hm_2d, gt_dovf_2d, scale_hm)
            if getattr(self, "summary", None):
                for k, v in loss_dict.items():
                    self.summary.add_scalar(k, float(v.item() if torch.is_tensor(v) else v), step_idx)
                self.summary.add_scalar("consensus_2d_px_inst", self._last_cons_px, step_idx)
        self._maybe_viz(batch, preds, "train", step_idx)
        return preds, loss_dict

    def _ds_bucket(self, name):
        """Lazily-created per-dataset val metric bucket (keyed by dataset_name)."""
        b = self._per_ds_metrics.get(name)
        if b is None:
            b = {
                "cons2d": MeanEPE(self._metric_cfg, f"{name}_consensus_2d_px"),
                "hm2d": MeanEPE(self._metric_cfg, f"{name}_heatmap_2d_px"),
                "dlt_cons": MeanEPE(self._metric_cfg, f"{name}_dlt_cons_mpjpe"),
                "dlt_hm": MeanEPE(self._metric_cfg, f"{name}_dlt_hm_mpjpe"),
            }
            self._per_ds_metrics[name] = b
        return b

    def _eval_step_2d(self, batch, step_idx):
        """Stage-A val: per-view 2D reprojection px + cheap multi-view DLT 3D MPJPE
        (no Theseus). Triangulates the heatmap-argmax 2D and the DOVF-consensus 2D
        across each scene's views -> 3D, vs GT joints. The consensus-DLT 3D is the
        headline number to compare against POEM's 2-view MPJPE. Metrics are fed both
        in aggregate and per-dataset (bucketed by the per-sample dataset_name tag)."""
        preds = self._forward_2d(batch)
        device = preds["hm_logits"].device
        scale_hm = self._hm_scale(device, preds["hm_logits"].dtype)        # img->hm
        inv = 1.0 / scale_hm                                               # hm->img px
        gt_hm_2d, gt_dovf_2d = self._gt_2d_hm(batch, scale_hm)
        gt_cons = gt_dovf_2d if self.dovf_target != self.heatmap_target else gt_hm_2d

        # ── 2D reprojection error (image px) ──
        if preds["consensus_2d"] is not None:
            self.CONS2D_PX.feed(preds["consensus_2d"] * inv, gt_kp=gt_cons * inv)
        self.HM2D_PX.feed(preds["hm_coords"] * inv, gt_kp=gt_hm_2d * inv)

        # ── multi-view DLT triangulation -> 3D MPJPE (no solver) ──
        B = len(batch["cam_view_num"]); cvn = batch["cam_view_num"]; J = self.num_joints
        K_all = batch["target_cam_intr"].view(-1, 3, 3)
        w2c_all = torch.linalg.inv(batch["target_cam_extr"].view(-1, 4, 4))  # world->cam
        gt_joints = batch["master_joints_3d"].view(B, J, 3)
        conf_all = preds["consensus_conf"]
        names = batch.get("dataset_name", None)        # per-scene source tag (len B) or None
        cons_3d, hm_3d, gt_keep = [], [], []
        for i in range(B):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
            nm = names[i] if names is not None else None
            # per-dataset 2D px (per view; no view-count requirement)
            if nm is not None:
                bkt = self._ds_bucket(nm)
                if preds["consensus_2d"] is not None:
                    bkt["cons2d"].feed(preds["consensus_2d"][s:e] * inv, gt_kp=gt_cons[s:e] * inv)
                bkt["hm2d"].feed(preds["hm_coords"][s:e] * inv, gt_kp=gt_hm_2d[s:e] * inv)
            if N < 2:                                  # DLT needs >=2 views
                continue
            K = K_all[s:e].unsqueeze(0); w2c = w2c_all[s:e].unsqueeze(0)
            w = (conf_all[s:e] if conf_all is not None
                 else gt_joints.new_ones(N, J)).unsqueeze(0).clamp(min=1e-3)
            hm_img = (preds["hm_coords"][s:e] * inv).unsqueeze(0)          # (1,N,J,2) img px
            hm_tri = weighted_dlt(hm_img, K, w2c, w)                       # (1,J,3) master
            cons_tri = None
            if preds["consensus_2d"] is not None:
                cons_img = (preds["consensus_2d"][s:e] * inv).unsqueeze(0)
                cons_tri = weighted_dlt(cons_img, K, w2c, w)
            # per-dataset DLT-3D (kept even for coord-frame-artefact sets like Interhand)
            if nm is not None:
                self._ds_bucket(nm)["dlt_hm"].feed(hm_tri, gt_kp=gt_joints[i:i + 1])
                if cons_tri is not None:
                    self._ds_bucket(nm)["dlt_cons"].feed(cons_tri, gt_kp=gt_joints[i:i + 1])
            # aggregate DLT-3D excludes coord-frame-artefact datasets so the headline
            # proxy isn't polluted (e.g. Interhand raw-DLT ~38mm; see results_p0.md)
            if nm is not None and nm in self._agg3d_exclude:
                continue
            hm_3d.append(hm_tri)
            if cons_tri is not None:
                cons_3d.append(cons_tri)
            gt_keep.append(gt_joints[i:i + 1])
        if gt_keep:
            gt_cat = torch.cat(gt_keep, 0)
            self.DLT_HM_MPJPE.feed(torch.cat(hm_3d, 0), gt_kp=gt_cat)
            if cons_3d:
                self.DLT_CONS_MPJPE.feed(torch.cat(cons_3d, 0), gt_kp=gt_cat)
        self._maybe_viz(batch, preds, "val", step_idx)
        return preds

    def _eval_step(self, batch, step_idx):
        if self.pretrain_2d:
            return self._eval_step_2d(batch, step_idx)
        B = len(batch["cam_view_num"])
        preds = self._forward_impl(batch, mode="val")
        gt_joints = batch["master_joints_3d"].view(B, self.num_joints, 3)
        gt_verts = batch["master_verts_3d"].view(B, self.num_verts, 3)
        pj = preds["pred_joints_3d"]; pv = preds["pred_verts_3d"]
        pr = pj[:, self.center_idx:self.center_idx + 1]
        gr = gt_joints[:, self.center_idx:self.center_idx + 1]
        self.MPJPE_3D.feed(pj, gt_kp=gt_joints)
        self.MPVPE_3D.feed(pv, gt_kp=gt_verts)
        self.PA.feed(pj, gt_joints, pv, gt_verts)
        self.MPJPE_3D_REL.feed(pj - pr, gt_kp=gt_joints - gr)
        self.MPVPE_3D_REL.feed(pv - pr, gt_kp=gt_verts - gr)
        self._maybe_viz(batch, preds, "val", step_idx)
        return preds

    def validation_step(self, batch, step_idx, **kwargs):
        return self._eval_step(batch, step_idx)

    def testing_step(self, batch, step_idx, **kwargs):
        return self._eval_step(batch, step_idx)

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self._log_metric_scalars([self.loss_metric], comment, epoch_idx)
        if not self.pretrain_2d:
            recorder.record_metric([self.MPJPE_3D_TRAIN], epoch_idx, comment=comment)
            self._log_metric_scalars([self.MPJPE_3D_TRAIN], comment, epoch_idx)
        self.loss_metric.reset(); self.MPJPE_3D_TRAIN.reset()

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        if self.pretrain_2d:
            agg = [self.CONS2D_PX, self.HM2D_PX, self.DLT_CONS_MPJPE, self.DLT_HM_MPJPE]
            recorder.record_metric(agg, epoch_idx, comment=comment)
            self._log_metric_scalars(agg, comment, epoch_idx)
            for m in agg:
                m.reset()
            # per-dataset breakdown (one checkpoint, evaluated on all 4 eval sets)
            for name, b in sorted(self._per_ds_metrics.items()):
                ms = [b["cons2d"], b["hm2d"], b["dlt_cons"], b["dlt_hm"]]
                recorder.record_metric(ms, epoch_idx, comment=f"{comment}-{name}")
                self._log_metric_scalars(ms, f"{comment}-{name}", epoch_idx)
                for m in b.values():
                    m.reset()
            return
        agg = [self.MPJPE_3D, self.MPVPE_3D, self.MPJPE_3D_REL, self.MPVPE_3D_REL, self.PA]
        recorder.record_metric(agg, epoch_idx, comment=comment)
        self._log_metric_scalars(agg, comment, epoch_idx)
        for m in agg:
            m.reset()

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        self.on_val_finished(recorder, epoch_idx, **kwargs)

    def format_metric(self, mode="val"):
        if self.pretrain_2d:
            if mode == "train":
                c = getattr(self, "_last_cons_px", float("nan"))
                h = getattr(self, "_last_hm_px", float("nan"))
                return f"consensus2D={c:.2f}px hm_argmax={h:.2f}px (img)"
            return (f"cons2D={self.CONS2D_PX.get_result():.2f}px | "
                    f"DLT_cons={self.DLT_CONS_MPJPE.get_result() * 1000:.1f}mm | "
                    f"DLT_hm={self.DLT_HM_MPJPE.get_result() * 1000:.1f}mm")
        if mode == "train":
            inst = getattr(self, "_last_mpjpe_mm", float("nan"))
            return f"MPJPE_inst={inst:.1f}mm avg={self.MPJPE_3D_TRAIN.get_result() * 1000:.1f}mm"
        return f"MPJPE={self.MPJPE_3D.get_result():.4f} | MPVPE={self.MPVPE_3D.get_result():.4f}"

    def forward(self, batch, step_idx, mode="val", epoch_idx=0, **kwargs):
        self._cur_epoch = epoch_idx
        if mode == "train":
            return self.training_step(batch, step_idx, epoch_idx=epoch_idx, **kwargs)
        return self._eval_step(batch, step_idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Optimizer param groups
    # ──────────────────────────────────────────────────────────────────────────

    def get_param_groups(self, train_cfg):
        base_lr = float(train_cfg.LR)
        bb_scale = float(train_cfg.get("BACKBONE_LR_SCALE", 1.0))
        bb_params, other = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (bb_params if n.startswith("img_backbone.") else other).append(p)
        groups = [{"params": other, "lr": base_lr, "name": "heads"}]
        if bb_params:
            groups.append({"params": bb_params, "lr": base_lr * bb_scale, "name": "backbone"})
        return groups
