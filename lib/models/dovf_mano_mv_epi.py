"""
DOVFManoMVEpi — precision-voting + recurrent epipolar information-filter refiner.
=================================================================================

Subclass of :class:`DOVFManoMVUnc` that REPLACES the inert mean-pool sigma-point
scorer (``MixtureMCHead``) with the principled epipolar pipeline:

  Phase A  per-vote precision voting  -> consensus + fused Ω2   (precision_consensus_2d)
  Phase B  param-free Ω2 triangulation -> belief (μ, Σ3)        (triangulate_omega2, KEEP)
  Phase C  recurrent epipolar information filter -> refined μ   (run_epipolar_filter)

The 3D fusion is parameter-free; the only learned pieces are the 2D front-end
(offset + per-vote precision, inherited heads) and the shared ``EpipolarRefiner``.
Output ``pred_joints_3d`` = the refined belief mean μ (predictor-only; no MANO fit,
verts are zeros — the headline metric is MPJPE). Leaves ``DOVFManoMVUnc`` (FRcombo)
untouched. Config: ``MODEL.TYPE: DOVFManoMVEpi`` + an ``EPI`` block; set
``MC_HEAD: false`` (we don't build the scorer) but keep the ``unc_head`` (per-vote chol).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint

from lib.utils.builder import MODEL
from lib.models.dovf_mano_mv_unc import DOVFManoMVUnc
from lib.models.dovf.analytic_fitter_unc import gaussian_nll
from lib.models.dovf.triangulate_resample import _pad_by_scene, triangulate_omega2
from lib.models.dovf.epipolar_filter import (
    precision_consensus_2d, gaussian_nll_3d_from_prec, EpipolarRefiner,
    run_epipolar_filter, run_heatmap_filter, proj_and_jac)
from lib.models.dovf import build_dovf_target


@MODEL.register_module()
class DOVFManoMVEpi(DOVFManoMVUnc):
    """Precision-voting + epipolar information-filter DOVF estimator."""

    def __init__(self, cfg):
        super().__init__(cfg)
        epi = cfg.get("EPI", {})
        self.T = int(epi.get("T", 3))
        self.D = int(epi.get("D", 16))           # depth hyps along the epipolar line
        self.M = int(epi.get("M", 9))            # in-plane window samples
        self.kappa = float(epi.get("KAPPA", 2.0))  # beam half-extent in σ_z
        self.meas = str(epi.get("MEAS", "corr2d"))
        self.ess_tau = bool(epi.get("ESS_TAU", False))
        self.detach_2d_for_3d = bool(epi.get("DETACH_2D_FOR_3D", False))
        self.detach_mu_iters = bool(epi.get("DETACH_MU_ITERS", False))
        # Robust IRLS triangulation: reject wrong-hand / occluded outlier views (InterHand
        # two-hand ambiguity) by Mahalanobis-residual reweighting. Inference-time, no new params.
        self.robust_triang = bool(epi.get("ROBUST_TRIANG", False))
        self.robust_c2 = float(epi.get("ROBUST_C2", 25.0))
        self.robust_iter = int(epi.get("ROBUST_ITER", 4))
        self.robust_kind = str(epi.get("ROBUST_KIND", "cauchy"))
        self.robust_master_w = float(epi.get("ROBUST_MASTER_W", 3.0))  # master-anchor weight
        # Fit-in-loop (Stage B): run the analytical LM MANO fit INSIDE training and
        # supervise its joints -> calibrates the fields+Omega2 FOR the solver they feed
        # at inference. Init mirrors inference exactly: Kabsch(flat hand -> mu), beta=0.
        self.fit_in_loop = bool(epi.get("FIT_IN_LOOP", False))
        # BETA_HEAD: predict MANO shape feed-forward (ManoInitHead betas output on the
        # scene-pooled descriptor). Replaces the alternating beta-fit at inference:
        # single LM solve with beta = beta_hat (no L-BFGS, no 3x LM rounds).
        self.beta_head_on = bool(epi.get("BETA_HEAD", False))
        # FIT_GT_BETAS: run the train-time fit-in-loop with the GT MANO shape
        # (calibrated-shape / oracle conditioning) instead of beta_hat/zeros, so
        # the fields + precisions adapt end-to-end around correct bone lengths.
        # The beta head keeps training through its own L2 loss regardless.
        self.fit_gt_betas = bool(epi.get("FIT_GT_BETAS", False))
        # BETA_BONE_FEAT: additive residual to beta_hat from mu's 20 bone lengths
        # (zero-init last layer -> warm-start-identical). Fixes the beta head's
        # regression-to-mean shrinkage: the pooled image descriptor is weakly shape-
        # informative, while mu's bone lengths (accurate to ~6mm) span the beta subspace
        # that drives MPVPE. See docs/dexycb_oakink_gap_analysis.md (2e).
        self.beta_bone_feat = bool(epi.get("BETA_BONE_FEAT", False))
        # BETA_BONE_ONLY: beta_hat = bone_mlp(bones) alone (drop the init_head descriptor
        # term). The offline probe (docs/dexycb_oakink_gap_analysis.md 2e) showed bones-only
        # BEATS desc+bones (IH bone-err 2.36 vs 3.11mm) — the descriptor pathway is noise.
        self.beta_bone_only = bool(epi.get("BETA_BONE_ONLY", False))
        # BETA_TRAIN_ONLY: freeze every parameter except beta_bone_mlp — the pose pathway
        # stays byte-identical to the warm-start checkpoint, so only shape can change and
        # the retrain is (near-)deterministic. Must run in __init__ (before DDP wrap).
        self.beta_train_only = bool(epi.get("BETA_TRAIN_ONLY", False))
        if self.beta_bone_feat:
            self.beta_bone_mlp = torch.nn.Sequential(
                torch.nn.Linear(20, 128), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 128), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 10))
            torch.nn.init.zeros_(self.beta_bone_mlp[-1].weight)
            torch.nn.init.zeros_(self.beta_bone_mlp[-1].bias)
            # OpenPose-21 bone topology (child, parent) for mu -> 20 bone lengths
            self.register_buffer("_bone_child", torch.tensor(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
                persistent=False)
            self.register_buffer("_bone_parent", torch.tensor(
                [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]),
                persistent=False)
        # learned ESS/temperature τ (Ω2 = A·τ); init at the old mc_omega_scale≈0.6
        init_tau = float(epi.get("TAU_INIT", 0.6))
        self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(init_tau))))
        self.refiner = EpipolarRefiner(
            self.feat_dim, kind=str(epi.get("REFINER", "attn")),
            d=int(epi.get("DIM", 128)), heads=int(epi.get("HEADS", 4)),
            layers=int(epi.get("LAYERS", 2)), d_hidden=int(epi.get("HIDDEN", 128)),
            gate_bias=float(epi.get("GATE_BIAS", -2.0)),
            cholM_diag_bias=float(epi.get("CHOLM_BIAS", -1.0)))

        # XGATE: cross-view occlusion gate + bounded 2D correction at the FINAL model's
        # Phase-B triangulation (active at trusted eval, unlike the unc-stage refiner which
        # lives only in DOVFManoMVUnc). Same class + module name (xview_gate, CrossViewRefiner)
        # as the refine unc stage so INIT_FROM a refine unc ckpt loads it by name. gate scales
        # omega2 (down-weights occluded views); corr adjusts the consensus 2D pre-triangulation.
        # Measured motivation (docs/occlusion_2d_ceiling_findings.md): occluded joints are
        # "confident but wrong" and the aleatoric Omega2 alone under-flags them.
        self.use_xgate = bool(epi.get("XGATE", False))
        if self.use_xgate:
            from lib.models.dovf.uncertainty_modules import CrossViewRefiner as _XVR
            self.xview_gate = _XVR(self.feat_dim, d=int(epi.get("XGATE_DIM", 256)),
                                   heads=int(epi.get("XGATE_HEADS", 8)),
                                   layers=int(epi.get("XGATE_LAYERS", 3)), cam_dim=6,
                                   corr_max=float(epi.get("CORR_MAX", 6.0)))
        self._last_gate = None

        # JFORMER: joint-query multi-view deformable cross-attention decoder (medium-size
        # prior head; see lib/models/dovf/jointformer.py for the design rationale). Runs
        # AFTER Phase B/C on the triangulated mu; its last stage becomes mu_final ->
        # pred_joints / fit anchor / trusted-eval mu, so it is active end-to-end at eval.
        self.use_jformer = bool(epi.get("JFORMER", False))
        self.jf_mono_seed = bool(epi.get("JF_MONO_SEED", False))
        # JF_VIT: append frozen WiLoR hand-foundation ViT pyramid levels to the decoder's
        # deformable sampling set. Rationale (measured record): the hand-FM prior is the ONLY
        # lever that ever beat POEM at 2v (C1, results_p0.md), but ViT features KILL the dense
        # DOVF voting (stride-16 too coarse, §2.6) — they need a POINT-SAMPLING consumer,
        # which is exactly what the JointFormer is. HRNet trunk keeps the dense seed;
        # the decoder reads BOTH sources. ViT fully frozen; laterals + 1x1 projections train.
        self.use_jf_vit = bool(epi.get("JF_VIT", False))
        if self.use_jformer and self.use_jf_vit:
            from lib.models.backbones.wilor_vit_pyramid import WiLoRViTPyramid
            from yacs.config import CfgNode as _CN
            _vc = _CN({"WILOR_CKPT": str(epi.get("JF_VIT_CKPT",
                       "lib/external/wilor/pretrained_models/wilor_final.ckpt")),
                       "FREEZE": True, "UNFREEZE_LAST_N": 0})
            self.jf_vit_backbone = WiLoRViTPyramid(_vc)
            # project the first 3 ViT pyramid levels (64/128/256 ch) to feat_dim
            self.jf_vit_proj = torch.nn.ModuleList([
                torch.nn.Conv2d(c, self.feat_dim, 1) for c in (64, 128, 256)])
            cprint(f"[{self.name}] JF_VIT: WiLoR-ViT prior levels -> decoder sampling set", "cyan")
        if self.use_jformer:
            from lib.models.dovf.jointformer import JointFormerHead
            self.jointformer = JointFormerHead(
                self.feat_dim, num_joints=self.num_joints,
                d=int(epi.get("JF_DIM", 256)), layers=int(epi.get("JF_LAYERS", 6)),
                heads=int(epi.get("JF_HEADS", 8)), n_points=int(epi.get("JF_POINTS", 8)),
                n_scales=int(epi.get("JF_SCALES", 3)), ffn=int(epi.get("JF_FFN", 1024)),
                seed_noise=float(epi.get("JF_SEED_NOISE", 0.0)),
                mask_p=float(epi.get("JF_MASK_P", 0.0)),
                view_drop=float(epi.get("JF_VIEW_DROP", 0.0)),
                clean_p=float(epi.get("JF_CLEAN_P", 0.0)))
            cprint(f"[{self.name}] JointFormer d={epi.get('JF_DIM',256)} "
                   f"L={epi.get('JF_LAYERS',6)} P={epi.get('JF_POINTS',8)} "
                   f"S={epi.get('JF_SCALES',3)} | "
                   f"{sum(p.numel() for p in self.jointformer.parameters())/1e6:.2f}M params", "cyan")

        # loss weights
        L = getattr(cfg, "LOSS", None) or {}
        self.w_mu3d_ds = float(L.get("MU3D_DEEPSUP_WEIGHT", 10.0))
        self.w_nll3 = float(L.get("J3D_NLL_WEIGHT", 0.1))
        self.w_vote_nll = float(L.get("VOTE_NLL_WEIGHT", 1.0))
        self.w_cons2d = float(L.get("CONSENSUS_2D_WEIGHT", 1.0))
        self.w_gate = float(L.get("GATE_REG_WEIGHT", 0.01))
        # JointFormer deep supervision (absolute joints, per stage, gamma-discounted).
        self.w_jf = float(L.get("JF_WEIGHT", 10.0))
        self.jf_gamma = float(L.get("JF_GAMMA", 0.8))
        # Occ-supervised XGATE: BCE(gate, 1-occluded) from occ_annotations/v2 (via ANN_DIR).
        self.w_gate_occ = float(L.get("GATE_OCC_WEIGHT", 0.0))
        # OCC-WEIGHTED vote-NLL (mirrors the unc stage): upweight the per-vote precision
        # NLL on occluded (view,joint)s so the fit stage keeps Omega wide where votes are
        # unreliable instead of re-collapsing it. See docs/dexycb_oakink_gap_analysis.md.
        self.w_occ_nll = float(L.get("OCC_NLL_WEIGHT", 0.0))
        self.occ_nll_self = float(L.get("OCC_NLL_SELF", 0.5))
        if (self.use_xgate and self.w_gate_occ > 0) or self.w_occ_nll > 0:
            from lib.utils.transform import mano_to_openpose
            with torch.no_grad():
                _v0 = self.mano_layer(torch.zeros(1, 48), torch.zeros(1, 10)).verts
                _j0 = mano_to_openpose(self.J_regressor, _v0)[:, :self.num_joints]
                _j2v = torch.cdist(_j0[0], _v0[0]).argmin(1)
            self.register_buffer("gate_occ_j2v", _j2v, persistent=False)
        self.w_bidir = float(L.get("BIDIR_WEIGHT", 0.1))
        self.ds_gamma = float(L.get("DEEPSUP_GAMMA", 0.8))
        # Direct beam-matching supervision: teach softw to localize the GT depth along the
        # epipolar beam (decoupled from the gated 3D-loss gradient -> breaks the bootstrap stall).
        self.w_beam_ce = float(L.get("BEAM_CE_WEIGHT", 0.0))
        self.w_fit3d = float(L.get("FIT_JOINTS_WEIGHT", 10.0))   # fit-in-loop joint loss
        self.w_beta = float(L.get("BETA_WEIGHT", 1.0))            # beta-head supervision
        # BETA_DIM_SCALED: per-dim standardized beta loss (anti-shrinkage; see 2e in
        # docs/dexycb_oakink_gap_analysis.md). EMA buffer of GT per-dim std.
        self.beta_dim_scaled = bool(L.get("BETA_DIM_SCALED", False))
        if self.beta_dim_scaled:
            self.register_buffer("_beta_std", torch.ones(10))
        self.beam_ce_temp = float(L.get("BEAM_CE_TEMP", 0.01))   # soft-target temp (m)

        # AUX correspondence task: a descriptor head trained with MANO cross-view correspondence
        # to make the backbone features 3D-consistent -> better 2D localization -> better
        # triangulation at ALL views. Used only in training (inference is param-free triangulation).
        self.desc_dim = int(epi.get("DESC_DIM", 0))
        self.w_corr = float(L.get("CORR_WEIGHT", 0.0))
        if self.desc_dim > 0:
            from lib.models.dovf.descriptor import DescriptorHead
            self.descriptor_head = DescriptorHead(self.feat_dim, self._n_scales, d_desc=self.desc_dim)
            self.register_buffer("_mano_faces", self.mano_layer.th_faces.long(), persistent=False)

        # Phase D: learned 3D hand-structure prior (uncertainty-conditioned). Attacks the 2v depth
        # floor via the hand-pose manifold (not cross-view appearance, which is unrealizable).
        self.pose_refine = bool(epi.get("POSE_REFINE", False))
        # POEM-lite: the refiner additionally fuses per-view geometric evidence (cross-view attention)
        self.pose_mv_evidence = bool(epi.get("POSE_MV_EVIDENCE", False))
        if self.pose_refine:
            from lib.models.dovf.pose_refiner import PoseRefiner3D, PoseRefiner3DMV
            Cls = PoseRefiner3DMV if self.pose_mv_evidence else PoseRefiner3D
            self.pose_refiner = Cls(
                num_joints=self.num_joints, center_idx=self.center_idx,
                d=int(epi.get("POSE_DIM", 256)), heads=int(epi.get("POSE_HEADS", 8)),
                layers=int(epi.get("POSE_LAYERS", 4)), max_corr=float(epi.get("POSE_MAXCORR", 0.05)),
                iters=int(epi.get("POSE_ITERS", 2)), unc_gate=bool(epi.get("POSE_UNC_GATE", False)))
            cprint(f"[{self.name}] {Cls.__name__} d={epi.get('POSE_DIM',256)} "
                   f"layers={epi.get('POSE_LAYERS',4)} iters={epi.get('POSE_ITERS',2)} | "
                   f"{sum(p.numel() for p in self.pose_refiner.parameters())/1e6:.2f}M params", "cyan")

        nparam = sum(p.numel() for p in self.refiner.parameters())
        cprint(f"[{self.name}] EPI refiner={self.refiner.kind} T={self.T} D={self.D} "
               f"M={self.M} κ={self.kappa} | refiner={nparam/1e6:.3f}M params | "
               f"τ0={init_tau} ess={self.ess_tau} detach2d={self.detach_2d_for_3d}", "cyan")

        # ── VERTEX head (POEM-comparable mesh output) ──────────────────────────
        # Predictor-only Epi gives SOTA joints μ but zero verts. Graft the joints-
        # anchored free-vertex MeshDecoder onto μ_final: it Kabsch-places a MANO
        # template at μ then learns the articulation residual -> 778 verts. Joints
        # read back via J_regressor (joints-from-verts) for POEM's MPVPE/MPJPE.
        self.mesh_on = bool(epi.get("MESH_DECODER", False))
        # MESH_IMG: also feed per-vertex IMAGE descriptors (sampled from descriptor_head across
        # views) into the decoder -> injects per-subject shape the joints can't carry.
        self.mesh_img = bool(epi.get("MESH_IMG", False))
        if self.mesh_on:
            from lib.models.dovf.mesh_decoder import MeshDecoder
            from lib.utils.transform import mano_to_openpose
            with torch.no_grad():
                _o = self.mano_layer(torch.zeros(1, 48), torch.zeros(1, 10))
                _Jt = mano_to_openpose(self.J_regressor, _o.verts)[0, :self.num_joints]
                _w = _Jt[self.center_idx]
                Vt_rel = (_o.verts[0] - _w).contiguous()
                Jt_rel = (_Jt - _w).contiguous()
            assert not (self.mesh_img and self.desc_dim <= 0), "MESH_IMG needs DESC_DIM>0"
            self.mesh_decoder = MeshDecoder(
                Vt_rel, Jt_rel, num_joints=self.num_joints, d=int(epi.get("MESH_DIM", 256)),
                heads=int(epi.get("MESH_HEADS", 8)), layers=int(epi.get("MESH_LAYERS", 4)),
                center_idx=self.center_idx, feat_dim=self.desc_dim if self.mesh_img else 0)
            # detach μ feeding the mesh head (don't perturb the SOTA joint pipeline) unless
            # an explicit end-to-end finetune is requested.
            self.mesh_detach_mu = bool(epi.get("MESH_DETACH_MU", True))
            self.w_verts3d = float(L.get("VERTS3D_WEIGHT", 10.0))
            self.w_jfromv = float(L.get("JFROMV_WEIGHT", 10.0))
            self.w_edge = float(L.get("EDGE_WEIGHT", 0.0))
            self.w_normal = float(L.get("NORMAL_WEIGHT", 0.0))
            self.w_lap = float(L.get("LAPLACIAN_WEIGHT", 0.0))
            # MANO mesh edges (V,2) and faces for the surface regularizers, built once.
            faces = self.mano_layer.th_faces.long()
            self.register_buffer("_mesh_faces", faces, persistent=False)
            e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
            e = torch.unique(torch.sort(e, dim=1).values, dim=0)              # (E,2) undirected
            self.register_buffer("_mesh_edges", e, persistent=False)
            cprint(f"[{self.name}] MeshDecoder d={epi.get('MESH_DIM',256)} "
                   f"layers={epi.get('MESH_LAYERS',4)} | "
                   f"{sum(p.numel() for p in self.mesh_decoder.parameters())/1e6:.2f}M params | "
                   f"detach_mu={self.mesh_detach_mu} edge={self.w_edge} normal={self.w_normal} "
                   f"lap={self.w_lap}", "cyan")
        else:
            self.mesh_on = False

        # BETA_TRAIN_ONLY freeze: everything except beta_bone_mlp. Done at the END of
        # __init__ (all modules built) and BEFORE the DDP wrap in the train script —
        # DDP only registers grad hooks for params requiring grad at construction.
        if self.beta_train_only:
            assert self.beta_bone_feat, "BETA_TRAIN_ONLY requires BETA_BONE_FEAT"
            # The base-class INIT_FROM load ran BEFORE the epi modules existed, so
            # refiner/unc_head/descriptor_head/pose_refiner stayed at RANDOM init (they
            # are normally trained by the fit stage — here they'd be frozen random and
            # poison Omega²: measured DexYCB μ 7.1→8.2). Reload the FULL checkpoint now
            # that the graph is complete, THEN freeze.
            _init_from = cfg.get("INIT_FROM", "")
            if _init_from:
                from lib.utils.io_utils import load_model as _lm
                _lm(self, _init_from, strict=False, map_location="cpu")
                cprint(f"[{self.name}] BETA_TRAIN_ONLY: re-loaded FULL ckpt post-construction "
                       f"from {_init_from}", "cyan")
            n_frozen = n_train = 0
            for n, p in self.named_parameters():
                if n.startswith("beta_bone_mlp."):
                    n_train += p.numel()
                else:
                    p.requires_grad_(False); n_frozen += p.numel()
            cprint(f"[{self.name}] BETA_TRAIN_ONLY: {n_train/1e3:.1f}k trainable "
                   f"(beta_bone_mlp) | {n_frozen/1e6:.1f}M frozen", "cyan")

    def train(self, mode=True):
        """BETA_TRAIN_ONLY: keep the frozen graph in EVAL mode during training.
        requires_grad=False does NOT stop BatchNorm running-stat updates — 8 epochs of
        train-mode forwards drift the BN buffers and degrade the frozen backbone (measured:
        DexYCB μ 7.1→8.2). Eval-mode BN uses the frozen stats → checkpoint buffers stay
        byte-identical to the warm start AND the β̂ head trains on eval-consistent features."""
        super().train(mode)
        if getattr(self, "beta_train_only", False) and mode:
            for name, mod in self.named_modules():
                if name and not name.startswith("beta_bone_mlp"):
                    mod.eval()
            self.training = True                      # keep top-level train-mode semantics
            self.beta_bone_mlp.train()
        return self

    # ──────────────────────────────────────────────────────────────────────────
    def _forward_impl(self, batch, mode="val", epoch_idx=0):
        images = batch["image"]
        if images.dim() == 5:
            images = images.view(-1, *images.shape[-3:])
        K_all = batch["target_cam_intr"].view(-1, 3, 3)
        extr_all = batch["target_cam_extr"].view(-1, 4, 4)
        w2c_all = torch.linalg.inv(extr_all)
        cvn = batch["cam_view_num"]
        device = images.device

        feats = self._backbone_forward(images)
        pyramid = self.neck(feats)
        hm_logits, hm_probs, hm_coords = self.heatmap_head(pyramid[0])
        dovf_field, dovf_per_scale = self.dovf_head(pyramid)
        cholP = self.unc_head(pyramid[0])                              # (BN,J,3,h,w) per-VOTE chol
        desc = self.descriptor_head(pyramid) if self.desc_dim > 0 else None  # aux correspondence

        # ── PHASES A/B/C run in fp32 (linalg: solve/cholesky/inv/eigh) ──
        with torch.cuda.amp.autocast(enabled=False):
            sx, sy = self._hm_scale(device, torch.float32)
            K_hm = K_all.float().clone()
            K_hm[:, 0, :] = K_hm[:, 0, :] * sx; K_hm[:, 1, :] = K_hm[:, 1, :] * sy
            feat0 = pyramid[0].float()

            # PHASE A — precision-weighted voting
            cons, omega2 = precision_consensus_2d(
                hm_probs.float(), dovf_field.float(), cholP.float(), self.log_tau,
                ess=self.ess_tau)
            cons_tri = cons.detach() if self.detach_2d_for_3d else cons

            # PHASE B — param-free Ω2 triangulation -> belief (μ, Σ3)
            Nmax = int(max(cvn))
            cons_pad, vmask = _pad_by_scene(cons_tri, cvn, Nmax)
            K_pad, _ = _pad_by_scene(K_hm, cvn, Nmax)
            w2c_pad, _ = _pad_by_scene(w2c_all.float(), cvn, Nmax)
            om_pad, _ = _pad_by_scene(omega2, cvn, Nmax)
            # XGATE: cross-view occlusion gate + bounded 2D correction, pre-triangulation.
            self._last_gate = None
            if self.use_xgate:
                from lib.models.dovf.uncertainty_modules import sample_tokens, camera_encoding
                hm_hh, hm_ww = hm_probs.shape[-2:]
                tok = sample_tokens(feat0, cons.detach(), hm_hh, hm_ww)        # (BN,J,C)
                cam = camera_encoding(w2c_all.float())                          # (BN,6)
                tok_pad, _ = _pad_by_scene(tok, cvn, Nmax)
                cam_pad, _ = _pad_by_scene(cam, cvn, Nmax)
                g_pad, c_pad, _xaux = self.xview_gate(tok_pad, cam_pad, vmask)  # (B,N,J),(B,N,J,2)
                cons_pad = cons_pad + c_pad                                     # corrected 2D evidence
                om_pad = om_pad * g_pad.clamp(min=1e-3).unsqueeze(-1).unsqueeze(-1)
                self._last_gate = g_pad
            if os.environ.get("TRIANG_ISO"):
                # Eval-only COMPLETE-uncertainty ablation: replace the learned per-view
                # anisotropic Omega2 with identity so mu3D becomes an UNWEIGHTED (DLT-style)
                # triangulation. Combined with the solver's UNC_OFF (Omega=I in the 2D term)
                # this removes the learned reliability from BOTH places it acts -> the proper
                # "uncertainty off" protocol. Robust IRLS is left on (not the learned Omega).
                om_pad = torch.eye(2, device=om_pad.device, dtype=om_pad.dtype).expand_as(om_pad).contiguous()
            mu, Sigma3 = triangulate_omega2(cons_pad, K_pad, w2c_pad, om_pad, vmask,
                                            robust=self.robust_triang, robust_iter=self.robust_iter,
                                            robust_c2=self.robust_c2, robust_kind=self.robust_kind,
                                            master_w=self.robust_master_w)

            # PHASE C — recurrent epipolar information filter
            gen = None
            if mode != "train":
                gen = torch.Generator(device="cpu"); gen.manual_seed(1)   # deterministic eval pairs
            I3 = torch.eye(3, device=device)
            if self.pose_refine:
                if self.pose_mv_evidence:
                    mu_final, pr_iters = self.pose_refiner(
                        mu, Sigma3, cons_pad, K_pad, w2c_pad, om_pad, vmask, return_iters=True)
                else:
                    mu_final, pr_iters = self.pose_refiner(mu, Sigma3, return_iters=True)
                epi = dict(mu=mu_final, Sigma3=Sigma3, Omega3=torch.linalg.inv(Sigma3 + 1e-6 * I3),
                           mu_iters=pr_iters, softw=[], beam_pz=[], gates=None, cmeas=[], vp_idx=[])
            elif self.meas == "heatmap":
                hf = run_heatmap_filter(mu, Sigma3, cons, hm_probs.float(), omega2,
                                        K_hm, w2c_all.float(), cvn, T=self.T, D=self.D,
                                        kappa=self.kappa, om_scale=float(self.cfg.get("EPI", {}).get("OM_SCALE", 0.3)),
                                        generator=gen)
                epi = {**hf, "Omega3": torch.linalg.inv(hf["Sigma3"] + 1e-6 * I3),
                       "softw": [], "beam_pz": [], "cmeas": [], "vp_idx": []}
            else:
                epi = run_epipolar_filter(
                    mu, Sigma3, cons, feat0, K_hm, w2c_all.float(), cvn, self.refiner,
                    T=self.T, D=self.D, M=self.M, kappa=self.kappa, meas=self.meas,
                    generator=gen, detach_mu_iters=self.detach_mu_iters)

        mu_final = epi["mu"]

        # ── JFORMER: deformable multi-view decoder refines mu; last stage IS mu_final ──
        jf_iters = None
        if self.use_jformer:
            from lib.models.dovf.uncertainty_modules import camera_encoding
            with torch.cuda.amp.autocast(enabled=False):
                # MONO SEED (stage-2 low-view): at N=1 the eps-solve seed is depth-garbage
                # (mu collapses toward the origin; ARCTIC 1v seed was ~0.7m off). Replace it
                # with a depth-from-scale ray seed: 2D consensus is reliable at 1 view, and
                # z ~ f * HAND_SPAN / px_span puts the hand at a plausible absolute depth
                # (+-30% ~ 5-15cm) that the decoder can refine monocularly.
                if bool(getattr(self, "jf_mono_seed", False)):
                    HAND_SPAN = 0.12                                       # metres; 0.18 3D-span / ~1.5 median foreshortening (measured)
                    off = 0
                    mu_ms = mu_final.detach().float().clone()
                    for bi in range(len(cvn)):
                        n = int(cvn[bi])
                        if n == 1:
                            k = off
                            c2 = cons[k].float()                            # (J,2) hm px
                            span = (c2.max(0).values - c2.min(0).values).max().clamp(min=4.0)
                            f = 0.5 * (K_hm[k, 0, 0] + K_hm[k, 1, 1])
                            z = (f * HAND_SPAN / span).clamp(0.2, 2.0)
                            uv1 = torch.cat([c2, torch.ones_like(c2[:, :1])], 1)   # (J,3)
                            pc = z * torch.einsum("ab,jb->ja", torch.linalg.inv(K_hm[k].float()), uv1)
                            T = extr_all[k].float()                          # cam -> master/world
                            mu_ms[bi] = torch.einsum("ab,jb->ja", T[:3, :3], pc) + T[:3, 3]
                        off += n
                    mu_final = mu_ms
                cam_enc = camera_encoding(w2c_all.float())
                jf_feats = [f.float() for f in pyramid]
                if getattr(self, "use_jf_vit", False):
                    # ViT expects 256x256 POEM-normalised crops; our input may be 384.
                    im256 = images if images.shape[-1] == 256 else \
                        F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
                    vit_pyr = self.jf_vit_backbone(image=im256.float())
                    # levels registered to the same crop -> the decoder's (w_level/w_hm) scale
                    # math holds; project to feat_dim and append the 3 finest levels.
                    jf_feats = jf_feats[:3] + [proj(v.float()) for proj, v in
                                               zip(self.jf_vit_proj, vit_pyr[:3])]
                jf_iters = self.jointformer(
                    mu_final.detach().float(), epi["Sigma3"].detach().float(),
                    jf_feats, K_hm, w2c_all.float(), cvn, cam_enc)
            # Sanitize: early-training transients (extreme mu from the eps-solve at low
            # views, W64 warmup spikes) must not propagate inf/NaN into the fit/metrics.
            mu_final = torch.nan_to_num(jf_iters[-1], nan=0.0, posinf=10.0, neginf=-10.0)
        pred_joints = mu_final

        # ── VERTEX head: deform a μ-anchored MANO template -> 778 verts ──
        mc_verts = None
        if self.mesh_on:
            mu_in = mu_final.detach() if self.mesh_detach_mu else mu_final
            with torch.cuda.amp.autocast(enabled=False):
                vfeat = None
                if self.mesh_img and desc is not None:
                    vfeat = self._sample_vertex_desc(mu_in.float(), desc.float(),
                                                     K_hm, w2c_all.float(), cvn)
                mc_verts = self.mesh_decoder(mu_in.float(), vfeat=vfeat)    # (B,V,3)
            pred_verts = mc_verts
        else:
            pred_verts = mu_final.new_zeros(mu_final.shape[0], self.num_verts, 3)
            vfeat = None

        # ── BETA HEAD: feed-forward per-scene MANO shape from pooled descriptors ──
        beta_hat = None
        if self.beta_head_on:
            with torch.cuda.amp.autocast(enabled=False):
                dv = pyramid[-1].float().mean(dim=(-2, -1))            # (BN, feat_dim)
                bh, off = [], 0
                for bi in range(len(cvn)):
                    n = int(cvn[bi])
                    bh.append(self.init_head(dv[off:off + n].mean(0, keepdim=True))[1])
                    off += n
                beta_hat = torch.cat(bh, 0)                            # (B, 10)
                if self.beta_bone_feat:
                    # additive residual from mu's bone lengths (detached: the beta loss
                    # must not push the triangulated evidence)
                    mu_d = mu_final.detach().float()                   # (B, J, 3)
                    bl = (mu_d[:, self._bone_child] - mu_d[:, self._bone_parent]).norm(dim=-1)
                    bres = self.beta_bone_mlp(bl * 10.0)               # (B, 10)
                    # bones-only: DROP the (probe-verified noisy) descriptor term
                    beta_hat = bres if self.beta_bone_only else beta_hat + bres

        # ── FIT-IN-LOOP (train only): analytical LM fit on the live fields ──
        # Gradients reach dovf_field (residuals) + chol field (Omega) through the
        # solver (backward per UNC.BACKWARD, default implicit at the LM fixed point).
        # The mu anchor is detached (it is a target, not a gradient path).
        fit_joints = None
        if self.fit_in_loop and mode == "train":
            with torch.cuda.amp.autocast(enabled=False):
                pose0, trans0 = self._kabsch_flat_init(mu_final.detach().float())
                betas0 = (beta_hat.float() if beta_hat is not None
                          else mu_final.new_zeros(mu_final.shape[0], 10, dtype=torch.float32))
                if self.fit_gt_betas and "target_mano_shape" in batch:
                    # GT-shape conditioning: per-scene GT betas (first view of each
                    # scene); constant wrt the graph, so the fit loss trains the
                    # fields/precisions around the correct bone lengths.
                    gt_b_view = batch["target_mano_shape"].view(-1, 10).float()
                    _first = torch.as_tensor(
                        np.concatenate([[0], np.cumsum(cvn)[:-1]]),
                        device=gt_b_view.device, dtype=torch.long)
                    betas0 = gt_b_view[_first]
                pose_f, trans_f, _fa = self._run_fitter_analytic_unc(
                    pose0, betas0.detach(), trans0, dovf_field.float(), cholP.float(), None,
                    K_all.float(), w2c_all.float(), cvn, device, mode,
                    mc_mu3d=mu_final.detach().float())
                # decode with the ATTACHED beta_hat: the fit loss also trains the beta
                # head through the final forward kinematics (with FIT_GT_BETAS the
                # betas are GT constants and the fit loss trains the fields only).
                fit_joints, _ = self._decode_mano(pose_f, betas0, trans_f)

        return {
            "beta_hat": beta_hat,
            "fit_joints_3d": fit_joints,
            "pred_joints_3d": pred_joints,
            "pred_verts_3d": pred_verts,
            "mc_verts": mc_verts,
            "vfeat": vfeat,
            "mu_iters": epi["mu_iters"],
            "Sigma3": epi["Sigma3"], "Omega3": epi["Omega3"],
            "gates": epi["gates"], "cmeas": epi["cmeas"], "vp_idx": epi["vp_idx"],
            "softw": epi["softw"], "beam_pz": epi["beam_pz"], "desc": desc,
            "consensus_2d": cons, "omega2": omega2,
            "hm_logits": hm_logits, "hm_probs": hm_probs, "hm_coords": hm_coords,
            "dovf_field": dovf_field, "dovf_per_scale": dovf_per_scale,
            "chol_field": cholP,
            "K_hm": K_hm, "w2c": w2c_all, "cvn": cvn,
            # placeholders the base viz/eval may look for
            "gate": None, "mu3d": mu_final, "L3d": None,
            "init_pose": None, "init_betas": None,
            "jf_iters": jf_iters,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _kabsch_flat_init(self, mu):
        """Batched Kabsch init (mirrors inference): rigid-align the flat-mean-hand
        MANO joints to mu -> global axis-angle + translation; articulation/beta = 0."""
        from pytorch3d.transforms import matrix_to_axis_angle
        from lib.utils.transform import mano_to_openpose
        B = mu.shape[0]
        with torch.no_grad():
            z48 = mu.new_zeros(B, 48); z10 = mu.new_zeros(B, 10)
            jc = mano_to_openpose(self.J_regressor,
                                  self.mano_layer(z48, z10).verts)[:, :self.num_joints]
            mc = jc.mean(1, keepdim=True); mm = mu.mean(1, keepdim=True)
            H = (jc - mc).transpose(1, 2) @ (mu - mm)
            U, S, Vh = torch.linalg.svd(H); V = Vh.transpose(1, 2)
            Z = torch.eye(3, device=mu.device, dtype=mu.dtype).repeat(B, 1, 1)
            Z[:, 2, 2] = torch.sign(torch.det(V @ U.transpose(1, 2)))
            go = matrix_to_axis_angle(V @ Z @ U.transpose(1, 2))          # (B,3)
            pose0 = torch.cat([go, mu.new_zeros(B, 45)], dim=1)           # (B,48)
            jgo = mano_to_openpose(self.J_regressor,
                                   self.mano_layer(pose0, z10).verts)[:, :self.num_joints]
            trans0 = (mu - jgo).mean(1)                                   # (B,3)
        return pose0, trans0

    # ──────────────────────────────────────────────────────────────────────────
    def _sample_vertex_desc(self, mu, desc, K_hm, w2c, cvn):
        """Per-vertex image descriptor, fused across a scene's views. Reproject the
        μ-anchored template verts V0 into each view, bilinear-sample the descriptor map,
        average over views -> (B,V,Dd). Injects per-subject shape into the mesh decoder."""
        from lib.models.dovf.descriptor import _project, _samp
        _, _, V0 = self.mesh_decoder.place_template(mu)                # (B,V,3) master-frame
        Hh, Ww = desc.shape[-2:]; Dd = desc.shape[1]; B = V0.shape[0]
        out = V0.new_zeros(B, self.num_verts, Dd)
        off = 0
        for b in range(B):
            nv = int(cvn[b]); acc = V0.new_zeros(self.num_verts, Dd)
            for vi in range(off, off + nv):
                uv = _project(V0[b], K_hm[vi], w2c[vi])               # (V,2) heatmap px
                acc = acc + _samp(desc[vi], uv, Hh, Ww)               # (V,Dd)
            out[b] = acc / max(nv, 1)
            off += nv
        return out

    def compute_loss(self, preds, batch, epoch_idx=None):
        device = preds["hm_logits"].device
        B = len(batch["cam_view_num"])
        gt_joints = batch["master_joints_3d"].view(B, self.num_joints, 3)
        scale_hm = self._hm_scale(device, preds["hm_logits"].dtype)
        gt_2d = self._project_to_views(gt_joints, batch, self.n_hm_pts) * scale_hm   # (BN,J,2)
        h, w = self.heatmap_size

        # ── 2D front-end (keep the bottleneck sharp) ──
        loss_hm = self._heatmap_loss(preds, gt_2d)
        gt_field = build_dovf_target(gt_2d, h, w)
        loss_dovf = self._dovf_l1(preds["dovf_field"], gt_field, gt_2d)
        loss_cons = self.criterion_2d(preds["consensus_2d"], gt_2d)

        # ── per-vote NLL (support-masked by the heatmap): trains each vote's precision
        #    to its own error -> makes the fused Ω2 principled ──
        delta = (preds["dovf_field"] - gt_field).permute(0, 1, 3, 4, 2)     # (BN,J,h,w,2)
        Lp = preds["chol_field"].permute(0, 1, 3, 4, 2)                    # (BN,J,h,w,3)
        nllv = gaussian_nll(delta.float(), Lp.float())                     # (BN,J,h,w)
        wsup = preds["hm_probs"].detach()
        if self.w_occ_nll > 0 and "occ_labels" in batch:
            bn_, p_ = wsup.shape[:2]
            occ = batch["occ_labels"].to(wsup.device).view(bn_, 778)
            occ_j = occ[:, self.gate_occ_j2v].long()                        # (bn,p)
            valid = occ_j < 200
            is_obj = valid & ((occ_j & 2) > 0)
            is_self = valid & ((occ_j & 1) > 0) & (~is_obj)
            ow = torch.ones(bn_, p_, device=wsup.device, dtype=wsup.dtype)
            ow = ow + self.w_occ_nll * is_obj.to(wsup.dtype)
            ow = ow + self.w_occ_nll * self.occ_nll_self * is_self.to(wsup.dtype)
            wsup = wsup * ow.view(bn_, p_, 1, 1)
        loss_vote = (wsup * nllv).sum() / (wsup.sum() + 1e-6)

        # ── 3D mean, deep-supervised across refine iters (RAFT ramp) ──
        mu_iters = preds["mu_iters"]; Tn = len(mu_iters)
        loss_mu = gt_joints.new_zeros(())
        for t, mt in enumerate(mu_iters):
            loss_mu = loss_mu + (self.ds_gamma ** (Tn - 1 - t)) * (mt - gt_joints).abs().mean()

        # ── 3D belief NLL (calibrate Σ3; −log|Ω3| punishes correlated-pair over-confidence) ──
        nll3 = (gaussian_nll_3d_from_prec(preds["pred_joints_3d"] - gt_joints,
                                          preds["Omega3"]).mean()
                if self.w_nll3 > 0 else gt_joints.new_zeros(()))

        # ── gate regularizer (discourage double-counting correlated pairs) ──
        loss_gate = preds["gates"].mean() if preds["gates"] is not None else gt_joints.new_zeros(())

        # ── bidirectional / reprojection consistency: refined μ should reproject to
        #    the measurements it was fused from (regularizes the gate/refiner) ──
        loss_bidir = gt_joints.new_zeros(())
        if preds["cmeas"] and self.w_bidir > 0:
            mu_f = preds["pred_joints_3d"].detach()      # geometry target, not the 2D head
            K_hm = preds["K_hm"]; w2c = preds["w2c"]
            n = 0
            for c_meas, ivp in zip(preds["cmeas"], preds["vp_idx"]):
                uv_pred, _ = proj_and_jac(mu_f, K_hm[ivp].float(), w2c[ivp].float())
                loss_bidir = loss_bidir + (uv_pred - c_meas).norm(dim=-1).mean()
                n += 1
            loss_bidir = loss_bidir / max(n, 1)

        # ── beam-matching CE: teach softw to localize GT depth along the epipolar beam ──
        loss_beam = gt_joints.new_zeros(())
        if self.w_beam_ce > 0 and preds.get("softw"):
            nb = 0
            for softw, Pz in zip(preds["softw"], preds["beam_pz"]):
                dist = (Pz - gt_joints[:, :, None, :]).norm(dim=-1)        # (B,J,D)
                tgt = torch.softmax(-dist.detach() / self.beam_ce_temp, dim=-1)   # soft label
                loss_beam = loss_beam - (tgt * (softw.clamp_min(1e-9)).log()).sum(-1).mean()
                nb += 1
            loss_beam = loss_beam / max(nb, 1)

        # ── AUX correspondence: MANO-supervised cross-view descriptor matching makes the
        #    backbone features 3D-consistent -> better 2D localization -> better triangulation ──
        loss_corr = gt_joints.new_zeros(())
        corr_acc = 0.0
        if self.desc_dim > 0 and self.w_corr > 0 and preds.get("desc") is not None:
            from lib.models.dovf.descriptor import correspondence_loss, compute_vertex_normals
            verts = batch["master_verts_3d"].view(B, self.num_verts, 3).float()
            K_all = batch["target_cam_intr"].view(-1, 3, 3)
            w2c_all = torch.linalg.inv(batch["target_cam_extr"].view(-1, 4, 4)).float()
            scale_hm = self._hm_scale(device, torch.float32)
            K_hm = K_all.float().clone(); K_hm[:, 0, :] *= scale_hm[0]; K_hm[:, 1, :] *= scale_hm[1]
            normals = compute_vertex_normals(verts, self._mano_faces)
            loss_corr, st = correspondence_loss(preds["desc"].float(), verts, normals, K_hm, w2c_all,
                                                batch["cam_view_num"], h, w)
            corr_acc = st["acc"]

        # ── VERTEX mesh loss: MPVPE + joints-from-verts + surface regularizers ──
        loss_verts = gt_joints.new_zeros(())
        loss_jfromv = gt_joints.new_zeros(())
        loss_edge = gt_joints.new_zeros(())
        loss_normal = gt_joints.new_zeros(())
        loss_lap = gt_joints.new_zeros(())
        if self.mesh_on and preds.get("mc_verts") is not None:
            from lib.utils.transform import mano_to_openpose
            verts = preds["mc_verts"].float()
            gtv = batch["master_verts_3d"].view(B, self.num_verts, 3).float()
            loss_verts = (verts - gtv).norm(dim=-1).mean()                    # MPVPE
            jfromv = mano_to_openpose(self.J_regressor, verts)[:, :self.num_joints]
            loss_jfromv = (jfromv - gt_joints).norm(dim=-1).mean()
            if self.w_edge > 0 or self.w_normal > 0 or self.w_lap > 0:
                e = self._mesh_edges
                if self.w_edge > 0:                                           # edge-length match to GT
                    le_p = (verts[:, e[:, 0]] - verts[:, e[:, 1]]).norm(dim=-1)
                    le_g = (gtv[:, e[:, 0]] - gtv[:, e[:, 1]]).norm(dim=-1)
                    loss_edge = (le_p - le_g).abs().mean()
                if self.w_normal > 0:                                         # per-face normal consistency
                    from lib.models.dovf.descriptor import compute_vertex_normals
                    np_ = compute_vertex_normals(verts, self._mesh_faces)
                    ng_ = compute_vertex_normals(gtv, self._mesh_faces)
                    loss_normal = (1.0 - (np_ * ng_).sum(-1)).mean()
                if self.w_lap > 0:                                            # Laplacian smoothness (vs GT)
                    V = self.num_verts
                    deg = verts.new_zeros(V).index_add_(0, e[:, 0], torch.ones_like(e[:, 0], dtype=verts.dtype))
                    deg = deg.index_add_(0, e[:, 1], torch.ones_like(e[:, 1], dtype=verts.dtype)).clamp(min=1)
                    def _lap(x):
                        acc = x.new_zeros(x.shape[0], V, 3)
                        acc.index_add_(1, e[:, 0], x[:, e[:, 1]]); acc.index_add_(1, e[:, 1], x[:, e[:, 0]])
                        return x - acc / deg[None, :, None]
                    loss_lap = (_lap(verts) - _lap(gtv)).norm(dim=-1).mean()

        loss_fit = gt_joints.new_zeros(())
        if preds.get("fit_joints_3d") is not None:
            loss_fit = (preds["fit_joints_3d"] - gt_joints).norm(dim=-1).mean()
        loss_beta = gt_joints.new_zeros(())
        if preds.get("beta_hat") is not None and "target_mano_shape" in batch:
            gt_b_view = batch["target_mano_shape"].view(-1, 10).float()      # (BN,10)
            cvn_ = batch["cam_view_num"]
            first = torch.as_tensor(np.concatenate([[0], np.cumsum(cvn_)[:-1]]),
                                    device=gt_b_view.device, dtype=torch.long)
            gt_b = gt_b_view[first]                                          # (B,10) per scene
            if self.beta_dim_scaled:
                # per-dim standardized L2: equalizes gradient across beta dims (raw L2 is
                # dominated by the few high-variance dims and regresses the rest to 0,
                # producing the observed ||beta_hat|| shrinkage). EMA of GT per-dim std.
                if self.training and gt_b.shape[0] >= 4:
                    with torch.no_grad():
                        self._beta_std.mul_(0.99).add_(0.01 * gt_b.std(0).clamp(min=1e-2))
                loss_beta = (((preds["beta_hat"] - gt_b) / self._beta_std.clamp(min=1e-2)) ** 2).mean()
            else:
                loss_beta = ((preds["beta_hat"] - gt_b) ** 2).mean()

        # JointFormer deep supervision: absolute joints at every decoder stage.
        loss_jf = gt_joints.new_zeros(())
        if preds.get("jf_iters") is not None:
            its = preds["jf_iters"]
            for i, xj in enumerate(its):
                loss_jf = loss_jf + (self.jf_gamma ** (len(its) - 1 - i)) * \
                          (xj - gt_joints).norm(dim=-1).mean()

        loss = (self.w_fit3d * loss_fit
                + self.w_jf * loss_jf
                + self.w_beta * loss_beta
                + self.w_heatmap * loss_hm
                + self.w_dovf * loss_dovf
                + self.w_cons2d * loss_cons
                + self.w_vote_nll * loss_vote
                + self.w_mu3d_ds * loss_mu
                + self.w_nll3 * nll3
                + self.w_gate * loss_gate
                + self.w_bidir * loss_bidir
                + self.w_beam_ce * loss_beam
                + self.w_corr * loss_corr)
        if self.mesh_on:
            loss = (loss + self.w_verts3d * loss_verts + self.w_jfromv * loss_jfromv
                    + self.w_edge * loss_edge + self.w_normal * loss_normal
                    + self.w_lap * loss_lap)

        # Occ-supervised XGATE: BCE(gate, 1-occluded) on valid (view,joint)s (255=unknown).
        loss_gate_occ = None
        if (self.use_xgate and self.w_gate_occ > 0 and self._last_gate is not None
                and "occ_labels" in batch):
            gpad = self._last_gate                                            # (B,Nmax,J)
            occ = batch["occ_labels"].to(gpad.device).float().view(-1, 778)   # (BN,778)
            occ_j = occ[:, self.gate_occ_j2v]                                 # (BN,J)
            cvn_g = batch["cam_view_num"]
            flat = torch.cat([gpad[i, :int(n)] for i, n in enumerate(cvn_g)], 0)  # (BN,J)
            valid = occ_j < 200
            if valid.any():
                tgt = (occ_j < 0.5).float()
                # BCE on probabilities is unsafe under AMP autocast -> force fp32 region.
                with torch.autocast(device_type="cuda", enabled=False):
                    g = flat.float().clamp(1e-4, 1 - 1e-4)
                    loss_gate_occ = F.binary_cross_entropy(g[valid], tgt[valid].float())
                loss = loss + self.w_gate_occ * loss_gate_occ

        loss_dict = {
            "loss": loss, "loss_hm": loss_hm, "loss_dovf": loss_dovf, "loss_cons": loss_cons,
            "loss_vote_nll": loss_vote, "loss_mu3d": loss_mu, "loss_j3d_nll": nll3,
            "loss_gate": loss_gate, "loss_bidir": loss_bidir, "loss_beam": loss_beam,
            "loss_corr": loss_corr, "loss_fit3d": loss_fit, "loss_beta": loss_beta,
            "loss_jf": loss_jf,
        }
        if loss_gate_occ is not None:
            loss_dict["loss_gate_occ"] = loss_gate_occ; loss_dict["loss"] = loss
        if self.mesh_on:
            loss_dict.update({"loss_verts": loss_verts, "loss_jfromv": loss_jfromv,
                              "loss_edge": loss_edge, "loss_normal": loss_normal,
                              "loss_lap": loss_lap})
            self._last_mpvpe = float(loss_verts.detach())
        self._last_corr_acc = corr_acc
        # real-time un-stick visibility in the progress bar
        self._last_beam = float(loss_beam.detach())
        self._last_gatemean = float(loss_gate.detach())
        return loss, loss_dict

    def format_metric(self, mode="val"):
        s = super().format_metric(mode)
        if mode == "train":
            s += (f" beam={getattr(self, '_last_beam', float('nan')):.2f}"
                  f" gate={getattr(self, '_last_gatemean', float('nan')):.2f}"
                  f" corr_acc={getattr(self, '_last_corr_acc', float('nan'))*100:.0f}%")
            if getattr(self, "mesh_on", False):
                s += f" mpvpe={getattr(self, '_last_mpvpe', float('nan'))*1000:.1f}mm"
        return s

    # the epi pipeline emits no chol_field viz panel data the base unc-viz expects
    def _maybe_viz(self, batch, preds, mode, step_idx):
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Param groups: refiner + log_tau get the unc LR group; rest unchanged.
    # ──────────────────────────────────────────────────────────────────────────
    def get_param_groups(self, train_cfg):
        base_lr = float(train_cfg.LR)
        bb_scale = float(train_cfg.get("BACKBONE_LR_SCALE", 1.0))
        unc_lr = base_lr * self.unc_lr_scale
        bb, unc_d, unc_nd, other = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("img_backbone."):
                bb.append(p)
            elif n.startswith("refiner.") or n == "log_tau" or n.startswith("unc_head."):
                (unc_nd if (p.ndim <= 1 or "norm" in n.lower()) else unc_d).append(p)
            else:
                other.append(p)
        groups = [{"params": other, "lr": base_lr, "name": "heads"}]
        if unc_d:
            groups.append({"params": unc_d, "lr": unc_lr, "name": "epi"})
        if unc_nd:
            groups.append({"params": unc_nd, "lr": unc_lr, "weight_decay": 0.0, "name": "epi_nodecay"})
        if bb:
            groups.append({"params": bb, "lr": base_lr * bb_scale, "name": "backbone"})
        return groups
