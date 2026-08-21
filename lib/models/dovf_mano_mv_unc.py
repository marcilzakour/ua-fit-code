"""
DOVFManoMVUnc — uncertainty-aware variant of :class:`DOVFManoMV`.
=================================================================

New, registered model class (does NOT touch the base model). Adds:

  1. ``UncertaintyFieldHead``  -> per-(view,joint) anisotropic 2x2 precision Ω,
     sampled from a Cholesky FIELD co-located with the DOVF field.
  2. ``CrossViewGate``         -> per-(view,joint) consensus gate (epistemic trust)
     from a masked cross-view transformer with a relative-camera encoding.
  3. ``analytic_fit_unc``      -> block-weighted GN solve (A = ΣJᵀΩJ, g = ΣJᵀΩr),
     a strict generalization of the scalar fitter.
  4. A Gaussian-NLL field loss that trains the uncertainty head (no labels needed).

Everything reduces to the base model when Ω = w·I and gate = 1. Backbone / neck /
heatmap / DOVF / init heads and all 2D losses are inherited unchanged, so a
Stage-A (2D-pretrained) base checkpoint loads via INIT_FROM (the new modules init
fresh). Config: add a ``UNC`` block and ``LOSS.UNC_NLL_WEIGHT``; set
``MODEL.TYPE: DOVFManoMVUnc``.
"""

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint

from lib.utils.builder import MODEL
from lib.utils.logger import logger
from lib.models.dovf_mano_mv import DOVFManoMV
from lib.models.dovf.analytic_fitter_unc import analytic_fit_unc, gaussian_nll, _sample_field, chol_to_prec
from lib.models.dovf.uncertainty_modules import (
    UncertaintyFieldHead, CrossViewGate, CrossViewRefiner, sample_tokens, camera_encoding,
    chol3_to_prec, gaussian_nll_3d,
)
from lib.models.dovf.triangulate_resample import MixtureMCHead, CostVolumeHead
from lib.models.dovf.mesh_decoder import MeshDecoder, JointsPoseInit
from lib.utils.transform import mano_to_openpose
from lib.models.dovf import build_dovf_target, dovf_consensus_2d, dovf_vote_cov


@MODEL.register_module()
class DOVFManoMVUnc(DOVFManoMV):
    """Uncertainty- and consensus-aware DOVF MANO estimator."""

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.backend == "theseus" and self.fitter == "analytic", \
            "DOVFManoMVUnc requires BACKEND=theseus and FITTER=analytic"
        assert self.heatmap_target == self.dovf_target == "joints", \
            "DOVFManoMVUnc requires HEATMAP_TARGET=DOVF_TARGET=joints (consensus tokens)"

        unc_cfg = cfg.get("UNC", {})
        self.use_gate = bool(unc_cfg.get("XVIEW_GATE", True))
        # REFINE=true -> the cross-view transformer ALSO emits a bounded 2D residual
        # correction (DOVF refinement) so its capacity is usable, not just a scalar gate.
        self.use_refine = bool(unc_cfg.get("REFINE", False))
        # Backward through the unc fitter: "unroll" (robust default) | "implicit".
        self.unc_backward = unc_cfg.get("BACKWARD", "unroll")
        self.lm_damping = float(unc_cfg.get("LM_DAMPING", 1e-3))
        # ABLATION (paper tab:abl "- anisotropic Omega -> scalar"): force the
        # per-pixel precision isotropic at the field level (l21=0, l22 tied to
        # l11 in RAW space; chol2_to_prec maps raw1 linearly so raw1=0 -> l21=0).
        # One intervention point covers every consumer (NLL, consensus,
        # triangulation, solver 2D term).
        self.iso_omega = bool(unc_cfg.get("ISO_OMEGA", False))

        self.unc_head = UncertaintyFieldHead(self.feat_dim, self.n_dovf_pts)
        if self.use_gate:
            xv_kw = dict(d=int(unc_cfg.get("XVIEW_DIM", 128)),
                         heads=int(unc_cfg.get("XVIEW_HEADS", 4)),
                         layers=int(unc_cfg.get("XVIEW_LAYERS", 2)), cam_dim=6)
            if self.use_refine:
                self.xview_gate = CrossViewRefiner(
                    self.feat_dim, corr_max=float(unc_cfg.get("CORR_MAX", 6.0)),
                    predict_trans=bool(unc_cfg.get("PREDICT_TRANS", False)),
                    dtrans_max=float(unc_cfg.get("DTRANS_MAX", 0.10)),
                    predict_j3d=bool(unc_cfg.get("PREDICT_J3D", False)),
                    dmu_max=float(unc_cfg.get("DMU_MAX", 0.20)), **xv_kw)
            else:
                self.xview_gate = CrossViewGate(self.feat_dim, **xv_kw)

        # LR for the new modules (esp. the cross-view transformer): a separate
        # param group, scaled off base LR. Default 1.0 (no change) but the gate
        # transformer typically prefers ~0.3-0.5 of the conv-head LR.
        self.unc_lr_scale = float(unc_cfg.get("LR_SCALE", 1.0))

        loss_cfg = getattr(cfg, "LOSS", None) or {}
        self.w_unc_nll = float(loss_cfg.get("UNC_NLL_WEIGHT", 1.0))
        self.w_root = float(loss_cfg.get("TRANS_LOSS_WEIGHT", 10.0))   # direct sup. of learned root
        self.w_nll3 = float(loss_cfg.get("J3D_NLL_WEIGHT", 0.1))       # NLL calibrates covariance only
        self.w_mu3d = float(loss_cfg.get("J3D_MEAN_WEIGHT", 10.0))     # L2 trains the 3D mean (accurate)
        # SCORER-REVIVAL supervision (Phase-1): the scorer collapsed to one-hot on the center
        # (mu_hat=mu, inert) despite the best sigma-point being 3-5mm closer. SCORE_CE_WEIGHT adds
        # CE(w, argmin-dist sigma-point) to TEACH it to pick; MUHAT_ANCHOR_WEIGHT scales the
        # ||mu_hat-GT|| term that rewards mu_hat=mu (set 0 to let the scorer move off-center).
        self.w_mc_ce = float(loss_cfg.get("SCORE_CE_WEIGHT", 0.0))
        self.w_mc_anchor = float(loss_cfg.get("MUHAT_ANCHOR_WEIGHT", 1.0))
        unc_cfg = cfg.get("UNC", {})
        # 3D-prior weight in the GN: rescales metric (m) Ω3 into the heatmap-px range
        # of the 2D term so the prior ASSISTS rather than dominates the solve.
        self.j3d_gn_scale = float(unc_cfg.get("J3D_GN_SCALE", 0.005))
        # Predictor-only: skip the GN solver and output mu3d directly (val MPJPE then
        # = the learned 3D head's accuracy). Pure 3D-regression test, ~2x faster.
        self.predictor_only = bool(unc_cfg.get("PREDICTOR_ONLY", False))
        # PAIR-MINING (train efficiency): load K views/scene, run the backbone ONCE per image,
        # then form 2-view PAIRS from the K views and train the cheap mc_head on each pair
        # (amortizes the per-image backbone, profiled ~2.6-4.8x; see profile_amortize.py).
        # PAIR_MAX caps pairs/scene (0=all). PAIR_COND_MIN drops near-degenerate pairs (small
        # baseline angle, rad) — important for Interhand's high-variance/two-hand view pairs.
        self.pair_mining = bool(unc_cfg.get("PAIR_MINING", False))
        self.pair_max = int(unc_cfg.get("PAIR_MAX", 0))
        self.pair_cond_min = float(unc_cfg.get("PAIR_COND_MIN", 0.0))
        # Monte-Carlo head: DOVF uplift -> Gaussian -> particles -> resample -> posterior.
        self.mc_head_on = bool(unc_cfg.get("MC_HEAD", False))
        if self.mc_head_on:
            use_costvol = unc_cfg.get("COST_VOLUME", False)
            HeadCls = CostVolumeHead if use_costvol else MixtureMCHead
            extra = {} if use_costvol else {"consistency": bool(unc_cfg.get("MC_CONSISTENCY", False))}
            self.mc_head = HeadCls(
                self.feat_dim, d=int(unc_cfg.get("XVIEW_DIM", 256)),
                heads=int(unc_cfg.get("XVIEW_HEADS", 8)),
                layers=int(unc_cfg.get("XVIEW_LAYERS", 3)),
                KS=int(unc_cfg.get("MC_PARTICLES", 48)), num_joints=self.num_joints, **extra)
            self.mc_sigma0 = float(unc_cfg.get("MC_SIGMA0", 2.0))    # 2D-accuracy floor (hm-px)
            # Omega2 reads ~1.44px vs true ~1.85px -> scale precision by (1.44/1.85)^2 to match
            self.mc_omega_scale = float(unc_cfg.get("MC_OMEGA_SCALE", 0.6))
            # model-based mode: feed mu_hat as a 3D prior into the MANO fitter -> mesh
            self.mc_prior_fit = bool(unc_cfg.get("MC_PRIOR_FIT", False))
            # LEARNED pose prior: anchor the solver to init_head's predicted pose (theta0)
            # with a learnable per-DoF precision (the init_head becomes a compact data prior;
            # the solver resolves 2v-underdetermined DoF toward the plausible learned pose).
            self.learned_prior_on = bool(unc_cfg.get("MC_LEARNED_PRIOR", False))
            if self.learned_prior_on:
                # per-DoF log-precision on the 45 finger axis-angle DoF (init = current λ)
                init_lp = float(np.log(max(self.fit_prior, 1e-3)))
                self.learned_pose_logprec = torch.nn.Parameter(torch.full((45,), init_lp))
            # mu_hat-conditioned pose warm-start (basin hardening): residual on init_head.
            self.mc_pose_init_on = bool(unc_cfg.get("MC_POSE_INIT", False))
            if self.mc_pose_init_on:
                self.joints_pose_init = JointsPoseInit(num_joints=self.num_joints,
                                                       d=int(unc_cfg.get("POSE_INIT_DIM", 256)))
            # FREE-vertex mesh head (no theta): deform template around mu_hat -> 778 verts
            self.mc_mesh_on = bool(unc_cfg.get("MC_MESH", False))
            if self.mc_mesh_on:
                with torch.no_grad():
                    _o = self.mano_layer(torch.zeros(1, 48), torch.zeros(1, 10))
                    _Jt = mano_to_openpose(self.J_regressor, _o.verts)[0, :self.num_joints]
                    _w = _Jt[self.center_idx]
                    Vt_rel = (_o.verts[0] - _w).contiguous()                    # root-relative
                    Jt_rel = (_Jt - _w).contiguous()
                self.mesh_decoder = MeshDecoder(
                    Vt_rel, Jt_rel, num_joints=self.num_joints, d=int(unc_cfg.get("MESH_DIM", 256)),
                    heads=int(unc_cfg.get("MESH_HEADS", 8)),
                    layers=int(unc_cfg.get("MESH_LAYERS", 4)), center_idx=self.center_idx)
                self.w_verts3d = float(unc_cfg.get("MESH_VERTS_WEIGHT", 10.0))
            else:
                self.mc_mesh_on = False
        self._last_unc_viz_epoch = -1            # val-viz cadence for the uncertainty panel

        # In 2D-pretrain the fitter never runs; freeze the new modules so DDP keeps
        # find_unused=false (mirrors the base model's freezing policy).
        if self.pretrain_2d:
            mods = [self.unc_head] + ([self.xview_gate] if self.use_gate else [])
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(False)

        # OCC-SUPERVISED GATE: BCE(gate, 1-occluded) from the dense occ_annotations/v2
        # labels (batch["occ_labels"], (BN,778) uint8 bitmask, 255=unknown). Gives the gate
        # the EPISTEMIC occlusion signal directly — measured (docs/occlusion_2d_ceiling
        # _findings.md): the aleatoric Omega2 under-flags self-occ ("confident but wrong").
        # 778 verts -> 21 joints via nearest vertex in mean MANO pose. Default 0 = off.
        self.w_gate_occ = float(loss_cfg.get("GATE_OCC_WEIGHT", 0.0))
        # OCC-WEIGHTED NLL: upweight the heteroscedastic DOVF-field NLL on occluded
        # (view,joint)s so the learned Omega actually widens where the votes are
        # unreliable. Phase-1 diagnosis (docs/dexycb_oakink_gap_analysis.md): Omega is
        # ~2x OVERCONFIDENT under object occlusion (2D err 3.2px vs learned sigma 1.6px on
        # DexYCB) because occluded votes are a minority the average NLL under-serves.
        # obj-occluded (& "both") get the full weight; self-occluded get OCC_NLL_SELF x it.
        self.w_occ_nll = float(loss_cfg.get("OCC_NLL_WEIGHT", 0.0))
        self.occ_nll_self = float(loss_cfg.get("OCC_NLL_SELF", 0.5))
        if self.w_gate_occ > 0 or self.w_occ_nll > 0:
            with torch.no_grad():
                _v0 = self.mano_layer(torch.zeros(1, 48), torch.zeros(1, 10)).verts
                _j0 = mano_to_openpose(self.J_regressor, _v0)[:, :self.num_joints]
                _j2v = torch.cdist(_j0[0], _v0[0]).argmin(1)
            self.register_buffer("gate_occ_j2v", _j2v, persistent=False)

        cprint(f"[{self.name}] uncertainty fitter | gate={self.use_gate} refine={self.use_refine} "
               f"backward={self.unc_backward} nll_w={self.w_unc_nll} gate_occ_w={self.w_gate_occ}", "cyan")

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation: base DOVF panels + an uncertainty panel (covariance ellipses
    # at the projected joints + the cross-view gate heatmap) -> TensorBoard.
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_viz(self, batch, preds, mode, step_idx):
        super()._maybe_viz(batch, preds, mode, step_idx)        # base DOVF/multiview panels
        if not self.viz_enabled or preds.get("chol_field") is None:
            return
        summ = getattr(self, "summary", None)
        if summ is None or getattr(summ, "rank", 0) != 0 or not hasattr(summ, "add_figure"):
            return
        if mode == "train":
            if self.viz_interval <= 0 or (step_idx % self.viz_interval) != 0:
                return
        else:  # val: once per epoch (own tracker, independent of the base panel's)
            if self._last_unc_viz_epoch == self._cur_epoch:
                return
            self._last_unc_viz_epoch = self._cur_epoch
        try:
            from lib.utils.dovf_unc_viz import build_uncertainty_panel
            fig = build_uncertainty_panel(self, batch, preds, mode=mode)
            if fig is not None:
                summ.add_figure(f"viz/{mode}_uncertainty", fig, global_step=step_idx)
        except Exception as e:
            logger.warning(f"[unc-viz] panel render failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Forward (copy of base _forward_impl with the uncertainty/gate path inserted)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_pairs(self, cons, feat, K_hm, w2c, om, cvn, gt_all):
        """Form 2-view pairs from each scene's K views and GATHER the per-image tensors into a
        pair layout (no recompute). Returns gathered (cons,feat,K_hm,w2c,om), cvn_p=[2]*P and
        mc_gt (P,J,3). Optional baseline-angle filter (PAIR_COND_MIN) drops near-degenerate
        pairs — key for Interhand's high-variance/two-hand view pairs."""
        dev = cons.device
        C = -torch.einsum("bji,bj->bi", w2c[:, :3, :3], w2c[:, :3, 3])        # cam centres (BN,3)
        h = gt_all.mean(1)                                                    # scene hand centroids (B,3)
        flat, scene = [], []
        off = 0
        for i, n in enumerate(cvn):
            n = int(n); vids = list(range(off, off + n))
            cand = [(a, b) for a in range(n) for b in range(a + 1, n)]
            if self.pair_cond_min > 0.0:                                       # drop small-baseline pairs
                kept = []
                for a, b in cand:
                    va = C[vids[a]] - h[i]; vb = C[vids[b]] - h[i]
                    cosang = (va @ vb) / (va.norm() * vb.norm() + 1e-6)
                    if torch.arccos(cosang.clamp(-1, 1)) >= self.pair_cond_min:
                        kept.append((a, b))
                cand = kept or cand[:1]                                        # never empty
            if self.pair_max and len(cand) > self.pair_max:
                cand = [cand[j] for j in torch.randperm(len(cand), device="cpu")[:self.pair_max].tolist()]
            for a, b in cand:
                flat += [vids[a], vids[b]]; scene.append(i)
            off += n
        flat = torch.as_tensor(flat, device=dev, dtype=torch.long)
        scene = torch.as_tensor(scene, device=dev, dtype=torch.long)
        cvn_p = [2] * len(scene)
        return (cons[flat], feat[flat], K_hm[flat], w2c[flat], om[flat], cvn_p, gt_all[scene])

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
        chol_field = self.unc_head(pyramid[0])                          # (BN, J, 3, h, w)
        if self.iso_omega:   # scalar-Omega ablation: isotropic precision field
            _l = 0.5 * (chol_field[:, :, 0] + chol_field[:, :, 2])
            chol_field = torch.stack([_l, torch.zeros_like(_l), _l], dim=2)
        consensus_2d, consensus_conf = dovf_consensus_2d(hm_probs, dovf_field)
        desc_view = pyramid[-1].mean(dim=(-2, -1))

        hm_h, hm_w = self.heatmap_size
        scale_hm2img = torch.tensor([self.image_size[1] / hm_w,
                                     self.image_size[0] / hm_h], device=device)

        B = len(cvn)
        opt_active = self._optimizer_active(mode, epoch_idx)

        init_pose_list, init_betas_list, trans0_list = [], [], []
        for i in range(B):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
            desc = desc_view[s:e].mean(0, keepdim=True)
            pose0, betas0 = self.init_head(desc)
            init_pose_list.append(pose0); init_betas_list.append(betas0)
            trans0_list.append(self._init_translation(
                consensus_2d, consensus_conf, hm_coords, dovf_field,
                K_all, w2c_all, extr_all, s, e, N, scale_hm2img, device))
        init_pose = torch.cat(init_pose_list, 0)
        init_betas = torch.cat(init_betas_list, 0)
        trans0 = torch.cat(trans0_list, 0)

        # consensus-2D feature tokens (BN, J, feat_dim) for the cross-view gate.
        tok_feats = sample_tokens(pyramid[0], consensus_2d, hm_h, hm_w) \
            if self.use_gate else None

        grad_ctx = nullcontext() if opt_active else torch.no_grad()
        # MC head runs FIRST so its mu_hat can prime the MANO fitter as a 3D prior.
        mc_X = mc_w = mu_hat = None
        mc_gt = mc_mu_pairs = None
        if self.mc_head_on:
            with grad_ctx, torch.cuda.amp.autocast(enabled=False):   # fp32: linalg (cholesky/eigh)
                sx, sy = self._hm_scale(device, torch.float32)
                K_hm = K_all.float().clone()
                K_hm[:, 0, :] = K_hm[:, 0, :] * sx; K_hm[:, 1, :] = K_hm[:, 1, :] * sy
                # Learned anisotropic 2D precision Omega2 (sampled at the consensus)
                # -> uncertainty-weighted GN triangulation. Omega2 is NLL-trained to
                # the actual 2D error (~1.44px vs true ~1.85px), so inflate slightly.
                L2 = _sample_field(consensus_2d.unsqueeze(1),
                                   chol_field.permute(0, 3, 4, 1, 2).unsqueeze(1),
                                   hm_h, hm_w, 3).squeeze(1)               # (BN,J,3) chol
                omega2 = chol_to_prec(L2) * self.mc_omega_scale            # (BN,J,2,2) precision
                # SCENE reconstruction (uses each scene's K views) -> mu_hat (B,J,3): drives
                # pred_joints / val / train-metric. DETACH geometry: loss_mc trains only the
                # scorer (via resampled features), never corrupting the 2D perception.
                mu_hat, mc_X, mc_w, _ = self.mc_head(
                    consensus_2d.float().detach(), pyramid[0].float(), K_hm,
                    w2c_all.float(), cvn, omega2=omega2.float().detach())
                # PAIR-MINING: mine 2-view pairs from the K views and train the cheap mc_head
                # per pair (backbone amortized once over the K images). The pair outputs feed
                # loss_mc (mc_X/mc_w/mc_gt); the scene mu_hat above stays for pred_joints/val.
                if self.pair_mining and mode == "train":
                    gt_all = batch["master_joints_3d"].view(B, self.num_joints, 3).float()
                    cons_p, feat_p, Khm_p, w2c_p, om_p, cvn_p, mc_gt = self._build_pairs(
                        consensus_2d.float(), pyramid[0].float(), K_hm,
                        w2c_all.float(), omega2.float(), cvn, gt_all)
                    mc_mu_pairs, mc_X, mc_w, _ = self.mc_head(
                        cons_p.detach(), feat_p, Khm_p, w2c_p, cvn_p, omega2=om_p.detach())
        # Basin hardening: warm-start the solver pose toward a mu_hat-consistent pose
        # (residual on init_head) so it always lands in the same correct basin.
        if getattr(self, "mc_pose_init_on", False) and mu_hat is not None:
            with grad_ctx, torch.cuda.amp.autocast(enabled=False):
                init_pose = init_pose + self.joints_pose_init(mu_hat.detach().float(), self.center_idx)
        with grad_ctx:
            pose_opt, trans_opt, fit_aux = self._run_fitter_analytic_unc(
                init_pose, init_betas, trans0, dovf_field, chol_field, tok_feats,
                K_all, w2c_all, cvn, device, mode,
                mc_mu3d=(mu_hat if (self.mc_head_on and self.mc_prior_fit) else None))
        if self.mc_head_on:
            fit_aux["mu3d"] = mu_hat; fit_aux["L3d"] = None
        if self.predictor_only:
            pred_joints = fit_aux["mu3d"]                       # learned 3D head is the output
            pred_verts = pred_joints.new_zeros(pred_joints.shape[0], self.num_verts, 3)
        else:
            pred_joints, pred_verts = self._decode_mano(pose_opt, init_betas, trans_opt)
        # FREE-vertex mesh head: deform template around (detached) mu_hat -> 778 verts;
        # joints read back via J_regressor (POEM-comparable, no MANO theta).
        mc_verts = None
        if getattr(self, "mc_mesh_on", False) and mu_hat is not None:
            with grad_ctx, torch.cuda.amp.autocast(enabled=False):   # fp32: svd (Kabsch)
                mc_verts = self.mesh_decoder(mu_hat.detach().float())
            pred_verts = mc_verts
            pred_joints = mano_to_openpose(self.J_regressor, mc_verts)[:, :self.num_joints]
        gate = fit_aux["gate"]

        return {
            "pred_root": fit_aux["pred_root"],  # learned 3D root estimate (or None)
            "mu3d": fit_aux["mu3d"],            # learned per-joint 3D mean (or None)
            "L3d": fit_aux["L3d"],              # learned per-joint 3x3 chol (or None)
            "mc_X": mc_X, "mc_w": mc_w,         # MC particles + posterior weights (pair layout if mining)
            "mc_gt": mc_gt,                     # per-pair GT (P,J,3) when pair-mining, else None
            "mc_mu_pairs": mc_mu_pairs,         # per-pair readout mean for the anchor term
            "mc_verts": mc_verts,               # free-vertex mesh (or None)
            "pred_joints_3d": pred_joints,
            "pred_verts_3d": pred_verts,
            "hm_logits": hm_logits, "hm_probs": hm_probs, "hm_coords": hm_coords,
            "dovf_field": dovf_field, "dovf_per_scale": dovf_per_scale,
            "chol_field": chol_field,
            "gate": gate,                       # (B,Nmax,J) consensus gate or None — for viz
            "consensus_2d": consensus_2d,
            "init_pose": init_pose, "init_betas": init_betas,
            "init_joints_rel": (self._mano_init_joints_rel(init_pose, init_betas)
                                if self.w_init_joints > 0 else None),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Block-weighted analytic fit (pads to N_max, builds gate, calls analytic_fit_unc)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_fitter_analytic_unc(self, pose0, betas, trans0, dovf_field, chol_field, tok_feats,
                                 K_all, w2c_all, cvn, device, mode="train", mc_mu3d=None):
        B = len(cvn)
        Nmax = int(max(cvn))
        hm_h, hm_w = self.heatmap_size
        J = self.num_joints
        sx, sy = self._hm_scale(device, K_all.dtype)

        K_hm_all = K_all.clone()
        K_hm_all[:, 0, :] = K_hm_all[:, 0, :] * sx
        K_hm_all[:, 1, :] = K_hm_all[:, 1, :] * sy
        cam_enc_all = camera_encoding(w2c_all)                         # (BN, 6)

        K_pad = K_hm_all.new_zeros(B, Nmax, 3, 3)
        w2c_pad = w2c_all.new_zeros(B, Nmax, 4, 4)
        w2c_pad[..., :, :] = torch.eye(4, device=device, dtype=w2c_all.dtype)
        dovf_pad = dovf_field.new_zeros(B, Nmax, hm_h, hm_w, J, 2)
        chol_pad = chol_field.new_zeros(B, Nmax, hm_h, hm_w, J, 3)
        vmask = torch.zeros(B, Nmax, device=device, dtype=dovf_field.dtype)
        cam_pad = cam_enc_all.new_zeros(B, Nmax, 6)
        tok_pad = (tok_feats.new_zeros(B, Nmax, J, tok_feats.shape[-1])
                   if self.use_gate else None)
        for i in range(B):
            s = int(np.sum(cvn[:i])); e = int(np.sum(cvn[:i + 1])); N = e - s
            K_pad[i, :N] = K_hm_all[s:e]
            w2c_pad[i, :N] = w2c_all[s:e]
            dovf_pad[i, :N] = dovf_field[s:e].permute(0, 3, 4, 1, 2)   # (N,J,2,h,w)->(N,h,w,J,2)
            chol_pad[i, :N] = chol_field[s:e].permute(0, 3, 4, 1, 2)   # (N,J,3,h,w)->(N,h,w,J,3)
            vmask[i, :N] = 1.0
            cam_pad[i, :N] = cam_enc_all[s:e]
            if self.use_gate:
                tok_pad[i, :N] = tok_feats[s:e]

        # Ablation switch (eval only): force the learned anisotropic precision to
        # Ω = I and the gate to 1, so the fitter reduces EXACTLY to the scalar
        # baseline (Ω = w·I, gate = 1) on the SAME weights — isolates what the
        # uncertainty add-on contributes at inference with zero training confound.
        unc_off = getattr(self, "unc_eval_off", False)

        gate = None
        corr = None
        pred_root = None
        mu3d = None
        L3d = None
        if self.use_gate and not unc_off:
            # gate is a content head — keep it in fp32 with the solve.
            if self.use_refine:
                gate, corr, aux = self.xview_gate(tok_pad.float(), cam_pad.float(), vmask.float())
                if aux.get("dtrans") is not None:
                    # learned 3D root-translation correction -> init + anchor target.
                    trans0 = trans0 + aux["dtrans"].to(trans0.dtype)
                    pred_root = trans0
                if aux.get("dmu3d") is not None:
                    # learned per-joint 3D mean = init MANO joints + bounded correction;
                    # paired with a learned 3x3 precision -> anisotropic 3D prior.
                    jw_init = self._decode_mano(pose0, betas, trans0)[0]      # (B,J,3) world
                    mu3d = jw_init + aux["dmu3d"].to(jw_init.dtype)
                    L3d = aux["L3d"]
            else:
                gate = self.xview_gate(tok_pad.float(), cam_pad.float(), vmask.float())  # (B,Nmax,J)

        if unc_off:
            # chol params [c,0,c] -> chol_to_prec gives identity precision, since
            # softplus(c)+1e-3 = 1  =>  c = log(exp(1-1e-3) - 1).
            import math
            c = math.log(math.exp(1.0 - 1e-3) - 1.0)
            chol_pad = torch.full_like(chol_pad, c)
            chol_pad[..., 1] = 0.0

        # Override with the MC head's mu_hat as the 3D prior (model-based mode):
        # fit MANO to the DOVF 2D field AND anchor to mu_hat (isotropic precision).
        if mc_mu3d is not None:
            mu3d = mc_mu3d
            L3d = None                      # -> scalar isotropic omega3 below

        if self.predictor_only:
            # skip the solver entirely — the learned 3D head IS the output here.
            return pose0, trans0, {"gate": gate, "pred_root": pred_root, "mu3d": mu3d, "L3d": L3d}

        # Couple the 3D prior into the fit only when its GN weight > 0. With scale=0
        # the head still trains (L2+NLL) but the fit is the undisturbed baseline
        # (and may use fast implicit backward) — isolates "is mu3d accurate?".
        use_prior = (mu3d is not None) and (self.j3d_gn_scale > 0)
        with torch.cuda.amp.autocast(enabled=False):
            if not use_prior:
                omega3 = None
            elif L3d is not None:
                omega3 = chol3_to_prec(L3d.float()) * self.j3d_gn_scale
            else:                            # isotropic prior from mu_hat
                B_, J_ = mu3d.shape[:2]
                omega3 = (torch.eye(3, device=device).expand(B_, J_, 3, 3)
                          * self.j3d_gn_scale)
                # Eval-only per-joint anchor-weight (e.g. cross-source consensus): if
                # `mc_omega_scale` (B,J) is set, scale the isotropic 3D-anchor precision
                # per joint -> trust the anchor where sources agree, defer to the 2D data
                # term where they disagree. None = uniform (default). No training effect.
                _sc = getattr(self, "mc_omega_scale", None)
                if _sc is not None:
                    omega3 = omega3 * _sc.to(omega3.dtype).view(B_, J_, 1, 1)
            # LEARNED pose prior: anchor fingers to init_head's pose (per-sample ref =
            # theta0) with a learnable per-DoF precision -> compact data prior.
            if getattr(self, "learned_prior_on", False):
                Pdim = 51
                p_prec = torch.zeros(Pdim, Pdim, device=device, dtype=torch.float32)
                p_prec[3:48, 3:48] = torch.diag(self.learned_pose_logprec.float().exp())
                p_ref = torch.cat([pose0.float(), trans0.float()], dim=-1)   # (B,51) init pose
                p_anchor = torch.zeros(Pdim, device=device, dtype=torch.float32)
                p_anchor[0:3] = 0.05; p_anchor[48:51] = 0.05                 # global orient+trans
            else:
                p_prec = getattr(self, "prior_prec", None)
                p_ref = getattr(self, "prior_ref", None)
                p_anchor = getattr(self, "prior_anchor", None)
            pose_opt, trans_opt = analytic_fit_unc(
                self.mano_layer, self.J_regressor,
                pose0.float(), trans0.float(), betas.float(),
                K_pad.float(), w2c_pad.float(), dovf_pad.float(), chol_pad.float(),
                gate=(gate.float() if gate is not None else None),
                corr=(corr.float() if corr is not None else None),
                mu3d=(mu3d.float() if use_prior else None),
                omega3=omega3,
                num_joints=J, H=hm_h, W=hm_w,
                max_iterations=self.fit_iters, step_size=self.fit_step,
                pose_prior_weight=self.fit_prior, log_radius=self.fit_log_radius.float(),
                lm_damping=self.lm_damping, view_weight=vmask.float(),
                jac_mode=(self.jac_mode_train if mode == "train" else self.jac_mode_eval),
                center_idx=self.center_idx, backward=self.unc_backward,
                prior_prec=p_prec, prior_ref=p_ref, prior_anchor=p_anchor,
            )
        return pose_opt, trans_opt, {"gate": gate, "pred_root": pred_root, "mu3d": mu3d, "L3d": L3d}

    # ──────────────────────────────────────────────────────────────────────────
    # Loss: base losses + Gaussian-NLL on the DOVF field error (trains uncertainty)
    # ──────────────────────────────────────────────────────────────────────────

    def _occ_nll_weight(self, batch, bn, p, device, dtype):
        """(bn,p,1,1) per-(view,joint) NLL multiplier: 1 for visible/unknown,
        (1+w) for object-occluded ("both" too), (1+w*self_frac) for self-occluded.
        Returns None if occ supervision is off or labels are absent."""
        if self.w_occ_nll <= 0 or "occ_labels" not in batch:
            return None
        occ = batch["occ_labels"].to(device).view(bn, 778)
        occ_j = occ[:, self.gate_occ_j2v].long()                     # (bn,p) 0-3, 255=unknown
        valid = occ_j < 200
        is_obj = valid & (((occ_j & 2) > 0))                          # object-occluded (2 or 3)
        is_self = valid & ((occ_j & 1) > 0) & (~is_obj)               # self-occluded only (1)
        w = torch.ones(bn, p, device=device, dtype=dtype)
        w = w + self.w_occ_nll * is_obj.to(dtype)
        w = w + self.w_occ_nll * self.occ_nll_self * is_self.to(dtype)
        return w.view(bn, p, 1, 1)

    def _dovf_nll(self, preds, batch):
        """Heteroscedastic NLL of the DOVF field error under the predicted 2x2 Ω,
        Gaussian-weighted near the GT joint (where the votes/consensus live)."""
        device = preds["dovf_field"].device
        scale_hm = self._hm_scale(device, preds["dovf_field"].dtype)
        _, gt_dovf_2d = self._gt_2d_hm(batch, scale_hm)               # (BN, J, 2) heatmap px
        h, w = self.heatmap_size
        gt_field = build_dovf_target(gt_dovf_2d, h, w)               # (BN, J, 2, h, w)
        pred = preds["dovf_field"]                                  # (BN, J, 2, h, w)
        chol = preds["chol_field"]                                 # (BN, J, 3, h, w)
        delta = (pred - gt_field).permute(0, 1, 3, 4, 2)           # (BN,J,h,w,2)
        Lp = chol.permute(0, 1, 3, 4, 2)                           # (BN,J,h,w,3)
        nll = gaussian_nll(delta, Lp)                              # (BN,J,h,w)

        # proximity weighting (same Gaussian as the base DOVF L1 when sigma>0)
        bn, p, _ = gt_dovf_2d.shape
        occ_w = self._occ_nll_weight(batch, bn, p, device, pred.dtype)   # (bn,p,1,1) or None
        s = self.dovf_loss_sigma
        if s > 0:
            xs = torch.linspace(0, w - 1, w, device=device, dtype=pred.dtype)
            ys = torch.linspace(0, h - 1, h, device=device, dtype=pred.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            cx = gt_dovf_2d[..., 0].view(bn, p, 1, 1)
            cy = gt_dovf_2d[..., 1].view(bn, p, 1, 1)
            wgt = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))   # (bn,p,h,w)
            if occ_w is not None:
                wgt = wgt * occ_w
            return (wgt * nll).sum() / (wgt.sum() + 1e-6)
        if occ_w is not None:
            return (occ_w * nll).sum() / (occ_w.expand_as(nll).sum() + 1e-6)
        return nll.mean()

    def compute_loss(self, preds, batch, epoch_idx=None):
        loss, loss_dict = super().compute_loss(preds, batch, epoch_idx=epoch_idx)
        if self.w_unc_nll > 0 and preds.get("chol_field") is not None:
            nll = self._dovf_nll(preds, batch)
            loss = loss + self.w_unc_nll * nll
            loss_dict = {**loss_dict, "loss_unc_nll": nll, "loss": loss}
        # Direct supervision of the learned 3D root (gives strong gradient that the
        # weak through-fit signal can't — root = wrist world pos = trans).
        if self.w_root > 0 and preds.get("pred_root") is not None:
            B = preds["pred_root"].shape[0]
            gt_root = batch["master_joints_3d"].view(B, self.num_joints, 3)[:, self.center_idx]
            l_root = (preds["pred_root"] - gt_root).norm(dim=-1).mean()
            loss = loss + self.w_root * l_root
            loss_dict = {**loss_dict, "loss_root": l_root, "loss": loss}
        # Occ-supervised gate: teach the cross-view gate to down-weight occluded (view,joint)s.
        if self.w_gate_occ > 0 and preds.get("gate") is not None and "occ_labels" in batch:
            gate = preds["gate"]                                             # (B, Nmax, J) in (0,1]
            occ = batch["occ_labels"].to(gate.device).float().view(-1, 778)  # (BN, 778)
            occ_j = occ[:, self.gate_occ_j2v]                                # (BN, J) 0-3, 255=unknown
            cvn = batch["cam_view_num"]
            flat = torch.cat([gate[i, :int(n)] for i, n in enumerate(cvn)], 0)  # (BN, J)
            valid = occ_j < 200
            if valid.any():
                tgt = (occ_j < 0.5).float()                                  # visible -> trust 1
                # BCE on probabilities is unsafe under AMP autocast -> force fp32 region.
                with torch.autocast(device_type="cuda", enabled=False):
                    g = flat.float().clamp(1e-4, 1 - 1e-4)
                    l_go = F.binary_cross_entropy(g[valid], tgt[valid].float())
                loss = loss + self.w_gate_occ * l_go
                loss_dict = {**loss_dict, "loss_gate_occ": l_go, "loss": loss}
        if preds.get("mc_X") is not None:
            # MC posterior loss (no collapse): expected sq-error under the weights
            # trains the weights to put mass on near-GT sigma-points; + a direct L2
            # anchor on the readout mean for stability (keeps it from drifting).
            # Pair-mining: mc_X/mc_w/mu are per-pair (P), with per-pair GT mc_gt.
            if preds.get("mc_gt") is not None:
                gt_j = preds["mc_gt"]                                     # (P,J,3)
                mu_anchor = preds["mc_mu_pairs"]                          # (P,J,3)
            else:
                B = preds["mc_X"].shape[0]
                gt_j = batch["master_joints_3d"].view(B, self.num_joints, 3)
                mu_anchor = preds["mu3d"]
            d2 = ((preds["mc_X"] - gt_j[:, :, None, :]) ** 2).sum(-1)     # (P,J,KS)
            loss_mc = (preds["mc_w"] * d2).sum(-1).mean()
            l_muhat = (mu_anchor - gt_j).norm(dim=-1).mean()             # readout anchor
            loss = loss + self.w_mu3d * (loss_mc + self.w_mc_anchor * l_muhat)
            loss_dict = {**loss_dict, "loss_mc": loss_mc, "loss_muhat": l_muhat, "loss": loss}
            if self.w_mc_ce > 0:                                          # TEACH the scorer to PICK
                best_k = d2.argmin(dim=-1)                                # (P,J) closest sigma-pt
                logw = preds["mc_w"].clamp(min=1e-9).log()
                loss_ce = -logw.gather(-1, best_k.unsqueeze(-1)).squeeze(-1).mean()
                loss = loss + self.w_mc_ce * loss_ce
                loss_dict = {**loss_dict, "loss_mc_ce": loss_ce, "loss": loss}
        if preds.get("mc_verts") is not None:
            # FREE-vertex mesh loss: MPVPE + joints-from-verts MPJPE (trains mesh_decoder).
            B = preds["mc_verts"].shape[0]
            gtv = batch["master_verts_3d"].view(B, self.num_verts, 3)
            gtj = batch["master_joints_3d"].view(B, self.num_joints, 3)
            l_verts = (preds["mc_verts"] - gtv).norm(dim=-1).mean()
            l_jfromv = (preds["pred_joints_3d"] - gtj).norm(dim=-1).mean()
            loss = loss + self.w_verts3d * (l_verts + l_jfromv)
            loss_dict = {**loss_dict, "loss_verts": l_verts, "loss_jfromv": l_jfromv, "loss": loss}
        if preds.get("mc_X") is None and preds.get("L3d") is not None and preds.get("mu3d") is not None:
            # Per-joint 3D Gaussian, DECOUPLED: L2 trains mean, NLL on detached
            # residual calibrates covariance.
            B = preds["mu3d"].shape[0]
            gt_j = batch["master_joints_3d"].view(B, self.num_joints, 3)
            delta = preds["mu3d"] - gt_j
            l_mu = delta.norm(dim=-1).mean()
            nll3 = gaussian_nll_3d(delta.detach(), preds["L3d"]).mean()
            loss = loss + self.w_mu3d * l_mu + self.w_nll3 * nll3
            loss_dict = {**loss_dict, "loss_mu3d": l_mu, "loss_j3d_nll": nll3, "loss": loss}
        return loss, loss_dict

    # ──────────────────────────────────────────────────────────────────────────
    # Param groups: separate LR for the new modules; no weight decay on the
    # transformer's norms/biases. The base groups (backbone, heads) are unchanged.
    # ──────────────────────────────────────────────────────────────────────────

    def get_param_groups(self, train_cfg):
        base_lr = float(train_cfg.LR)
        bb_scale = float(train_cfg.get("BACKBONE_LR_SCALE", 1.0))
        unc_lr = base_lr * self.unc_lr_scale
        bb, unc_decay, unc_nodecay, other = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("img_backbone."):
                bb.append(p)
            elif (n.startswith("unc_head.") or n.startswith("xview_gate.")
                  or n.startswith("mc_head.") or n.startswith("mesh_decoder.")
                  or n.startswith("joints_pose_init.")):
                # no weight decay on norms / biases / 1-D params (standard for transformers)
                if p.ndim <= 1 or "norm" in n.lower():
                    unc_nodecay.append(p)
                else:
                    unc_decay.append(p)
            else:
                other.append(p)
        groups = [{"params": other, "lr": base_lr, "name": "heads"}]
        if unc_decay:
            groups.append({"params": unc_decay, "lr": unc_lr, "name": "unc"})
        if unc_nodecay:
            groups.append({"params": unc_nodecay, "lr": unc_lr, "weight_decay": 0.0, "name": "unc_nodecay"})
        if bb:
            groups.append({"params": bb, "lr": base_lr * bb_scale, "name": "backbone"})
        return groups
