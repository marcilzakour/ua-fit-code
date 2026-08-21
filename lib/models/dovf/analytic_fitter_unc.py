"""
Uncertainty-aware analytic MANO fitter (block-weighted Gauss-Newton).
=====================================================================

Strict generalization of :mod:`lib.models.dovf.analytic_fitter`: the per-residual
SCALAR IRLS weight ``w`` (``W = diag(w)``) is replaced by a per-(view,joint) 2x2
PRECISION matrix ``Ω`` (block-diagonal ``W``), plus an optional per-(view,joint)
scalar ``gate`` (cross-view consensus trust). The residual is kept as
(view,joint) 2-vectors instead of being flattened to scalar rows, so the normal
equations become

    A = Σ_{v,j} Jᵀ Ω J + λI + μI ,   g = Σ_{v,j} Jᵀ Ω r + prior_grad

This reduces EXACTLY to ``analytic_fit`` when ``Ω = w·I₂`` and ``gate = 1``.

``Ω`` is sampled from a Cholesky FIELD ``chol`` co-located with the DOVF field
(same H×W grid), evaluated at the current projection ``uv`` each iteration — so
the precision is location-aware (it tracks the moving projection) and anisotropic.
The Jacobian chain, damping, step rule, prior, and backward modes are identical to
the scalar fitter; only the weighting changes.

Differentiable wrt: dovf, chol (uncertainty field), gate, pose0, trans0, betas,
log_radius. ``backward="unroll"`` (default, robust) or ``"implicit"`` (O(1) mem).
"""

from typing import Optional
import torch
import torch.nn.functional as F

from lib.utils.transform import mano_to_openpose

# Eval-only stabilization knob (None = off). When set, _build_system_unc clamps the
# 2x2 precision Ω's condition number (eigenvalue ratio) to <= this value. Used to
# test whether bounding the learned uncertainty's anisotropy removes solver
# divergence on hard (e.g. 2-view) samples. No effect on training (left at None).
EVAL_OMEGA_COND_MAX = None

# Eval-only cost-trajectory recorder (off by default → zero training/inference cost).
# When RECORD_TRAJ is True, _gn_loop_unc appends the batch-mean total cost at init and
# after every LM iteration to TRAJ (list of floats). Used by the solver-ablation
# harness to measure iterations-to-convergence. Reset TRAJ before each fit.
RECORD_TRAJ = False
TRAJ = []

# Eval-only pose-trajectory recorder (off by default -> zero training/inference cost).
# When RECORD_POSE_TRAJ is True, _gn_loop_unc appends (pose,trans) at init and after every
# LM iteration to POSE_TRAJ (list of (pose[:,:48], trans[:,48:]) CPU tensors). Used by the
# T x LM sweep to read the per-inner-iteration state = the LM axis at T=1 from ONE run.
# Reset POSE_TRAJ before each fit. Bit-identical to the default path when off.
RECORD_POSE_TRAJ = False
POSE_TRAJ = []

# Eval-only probe (paper tab:solver / Jacobian-agreement claim): when RECORD_JAC is
# set, every analytic-mode linearization ALSO computes the exact autograd articulated
# Jacobian dJc at the SAME point and records (cosine, rel-Frobenius-err, step-dir cos)
# between the analytic and exact Jacobian. Measures the claim directly, decoupled from
# any downstream mesh metric. Reset JAC_AGREE before each probe. See scripts/probe_jac_agreement.py.
RECORD_JAC = False
JAC_AGREE = []

# Eval-only probe (Gate B, Fisher-information study): when RECORD_FISHER is set, the
# GN loop stashes the CONVERGED normal-equation matrices so the Cramer-Rao prediction
# tr(G_k A^-1 G_k^T) can be compared against the measured per-joint error.
#   A_data = sum_m J^T Omega J  over the 2D observation blocks ONLY (the Fisher
#            information of the 2D evidence; no LM damping, no prior, no 3D anchor).
#   A_full = A_data + the 3D-anchor / barrier blocks actually used by the estimator.
#   dJc    = d(centered joints)/d(pose) at the converged theta -> G_k = [dJc_k | I3].
# Nothing else changes; the fit is bit-identical. Reset FISHER_OUT before each probe.
RECORD_FISHER = False
FISHER_OUT = []

# Eval-only solver-mode switch for the GN-vs-LM ablation (default "lm" = production).
#   "lm"    : adaptive Levenberg-Marquardt (per-sample trust region: accept-on-decrease,
#             λ down on accept / up on reject) — the shipped solver.
#   "fixed" : damped Gauss-Newton, FIXED λ=lm_damping, full steps, no accept/reject.
#   "pure"  : pure Gauss-Newton (λ→1e-8), full steps, no accept/reject.
# Isolates the value of the trust-region machinery. No effect on training (left "lm").
GN_MODE = "lm"

# Eval-only anatomical-rotation regularizer (manotorch anatomy limits). When
# ANATOMY_W is a float, the LM energy gains an inference-time term
#   E_anat = 0.5 * ANATOMY_W * Σ_axis relu(|ee_axis| - limit)^2
# over the 15 finger/thumb joints' anatomy-aligned euler angles (twist/spread/bend),
# extracted from MANO's global transforms via manotorch `AxisLayerFK`. It is a
# one-sided barrier: exactly zero inside the anatomical box (active set), so it only
# pushes back on implausible rotations produced by weak/few-view 2D evidence, and
# leaves valid poses untouched. Detached GN block (no learned params, no grad to any
# input) → composes with the LM trust region and the implicit backward's A. Default
# None = off (zero overhead). See scripts/ablate_solver.py "+anat" variant.
ANATOMY_W = None
# Eval-only: when set to an int k, apply the anatomy barrier ONLY in the last k LM
# iterations (the data term converges first, then the barrier refines rotations into
# the human range). Cuts cost ~max_iters/k with little accuracy loss. None = all iters.
ANATOMY_LAST_K = None
_ANAT_CACHE = {}

# Env override so ANY eval entry point (eval.py, eval_perdataset.py, DDP test) picks up
# the barrier without code edits: POEM_ANATOMY_W=50 [POEM_ANATOMY_LAST_K=5].
import os as _os
if _os.environ.get("POEM_ANATOMY_W"):
    ANATOMY_W = float(_os.environ["POEM_ANATOMY_W"])
if _os.environ.get("POEM_ANATOMY_LAST_K"):
    ANATOMY_LAST_K = int(_os.environ["POEM_ANATOMY_LAST_K"])

# Eval-only TEMPORAL-COHERENCE anchor (streaming reconstruction). When set to a
# tuple (target (B,num_joints,3), weight float) OR A LIST of such tuples (e.g. one for the
# PREVIOUS frame's pose and one for the NEXT frame's pose, for non-causal multi-frame
# smoothing), the LM energy gains an inference-time joint-space term per tuple
#   E_temp = 0.5 * weight * Σ_j || jw_j(θ) - target_j ||^2
# anchoring the current frame's fitted joints toward one or more references, so a
# per-frame solver produces temporally smooth output with ZERO retraining. Structurally
# identical to the μ̂ 3D-prior block (J3=[dJc|I3]), and merged INTO that block (`j3`) so
# it rides the existing accept/reject + normal-eqs plumbing. `target` is in world metres
# (same units as mu3d). None = off (zero cost). Set/cleared per frame by the streaming
# harness (single-threaded eval). See scripts/eval_temporal_energy.py.
TEMPORAL_ANCHOR = None
# Eval-only last-k schedule for TEMPORAL_ANCHOR (mirrors ANATOMY_LAST_K): None (default)
# applies the temporal anchor from iteration 0 (the ORIGINAL always-on behavior, unchanged
# for any existing caller). Set to an int k to apply it ONLY in the final k iterations
# instead — e.g. let the data/vertex term converge first, then pull toward neighbouring
# frames' poses to smooth out per-frame noise without fighting the data term early on.
TEMPORAL_ANCHOR_LAST_K = None

# Eval-only CONSECUTIVE-FRAME batch coupling (multi-frame joint solve). When the solver is
# called with a batch of B rows that are CONSECUTIVE frames in temporal order (caller's
# responsibility), setting TEMPORAL_NEIGHBOR_W (isotropic weight, float) adds
#   E_neighbor = 0.5 * w * Σ_i [ ||jw_i(θ) - jw_{i-1}(θ)||^2 + ||jw_i(θ) - jw_{i+1}(θ)||^2 ]
# (edge rows get only one side). The neighbour target is `jw` ITSELF (this same _build()
# call's freshly-computed joints for the adjacent batch row), so it is always the LATEST
# estimate as of the current GN iteration -- no external/stale target tensor needed, unlike
# TEMPORAL_ANCHOR. The cross-row Jacobian is treated as zero (target detached), so this
# couples rows only through the ITERATE, converging over iterations like a block
# Gauss-Seidel step embedded in one joint LM loop -- cheap (reuses dJc, no new autograd).
# Gated by the SAME `temporal_on`/TEMPORAL_ANCHOR_LAST_K last-k schedule as TEMPORAL_ANCHOR
# (both are "temporal smoothing", conventionally applied after the data term converges).
# None = off (zero cost, zero effect on any existing caller — including single-frame,
# B=1 calls, where there IS no neighbour and the block is a no-op regardless).
TEMPORAL_NEIGHBOR_W = None

# Eval-only per-view 2D REPROJECTION energy (external 2D evidence, e.g. WiLoR keypoints).
# When set to (target_uv (B,V,J,2) in HEATMAP px, weight), the LM energy gains
#   E_2D = 0.5 * weight * Σ_{v,j} vmask_v * || π(jw)_{v,j} - target_{v,j} ||^2
# i.e. project the fit's 3D joints into every view and match an external per-view 2D
# prediction (bundle-adjustment style). Reuses the projection Jacobian duv_dXw already
# computed for the DOVF data term (dr_duv = I here), so it is cheap. Unlike the
# triangulated-3D anchor, this injects EACH view's 2D directly, so it stays informative
# at few views / under occlusion where triangulation is under-constrained. target_uv must
# be padded-by-scene to match the solver's internal view order. None = off.
WILOR_2D = None

# Eval-only Huber/IRLS robustification for WILOR_2D (heatmap-px radius; None = off, EXACT
# original plain-MSE behavior). Mirrors the DOVF data term's own Huber IRLS scheme
# (`s = 1 if rn<=delta else sqrt(delta/rn)`, applied to the precision) but keyed off a fixed
# radius rather than a learnable one -- this term has no training signal to learn it from at
# eval time. Motivation: with a FIXED per-(view,joint) weight, one view whose 2D keypoint is
# roughly-correctly-located-but-still-substantially-wrong (e.g. WiLoR misjudging a
# self-occluded finger) gets FULL unclipped quadratic weight and can measurably distort the
# joint solve; downweighting by residual magnitude lets the OTHER, agreeing views dominate
# instead. None = off (zero cost, zero effect on any existing caller).
WILOR_2D_HUBER = None

# Eval-only per-view 2D VERTEX reprojection energy — the dense analogue of WILOR_2D. WiLoR's
# own MANO decode predicts ALL 778 vertices (not just the 21 joints) from the SAME forward
# pass, in the SAME vertex ORDER/topology as this solver's own `mano_layer` (both wrap the
# same MANO model) — so vertex i's WiLoR-predicted projection is a direct correspondence
# target for vertex i's fitted projection, no nearest-neighbour/silhouette matching needed.
# When set to (target_uv (B,V,Nv,2) HEATMAP px, weight[, vert_idx (LongTensor or None=all
# 778)[, om2d]]) OR a list of such tuples, the LM energy gains
#   E_2Dvert = 0.5 * weight * Σ_{v,i} vmask_v * || π(v_w)_{v,i} - target_{v,i} ||^2
# The vertex Jacobian ∂v/∂pose is computed via one batched vjp through the MANO decode
# (see `_verts_pose_jac`) — reuses the SAME generic `_joints_pose_jac`/`_project_and_jac`
# machinery the 21-joint term uses, since neither assumes anything about the point count.
# Cost scales with len(vert_idx) (778 by default) — markedly heavier per-iteration than the
# 21-joint WILOR_2D term. None = off (zero cost, zero effect on any existing caller).
WILOR_2D_VERTS = None
# Huber/IRLS radius for WILOR_2D_VERTS -- same mechanism/motivation as WILOR_2D_HUBER above,
# heatmap-px. None = off (exact original plain-MSE behavior).
WILOR_2D_VERTS_HUBER = None

# Eval-only OBJECT-PENETRATION energy (hand-object no-penetration). When set to a
# tuple (push_fn, weight), the LM energy gains an inference-time term
#   E_sdf = 0.5 * weight * Σ_{i ∈ penetrating verts} depth_i^2
# where push_fn maps the fit's world MANO vertices (B,Nv,3) -> an OUTWARD push vector
# (B,Nv,3) that is ZERO for vertices outside every object and equals (margin - sdf)·n̂
# for penetrating vertices (n̂ = world ∇SDF, sdf negative-inside). The GN residual is the
# 1-D penetration depth along the surface normal, r_i = -depth_i, with Jacobian
# n̂_iᵀ ∂v_i/∂θ (autograd through MANO FK for pose, identity for trans). Detached GN block
# (like the anatomy barrier) so it composes with the LM trust region + implicit backward.
# push_fn is a closure supplied by the caller over the object SDF grids + per-frame object
# poses expressed IN THE FIT (master-camera) FRAME — this module stays decoupled from any
# particular SDF backend. None = off. See src/scripts/uafit_backend.py.
OBJECT_SDF = None
# Cap on the number of deepest-penetrating vertices used per scene, to bound the
# per-iteration autograd (batched vjp) cost. Env: POEM_OBJECT_SDF_MAX_VERTS.
OBJECT_SDF_MAX_VERTS = int(_os.environ.get("POEM_OBJECT_SDF_MAX_VERTS", "256"))
# Optional last-k schedule (mirrors ANATOMY_LAST_K): apply the (expensive, autograd)
# object-SDF block only in the final k LM iterations — the data term converges first,
# then a couple of iters push out of objects. Cuts the per-fit autograd cost by
# ~max_iters/k with little accuracy loss. None = every iteration. Env: POEM_OBJECT_SDF_LAST_K.
OBJECT_SDF_LAST_K = (int(_os.environ["POEM_OBJECT_SDF_LAST_K"])
                     if _os.environ.get("POEM_OBJECT_SDF_LAST_K") else None)

# Reuse the verified Jacobian math from the scalar fitter (single source of truth).
from lib.models.dovf.analytic_fitter import (
    _mano_joints,
    mano_kinematic_jac,
    _joints_pose_jac,
    _project_and_jac,
    _sample_dovf_and_grad,
    _ancestor_mask,
    _so3_right_jacobian,
)


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical-rotation regularizer (inference-time energy)
# ──────────────────────────────────────────────────────────────────────────────

def _get_anat(mano_layer, device):
    """Lazily build + cache (AxisLayerFK, AnatomyConstraintLossEE) for this MANO.

    AxisLayerFK maps global joint transforms -> anatomy-aligned relative euler angles;
    AnatomyConstraintLossEE holds the per-joint twist/spread/bend limits. Keyed by
    (side, assets_root, device) so a mixed-side / multi-GPU run stays correct."""
    side = getattr(mano_layer, "side", "right")
    root = getattr(mano_layer, "mano_assets_root", "assets/mano_v1_2")
    key = (side, root, str(device))
    # tolerance tuning: new_tol = tol*ANATOMY_TOL_SCALE + ANATOMY_TOL_ADD (deg). ADD loosens the
    # strict twist/spread=0° limits (real poses have small nonzero twist/spread); SCALE widens the
    # bend/spread maxima. Env: POEM_ANATOMY_TOL_SCALE, POEM_ANATOMY_TOL_ADD. Cached per (scale,add).
    tsc = float(_os.environ.get("POEM_ANATOMY_TOL_SCALE", "1.0"))
    tad = float(_os.environ.get("POEM_ANATOMY_TOL_ADD", "0.0"))
    key = (side, root, str(device), tsc, tad)
    if key not in _ANAT_CACHE:
        from manotorch.axislayer import AxisLayerFK
        from manotorch.anatomy_loss import AnatomyConstraintLossEE
        fk = AxisLayerFK(side=side, mano_assets_root=root).to(device)
        al = AnatomyConstraintLossEE(reduction="none")
        if tsc == 1.0 and tad == 0.0:
            al.setup()
        else:
            def _sc(s):                                   # scale/add each "sign:deg" token
                return ",".join(f"{p.split(':')[0]}:{float(p.split(':')[1])*tsc+tad:g}"
                                for p in s.split(","))
            _d = dict(thumb_cmc=["+-:45", "+:45,-:15", "+:45,-:0"], thumb_mcp=["+-:0", "+-:10", "+:90,-:0"],
                      thumb_pip=["+-:0", "+-:0", "+:90,-:0"], finger_mcp=["+-:0", "+-:5", "+:90,-:0"],
                      finger_pip=["+-:0", "+-:0", "+:90,-:0"], finger_dip=["+-:0", "+-:0", "+:90,-:0"])
            al.setup(**{k: [_sc(x) for x in v] for k, v in _d.items()})
        _ANAT_CACHE[key] = (fk, al)
    return _ANAT_CACHE[key]


def _anat_residual(ee, al):
    """Per-axis anatomical hinge residuals from anatomy euler angles ee (B,16,3).

    Returns (B,45): r = relu(|ee_axis| - limit) for the (twist,spread,bend) of the 15
    articulated joints (global joint 0 excluded). Quadratic barrier E = ½‖r‖² is
    smoother than the module's L1 form and better conditioned for Gauss-Newton."""
    mcp = [1, 4, 10, 7]; pip = [2, 5, 11, 8]; dip = [3, 6, 12, 9]

    def joint_res(ee_j, cfg):                                     # ee_j (B,N,3) -> (B,N,3)
        return torch.stack([al._cal_loss_one_axis(ee_j[:, :, k], cfg[k]) for k in range(3)], dim=-1)

    parts = [joint_res(ee[:, mcp], al.finger_mcp),
             joint_res(ee[:, pip], al.finger_pip),
             joint_res(ee[:, dip], al.finger_dip),
             joint_res(ee[:, 13:14], al.thumb_cmc),
             joint_res(ee[:, 14:15], al.thumb_mcp),
             joint_res(ee[:, 15:16], al.thumb_pip)]
    return torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=-1)   # (B,45)


def _anatomy_block(mano_layer, pose, betas, weight):
    """Detached GN block for E_anat = ½·weight·‖relu(|ee|-limit)‖².

    Returns (A (B,51,51), g (B,51), cost (B,)). The residual r (weight folded in via
    √weight) and its Jacobian J = ∂r/∂pose (autograd through MANO FK -> AxisLayerFK,
    batched vjp like `_joints_pose_jac`) are detached, so this contributes the constant
    PD block JᵀJ / Jᵀr to the normal equations (trans DOFs untouched).

    EXACT active-axis masking: relu residuals are 0 inside the anatomical box, and a
    zero row adds nothing to JᵀJ / Jᵀr, so we only backprop the axes violating in ANY
    sample — cutting the vjp from 45 backward passes to |active| with identical output."""
    B = pose.shape[0]
    fk, al = _get_anat(mano_layer, pose.device)
    sw = float(weight) ** 0.5
    A = pose.new_zeros(B, 51, 51)
    g = pose.new_zeros(B, 51)
    with torch.enable_grad():
        pl = pose.detach().requires_grad_(True)
        out = mano_layer(pl, betas.detach())
        _, _, ee = fk(out.transforms_abs)                        # (B,16,3) anatomy euler
        r = _anat_residual(ee, al) * sw                          # (B,K)
        cost = 0.5 * (r.detach() ** 2).sum(-1)                   # (B,) full residual (inactive=0)
        idx = (r.detach().abs() > 0).any(dim=0).nonzero(as_tuple=False).squeeze(-1)  # active axes
        Ka = int(idx.numel())
        if Ka == 0:                                              # fully inside anatomical box
            return A, g, cost
        ra = r[:, idx]                                           # (B,Ka) keep graph
        eye = torch.eye(Ka, device=pose.device, dtype=r.dtype).unsqueeze(1).expand(Ka, B, Ka)
        (grad,) = torch.autograd.grad(ra, pl, grad_outputs=eye, is_grads_batched=True,
                                      retain_graph=False, create_graph=False)
    J = grad.permute(1, 0, 2).detach()                           # (B,Ka,48)
    ra = ra.detach()
    A[:, :48, :48] = J.transpose(1, 2) @ J                       # JᵀJ over active rows
    g[:, :48] = torch.einsum("bkp,bk->bp", J, ra)               # Jᵀr over active rows
    return A, g, cost


_SDF_DOMJOINT_CACHE = {}


def _object_sdf_block(mano_layer, pose, trans, betas, push_fn, weight, max_verts,
                      anc_mask, center_idx):
    """Detached GN block for the hand-object no-penetration energy
        E_sdf = ½·weight·Σ_i depth_i²   (i over penetrating MANO vertices),
    with an ANALYTIC vertex Jacobian (no autograd) — ~100x faster than the vjp version.

    Each penetrating vertex rides its DOMINANT skinning joint's kinematic chain, so its
    Jacobian is the SAME cross-product articulated Jacobian mano_kinematic_jac uses for
    joints/fingertips:  ∂v_i/∂θ_{j,m} = A[dom_i, j] · (axis_{j,m} × (v_i − p_j)), centered
    at center_idx (axis = Rg·Jr with the SO(3) right-Jacobian correction). The residual is
    the 1-D penetration depth along the surface normal, r_i = −depth_i, and the trans block
    is the identity projected onto n̂. Only the `max_verts` deepest penetrating verts per
    scene are used. Returns (A (B,51,51), g (B,51), cost (B,)).

    push_fn(verts (B,Nv,3)) -> push (B,Nv,3): OUTWARD displacement, zero for verts outside
    every object. Both verts and the object geometry live in the FIT frame.
    """
    B = pose.shape[0]
    sw = float(weight) ** 0.5
    A = pose.new_zeros(B, 51, 51)
    g = pose.new_zeros(B, 51)
    cost = pose.new_zeros(B)
    key = id(mano_layer)
    if key not in _SDF_DOMJOINT_CACHE:                              # dominant skinning joint per vertex
        _SDF_DOMJOINT_CACHE[key] = mano_layer.th_weights.argmax(-1).to(pose.device)   # (778,)
    dom = _SDF_DOMJOINT_CACHE[key]
    anc_mask = anc_mask.to(pose.device)
    with torch.no_grad():
        out = mano_layer(pose, betas)                              # forward only (no autograd)
        Tt = out.transforms_abs                                    # (B,16,4,4) uncentered
        Rg = Tt[:, :, :3, :3]; pg = Tt[:, :, :3, 3]               # (B,16,3,3),(B,16,3)
        theta = pose.reshape(B, 16, 3)
        axes = Rg @ _so3_right_jacobian(theta)                     # (B,16,3,3) [j,xyz,m]
        verts_c = out.verts                                        # (B,778,3) centered@center_idx
        world_v = verts_c + trans.unsqueeze(1)                     # (B,778,3) fit frame
        v_unc = verts_c + pg[:, center_idx:center_idx + 1]         # uncentered position (matches pg)
        push = push_fn(world_v)                                    # (B,778,3) outward, 0 outside
        depth = push.norm(dim=-1)                                  # (B,778)
        pen = depth > 1e-9
        if not bool(pen.any()):
            return A, g, cost
        nhat = push / depth.clamp(min=1e-9).unsqueeze(-1)         # (B,778,3) world normal
        rows_b, rows_i = [], []
        for b in range(B):
            idx = pen[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > max_verts:
                idx = idx[torch.topk(depth[b, idx], k=max_verts).indices]
            rows_b.append(torch.full((idx.numel(),), b, device=pose.device, dtype=torch.long))
            rows_i.append(idx)
        rows_b = torch.cat(rows_b); rows_i = torch.cat(rows_i)     # (N,)
        N = int(rows_b.numel())
        axes_r = axes[rows_b]                                      # (N,16,3,3) [j,xyz,m]
        pg_r = pg[rows_b]                                          # (N,16,3)

        def _pointjac(point, Amask):                              # point (N,3) uncentered -> (N,3,48)
            diff = point.unsqueeze(1) - pg_r                       # (N,16,3)
            d_e = diff.unsqueeze(-1).expand(N, 16, 3, 3)          # broadcast over m
            cr = torch.cross(axes_r, d_e, dim=2) * Amask.view(N, 16, 1, 1)   # (N,16,3,3)
            return cr.permute(0, 2, 1, 3).reshape(N, 3, 48)       # (N,3,48)  index = j*3+m

        Arow = anc_mask[dom[rows_i]]                               # (N,16) dominant-joint ancestors
        Acen = anc_mask[center_idx].view(1, 16).expand(N, 16)
        Jpose = _pointjac(v_unc[rows_b, rows_i], Arow) - _pointjac(pg_r[:, center_idx], Acen)  # centered
        nhat_r = nhat[rows_b, rows_i]                              # (N,3)
        Jn_pose = torch.einsum("nc,ncp->np", nhat_r, Jpose)       # (N,48)  n̂ᵀ ∂v/∂pose
        J = torch.cat([Jn_pose, nhat_r], dim=-1) * sw             # (N,51)  trans block = n̂
        r = -(sw * depth[rows_b, rows_i])                         # (N,)
        cost = cost.index_add(0, rows_b, 0.5 * (sw * depth[rows_b, rows_i]) ** 2)
        A.index_add_(0, rows_b, J.unsqueeze(-1) * J.unsqueeze(-2))
        g.index_add_(0, rows_b, J * r.unsqueeze(-1))
    return A, g, cost


def _object_sdf_block_autograd(mano_layer, pose, trans, betas, push_fn, weight, max_verts):
    """Reference (autograd) version of the SDF block, kept for validating the analytic one.
    Slow: one MANO backward per penetrating vertex via is_grads_batched."""
    B = pose.shape[0]
    sw = float(weight) ** 0.5
    A = pose.new_zeros(B, 51, 51)
    g = pose.new_zeros(B, 51)
    cost = pose.new_zeros(B)
    with torch.enable_grad():
        pl = pose.detach().requires_grad_(True)
        out = mano_layer(pl, betas.detach())
        verts_pose = out.verts + trans.detach().unsqueeze(1)     # (B,Nv,3) grad only wrt pose
        push = push_fn(verts_pose.detach())                      # (B,Nv,3) outward, 0 outside
        depth = push.norm(dim=-1)                                # (B,Nv) penetration depth
        pen = depth > 1e-9                                       # (B,Nv) penetrating mask
        if not bool(pen.any()):
            return A, g, cost
        nhat = push / depth.clamp(min=1e-9).unsqueeze(-1)        # (B,Nv,3) world normal, detached
        nhat = nhat.detach()
        # per-scene: select up to max_verts deepest penetrating verts, build a flat
        # (row -> (scene b, vert i)) list so the vjp is a single batched backward.
        rows_b, rows_i = [], []
        for b in range(B):
            idx = pen[b].nonzero(as_tuple=False).squeeze(-1)     # penetrating vert ids
            if idx.numel() == 0:
                continue
            if idx.numel() > max_verts:
                topk = torch.topk(depth[b, idx], k=max_verts).indices
                idx = idx[topk]
            rows_b.append(torch.full((idx.numel(),), b, device=pose.device, dtype=torch.long))
            rows_i.append(idx)
        rows_b = torch.cat(rows_b); rows_i = torch.cat(rows_i)   # (R,)
        R = int(rows_b.numel())
        # scalar s_r = n̂_rᵀ v_{b,i}(pose)  -> ∂s_r/∂pose is the pose Jacobian row.
        s = (nhat[rows_b, rows_i] * verts_pose[rows_b, rows_i]).sum(-1)   # (R,) keep graph
        cost = cost.index_add(0, rows_b, 0.5 * (sw * depth[rows_b, rows_i].detach()) ** 2)
        # s is 1-D (R,); is_grads_batched wants grad_outputs (R, *s.shape) = (R, R)
        eye = torch.eye(R, device=pose.device, dtype=s.dtype)
        (grad,) = torch.autograd.grad(s, pl, grad_outputs=eye, is_grads_batched=True,
                                      retain_graph=False, create_graph=False)
        # grad (R,B,48): row r's grad is nonzero only in scene rows_b[r]; gather it.
    Jpose = grad[torch.arange(R, device=pose.device), rows_b].detach() * sw   # (R,48)
    Jtrans = nhat[rows_b, rows_i].detach() * sw                              # (R,3)  ∂s/∂trans = n̂
    J = torch.cat([Jpose, Jtrans], dim=-1)                                    # (R,51)
    r = -(sw * depth[rows_b, rows_i].detach())                               # (R,) = -√w·depth
    # scatter the per-row outer products / gradients into the per-scene A, g.
    JJ = J.unsqueeze(-1) * J.unsqueeze(-2)                                    # (R,51,51)
    Jr = J * r.unsqueeze(-1)                                                  # (R,51)
    A.index_add_(0, rows_b, JJ)
    g.index_add_(0, rows_b, Jr)
    return A, g, cost


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty field: Cholesky params -> 2x2 SPD precision (information form)
# ──────────────────────────────────────────────────────────────────────────────

def chol_to_prec(L):
    """(...,3) raw Cholesky params -> (...,2,2) SPD precision Ω = M Mᵀ.

    M = [[l11, 0], [l21, l22]] lower-triangular, l11,l22 > 0 via softplus.
    Information form (larger Ω = more confident); no matrix inverse needed.
    """
    eps = 1e-3
    l11 = F.softplus(L[..., 0]) + eps
    l21 = L[..., 1]
    l22 = F.softplus(L[..., 2]) + eps
    o11 = l11 * l11
    o12 = l11 * l21
    o22 = l21 * l21 + l22 * l22
    row0 = torch.stack([o11, o12], dim=-1)
    row1 = torch.stack([o12, o22], dim=-1)
    return torch.stack([row0, row1], dim=-2)            # (...,2,2)


def gaussian_nll(delta, L):
    """Per-element Gaussian NLL ½ δᵀΩδ − ½ log|Ω| with Ω = M Mᵀ (chol params L).

    delta (...,2) residual (pred − gt), L (...,3) raw chol params. Used to TRAIN
    the uncertainty field: it learns large covariance where the field errs, and is
    kept from collapsing by the −½log|Ω| term. Returns (...,) NLL.
    """
    eps = 1e-3
    l11 = F.softplus(L[..., 0]) + eps
    l21 = L[..., 1]
    l22 = F.softplus(L[..., 2]) + eps
    # δᵀ Ω δ = ||Mᵀ δ||² ,  Mᵀ = [[l11, l21],[0, l22]]
    a = l11 * delta[..., 0] + l21 * delta[..., 1]
    b = l22 * delta[..., 1]
    quad = a * a + b * b
    logdet = 2.0 * (torch.log(l11) + torch.log(l22))
    return 0.5 * quad - 0.5 * logdet


def _sample_field(uv, field, H, W, C):
    """Bilinear-sample a co-located field at uv. field (B,V,H,W,J,C), uv (B,V,J,2)
    px -> (B,V,J,C). uv is expected DETACHED (the metric must not depend on θ);
    gradient still flows to `field` (the network output)."""
    B, V, J = uv.shape[:3]
    px = uv[..., 0].clamp(0.0, W - 1.0)
    py = uv[..., 1].clamp(0.0, H - 1.0)
    x0 = px.floor().long().clamp(0, W - 2)
    y0 = py.floor().long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (px - x0.float()).unsqueeze(-1)
    wy = (py - y0.float()).unsqueeze(-1)
    bb = torch.arange(B, device=uv.device).view(B, 1, 1).expand(B, V, J).reshape(-1)
    bv = torch.arange(V, device=uv.device).view(1, V, 1).expand(B, V, J).reshape(-1)
    bj = torch.arange(J, device=uv.device).view(1, 1, J).expand(B, V, J).reshape(-1)

    def corner(xi, yi):
        return field[bb, bv, yi.reshape(-1), xi.reshape(-1), bj, :].reshape(B, V, J, C)

    f00 = corner(x0, y0); f10 = corner(x1, y0)
    f01 = corner(x0, y1); f11 = corner(x1, y1)
    return (f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy)
            + f01 * (1 - wx) * wy + f11 * wx * wy)


# ──────────────────────────────────────────────────────────────────────────────
# One linearization: residual (2-vectors), detached Jacobian, 2x2 precision Ω
# ──────────────────────────────────────────────────────────────────────────────

def _verts_pose_jac(mano_layer, pose, betas, vert_idx=None):
    """(B,Nv,3) CENTRED MANO vertex positions + ∂verts/∂pose (B,Nv,3,48), via one batched vjp.

    `_joints_pose_jac` (imported above) only assumes its input is a (B,N,3) tensor
    differentiable wrt `pose` — it has no dependence on N=21 or any kinematic-joint
    semantics — so it works VERBATIM for arbitrary MANO vertices. `vert_idx` (LongTensor or
    None=all 778) selects which vertices to use; selecting BEFORE the vjp (rather than after)
    keeps the batched-grad cost proportional to len(vert_idx), not 778."""
    with torch.enable_grad():
        pl = pose.detach().requires_grad_(True)
        verts = mano_layer(pl, betas.detach()).verts              # (B,778,3) centred
        if vert_idx is not None:
            verts = verts[:, vert_idx]                            # (B,Nv,3)
        dV = _joints_pose_jac(verts, pl)                          # (B,Nv,3,48)
    return verts.detach(), dV


def _build_system_unc(mano_layer, J_regressor, pose, trans, betas, K, w2c, dovf, chol,
                      num_joints, H, W, jac_mode, anc_mask, center_idx, radius, view_weight,
                      gate=None, corr=None, mu3d=None, omega3=None, anat_on=True, sdf_on=True,
                      temporal_on=True):
    """Returns rv (B,M,2) [grad wrt dovf], Jm (B,M,2,51) [DETACHED, GN constant],
    Om (B,M,2,2) [grad wrt chol,gate], with M = V*J observation blocks.
    `corr` (B,V,J,2) is an optional cross-view 2D refinement ADDED to the DOVF
    residual (constant wrt θ, so it leaves the Jacobian unchanged; grad -> refiner)."""
    B = pose.shape[0]; P = 51
    if jac_mode == "analytic":
        out = mano_layer(pose, betas)
        jc = mano_to_openpose(J_regressor, out.verts)[:, :num_joints]
        jw = jc + trans.unsqueeze(1)
        dJc = mano_kinematic_jac(out, mano_layer, pose.detach(), betas, anc_mask,
                                 center_idx, num_joints).detach()
        if RECORD_JAC:
            # exact autograd articulated Jacobian at the SAME linearization point
            with torch.enable_grad():
                pl_ = pose.detach().requires_grad_(True)
                _, jcg_ = _mano_joints(mano_layer, J_regressor, pl_, betas, trans.detach(), num_joints)
                dJc_auto = _joints_pose_jac(jcg_, pl_).detach()
            a_ = dJc.reshape(dJc.shape[0], -1); e_ = dJc_auto.reshape(dJc.shape[0], -1)
            cos = torch.nn.functional.cosine_similarity(a_, e_, dim=-1)               # (B,)
            rel = (a_ - e_).norm(dim=-1) / e_.norm(dim=-1).clamp(min=1e-9)            # (B,)
            JAC_AGREE.append((cos.detach().cpu(), rel.detach().cpu()))
    else:
        with torch.enable_grad():
            pl = pose.detach().requires_grad_(True)
            _, jcg = _mano_joints(mano_layer, J_regressor, pl, betas, trans.detach(), num_joints)
            dJc = _joints_pose_jac(jcg, pl).detach()
        jw, _ = _mano_joints(mano_layer, J_regressor, pose, betas, trans, num_joints)

    uv, duv_dXw = _project_and_jac(jw, K, w2c)              # uv (B,V,J,2), (B,V,J,2,3)
    r, dr_duv = _sample_dovf_and_grad(uv, dovf, H, W)       # r (B,V,J,2), dr_duv (B,V,J,2,2)
    if corr is not None:
        r = r + corr                                       # cross-view 2D refinement (const wrt θ)
    Vn = uv.shape[1]

    # Jacobian dr/dθ (DETACHED, kept as (...,2,51) blocks — NOT flattened to scalars)
    dr_dXw = torch.einsum("bvjci,bvjik->bvjck", dr_duv.detach(), duv_dXw.detach())       # (B,V,J,2,3)
    dr_dpose = torch.einsum("bvjck,bvjkp->bvjcp", dr_dXw,
                            dJc.unsqueeze(1).expand(B, Vn, -1, -1, -1))                   # (B,V,J,2,48)
    Jm = torch.cat([dr_dpose, dr_dXw], dim=-1).reshape(B, Vn * num_joints, 2, P).detach()

    # 2x2 precision sampled at the CURRENT projection (uv detached -> no θ-dep;
    # grad still flows to `chol`). Robust × view × gate scalars multiply Ω.
    L = _sample_field(uv.detach(), chol, H, W, 3)          # (B,V,J,3)
    Om = chol_to_prec(L)                                   # (B,V,J,2,2)  grad wrt chol
    if EVAL_OMEGA_COND_MAX is not None:
        # Eval-time stabilization: cap the anisotropy of Ω (eigenvalue ratio) so a
        # pathological learned precision can't make the GN step blow up. Shrinks
        # toward isotropic by raising the smaller eigenvalue; no retraining needed.
        w, V = torch.linalg.eigh(Om)                       # w ascending: w[...,0] <= w[...,1]
        lo = w[..., 1:2] / EVAL_OMEGA_COND_MAX
        w = torch.cat([torch.maximum(w[..., 0:1], lo), w[..., 1:2]], dim=-1)
        Om = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
    rn = r.detach().norm(dim=-1)                           # (B,V,J)
    delta = radius.reshape(B, 1, 1) if radius.numel() == B else radius.reshape(())
    s = torch.where(rn <= delta, torch.ones_like(rn),
                    torch.sqrt(delta / rn.clamp(min=1e-9)))   # (B,V,J) huber IRLS (grad wrt radius)
    if view_weight is not None:
        s = s * view_weight.view(B, Vn, 1)
    if gate is not None:
        s = s * gate.view(B, Vn, num_joints)              # cross-view consensus trust (grad wrt gate)
    Om = Om * s[..., None, None]                          # (B,V,J,2,2)

    M = Vn * num_joints
    rv = r.reshape(B, M, 2)                               # grad wrt dovf
    Om = Om.reshape(B, M, 2, 2)

    # ── learned per-joint 3D Gaussian prior (mean mu3d + precision omega3) ──
    # adds a 3x3-weighted residual block r3 = jw - mu3d in the SAME normal eqs;
    # J3 = [dJc | I3] (∂jw/∂pose | ∂jw/∂trans), dJc reused (detached, GN-constant).
    j3 = None
    cost = 0.5 * torch.einsum("bma,bmac,bmc->b", rv, Om, rv)        # data cost (B,)
    if mu3d is not None and omega3 is not None:
        I3 = torch.eye(3, device=dJc.device, dtype=dJc.dtype).expand(B, num_joints, 3, 3)
        J3 = torch.cat([dJc, I3], dim=-1)                # (B,J,3,51)
        r3 = jw - mu3d                                   # (B,J,3) grad -> theta & mu3d
        A3 = torch.einsum("bjai,bjac,bjck->bik", J3, omega3, J3)   # (B,51,51)
        g3 = torch.einsum("bjai,bjac,bjc->bi", J3, omega3, r3)     # (B,51)
        j3 = (A3, g3)
        cost = cost + 0.5 * torch.einsum("bja,bjac,bjc->b", r3, omega3, r3)   # + 3D-prior cost

    # ── inference-time temporal-coherence anchor (streaming, eval-only) ──
    # Merge a joint-space quadratic pull toward `target` (e.g. previous frame's
    # reconstruction) into the SAME 3D-prior block, so it flows through the fitter's
    # accept/reject + implicit backward with no extra plumbing. Isotropic weight.
    if TEMPORAL_ANCHOR is not None and temporal_on:
        t_terms = TEMPORAL_ANCHOR if isinstance(TEMPORAL_ANCHOR, list) else [TEMPORAL_ANCHOR]
        I3t = torch.eye(3, device=dJc.device, dtype=dJc.dtype).expand(B, num_joints, 3, 3)
        J3t = torch.cat([dJc, I3t], dim=-1)                         # (B,J,3,51)  ∂jw/∂θ (shared)
        for t_tgt, t_w in t_terms:                                  # e.g. one per neighbouring frame
            om_t = I3t * float(t_w)                                 # (B,J,3,3) isotropic precision
            r3t = jw - t_tgt.to(jw.dtype)                           # (B,J,3) meters
            A3t = torch.einsum("bjai,bjac,bjck->bik", J3t, om_t, J3t)   # (B,51,51)
            g3t = torch.einsum("bjai,bjac,bjc->bi", J3t, om_t, r3t)     # (B,51)
            j3 = (A3t, g3t) if j3 is None else (j3[0] + A3t, j3[1] + g3t)
            cost = cost + 0.5 * torch.einsum("bja,bjac,bjc->b", r3t, om_t, r3t)

    # ── inference-time CONSECUTIVE-FRAME batch coupling (see TEMPORAL_NEIGHBOR_W docstring) ──
    if TEMPORAL_NEIGHBOR_W is not None and temporal_on and B > 1:
        tw = float(TEMPORAL_NEIGHBOR_W)
        I3n = torch.eye(3, device=dJc.device, dtype=dJc.dtype).expand(B, num_joints, 3, 3)
        J3n = torch.cat([dJc, I3n], dim=-1)                          # (B,J,3,51) -- reuses dJc, no new Jacobian
        om_n = I3n * tw
        jw_det = jw.detach()
        prev_ok = torch.zeros(B, device=jw.device, dtype=jw.dtype); prev_ok[1:] = 1.0
        prev_tgt = torch.cat([jw_det[:1], jw_det[:-1]], dim=0)       # row 0 is a dummy, masked by prev_ok
        r_p = (jw - prev_tgt) * prev_ok.view(B, 1, 1)
        A_p = torch.einsum("bjai,bjac,bjck->bik", J3n, om_n, J3n) * prev_ok.view(B, 1, 1)
        g_p = torch.einsum("bjai,bjac,bjc->bi", J3n, om_n, r_p)
        next_ok = torch.zeros(B, device=jw.device, dtype=jw.dtype); next_ok[:-1] = 1.0
        next_tgt = torch.cat([jw_det[1:], jw_det[-1:]], dim=0)       # row B-1 is a dummy, masked by next_ok
        r_n = (jw - next_tgt) * next_ok.view(B, 1, 1)
        A_n = torch.einsum("bjai,bjac,bjck->bik", J3n, om_n, J3n) * next_ok.view(B, 1, 1)
        g_n = torch.einsum("bjai,bjac,bjc->bi", J3n, om_n, r_n)
        An, gn = A_p + A_n, g_p + g_n
        j3 = (An, gn) if j3 is None else (j3[0] + An, j3[1] + gn)
        cost = (cost + 0.5 * torch.einsum("bja,bjac,bjc->b", r_p, om_n, r_p)
                     + 0.5 * torch.einsum("bja,bjac,bjc->b", r_n, om_n, r_n))

    # ── inference-time per-view 2D reprojection energy (external 2D evidence) ──
    # WILOR_2D = (tgt2d (B,V,J,2) hm-px, w2d scalar[, om2d (B,V,J,2,2) learned
    # precision]). With om2d the residual is block-weighted r Ω r (anisotropic,
    # per (view,joint)) -> the I/O uncertainty adapter's Omega slots in here;
    # without it the term is the legacy isotropic scalar-weighted L2.
    # WILOR_2D may be a single (tgt2d, w[, om2d]) tuple OR a LIST of such tuples
    # (one per frozen 2D expert). The projection Jacobian Jw is shared across
    # experts; each expert adds its own block-weighted residual -> product-of-experts.
    if WILOR_2D is not None:
        terms = WILOR_2D if isinstance(WILOR_2D, list) else [WILOR_2D]
        Pp = dJc.shape[-1]                                          # pose DOFs (48)
        dJc_e = dJc.unsqueeze(1).expand(B, Vn, num_joints, 3, Pp)   # (B,V,J,3,48)
        Jw_pose = torch.einsum("bvjck,bvjkp->bvjcp", duv_dXw.detach(), dJc_e)  # (B,V,J,2,48)
        Jw = torch.cat([Jw_pose, duv_dXw.detach()], dim=-1)        # (B,V,J,2,51) [pose|trans]
        M2 = Vn * num_joints
        Jw = Jw.reshape(B, M2, 2, Pp + 3)
        eye2 = torch.eye(2, device=uv.device, dtype=uv.dtype).view(1, 1, 1, 2, 2)
        for term in terms:
            tgt2d, w2d = term[0], term[1]
            om2d = term[2] if len(term) > 2 else None
            rw = uv.detach() - tgt2d.to(uv.dtype)                  # (B,V,J,2) heatmap px
            W2 = (om2d.to(uv.dtype) * float(w2d)) if om2d is not None \
                else eye2.expand(B, Vn, num_joints, 2, 2) * float(w2d)
            if WILOR_2D_HUBER is not None:
                rn2 = rw.detach().norm(dim=-1)                     # (B,V,J)
                delta2 = float(WILOR_2D_HUBER)
                s2 = torch.where(rn2 <= delta2, torch.ones_like(rn2),
                                 torch.sqrt(delta2 / rn2.clamp(min=1e-9)))
                W2 = W2 * s2[..., None, None]                      # downweight large (likely wrong-view) residuals
            if view_weight is not None:
                W2 = W2 * view_weight.view(B, Vn, 1, 1, 1)         # mask padded/dropped views
            rw = rw.reshape(B, M2, 2); W2 = W2.reshape(B, M2, 2, 2)
            A2 = torch.einsum("bmcp,bmcd,bmdq->bpq", Jw, W2, Jw)  # (B,51,51)
            g2 = torch.einsum("bmcp,bmcd,bmd->bp", Jw, W2, rw)    # (B,51)
            j3 = (A2, g2) if j3 is None else (j3[0] + A2, j3[1] + g2)
            cost = cost + 0.5 * torch.einsum("bmc,bmcd,bmd->b", rw, W2, rw)

    # ── inference-time per-view 2D VERTEX reprojection energy (dense analogue of WILOR_2D
    # above; see WILOR_2D_VERTS docstring). Reuses `_project_and_jac`/`_joints_pose_jac` —
    # both are generic over the point count, so no vertex-specific projection math is needed.
    if WILOR_2D_VERTS is not None:
        vterms = WILOR_2D_VERTS if isinstance(WILOR_2D_VERTS, list) else [WILOR_2D_VERTS]
        for vterm in vterms:
            tgt2dv, w2dv = vterm[0], vterm[1]
            vidx = vterm[2] if len(vterm) > 2 else None
            om2dv = vterm[3] if len(vterm) > 3 else None
            vcen, dV = _verts_pose_jac(mano_layer, pose, betas, vidx)         # (B,Nv,3),(B,Nv,3,48)
            vw = vcen + trans.unsqueeze(1)                                    # (B,Nv,3) world
            uvv, duv_dVw = _project_and_jac(vw, K, w2c)                       # (B,V,Nv,2),(B,V,Nv,2,3)
            Nv = vw.shape[1]; Pp_v = dV.shape[-1]                            # pose DOFs (48), local (WILOR_2D may be off)
            dV_e = dV.unsqueeze(1).expand(B, Vn, Nv, 3, Pp_v)
            Jv_pose = torch.einsum("bvick,bvikp->bvicp", duv_dVw.detach(), dV_e)  # (B,V,Nv,2,48)
            Jv = torch.cat([Jv_pose, duv_dVw.detach()], dim=-1)              # (B,V,Nv,2,51) [pose|trans]
            M2v = Vn * Nv
            Jv = Jv.reshape(B, M2v, 2, Pp_v + 3)
            rwv = uvv.detach() - tgt2dv.to(uvv.dtype)                        # (B,V,Nv,2)
            eye2v = torch.eye(2, device=uvv.device, dtype=uvv.dtype).view(1, 1, 1, 2, 2)
            W2v = (om2dv.to(uvv.dtype) * float(w2dv)) if om2dv is not None \
                else eye2v.expand(B, Vn, Nv, 2, 2) * float(w2dv)
            if WILOR_2D_VERTS_HUBER is not None:
                rnv = rwv.detach().norm(dim=-1)                             # (B,V,Nv)
                deltav = float(WILOR_2D_VERTS_HUBER)
                sv = torch.where(rnv <= deltav, torch.ones_like(rnv),
                                  torch.sqrt(deltav / rnv.clamp(min=1e-9)))
                W2v = W2v * sv[..., None, None]
            if view_weight is not None:
                W2v = W2v * view_weight.view(B, Vn, 1, 1, 1)
            rwv = rwv.reshape(B, M2v, 2); W2v = W2v.reshape(B, M2v, 2, 2)
            A2v = torch.einsum("bmcp,bmcd,bmdq->bpq", Jv, W2v, Jv)
            g2v = torch.einsum("bmcp,bmcd,bmd->bp", Jv, W2v, rwv)
            j3 = (A2v, g2v) if j3 is None else (j3[0] + A2v, j3[1] + g2v)
            cost = cost + 0.5 * torch.einsum("bmc,bmcd,bmd->b", rwv, W2v, rwv)

    # ── inference-time hand-object no-penetration energy (eval-only, off by default) ──
    # Penetrating MANO verts get a 1-D depth residual along the surface normal; the
    # block merges into `j3` so it flows through the LM accept/reject + implicit backward.
    if OBJECT_SDF is not None and sdf_on:
        _push_fn, _sdf_w = OBJECT_SDF
        As, gs, cs = _object_sdf_block(mano_layer, pose, trans, betas, _push_fn, _sdf_w,
                                       OBJECT_SDF_MAX_VERTS, anc_mask, center_idx)
        j3 = (As, gs) if j3 is None else (j3[0] + As, j3[1] + gs)
        cost = cost + cs

    # ── inference-time anatomical-rotation barrier (eval-only, off by default) ──
    janat = None
    if ANATOMY_W is not None and anat_on:
        Aa, ga, ca = _anatomy_block(mano_layer, pose, betas, ANATOMY_W)
        janat = (Aa, ga)
        cost = cost + ca
    return rv, Jm, Om, j3, cost, janat


def _normal_eqs_unc(rv, Jm, Om, theta, theta0, lam, lm_damping, prior=None, j3=None, janat=None):
    """A = Σ Jᵀ Ω J + μI + prior_prec (+ 3D prior block + anatomy block) ,  g = Σ Jᵀ Ω r + prior_grad."""
    B, M, _, P = Jm.shape
    Jm = Jm.float(); Om = Om.float(); rv = rv.float()    # solver runs in fp32 (AMP safety)
    theta = theta.float(); theta0 = theta0.float()
    eye = torch.eye(P, device=Jm.device, dtype=Jm.dtype).unsqueeze(0)
    # n=batch, m=obs block, (a,c)=2x2 residual-component indices, (p,q)=param indices.
    # NB: do NOT reuse 'b' for batch here — it collides with the 2x2 matrix axis.
    lmd = (lm_damping.to(Jm.dtype).view(-1, 1, 1)
           if torch.is_tensor(lm_damping) and lm_damping.ndim == 1 else lm_damping)
    A = torch.einsum("nmap,nmac,nmcq->npq", Jm, Om, Jm) + lmd * eye
    g = torch.einsum("nmap,nmac,nmc->np", Jm, Om, rv)
    if j3 is not None:                                   # learned 3D prior contribution
        A = A + j3[0].float()
        g = g + j3[1].float()
    if janat is not None:                                # anatomical-rotation barrier
        A = A + janat[0].float()
        g = g + janat[1].float()
    if prior is None:
        A = A + lam * eye
        g = g + lam * (theta - theta0)
    else:
        prec, ref, anchor = prior
        dt = A.dtype                                          # unify dtype (AMP safety)
        prec = prec.to(dt); anchor = anchor.to(dt)
        ref = ref.to(dt) if torch.is_tensor(ref) else ref
        th = theta.to(dt); th0 = theta0.to(dt)
        A = A + prec.unsqueeze(0) + torch.diag(anchor).unsqueeze(0)
        g = (g + torch.einsum("ij,bj->bi", prec, th - ref)
             + anchor * (th - th0))
    return A, g


def _gn_loop_unc(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf, chol, gate, radius,
                 num_joints, H, W, jac_mode, anc_mask, center_idx, view_weight,
                 max_iterations, step_size, lam, lm_damping, prior=None, corr=None,
                 mu3d=None, omega3=None):
    """Adaptive Levenberg-Marquardt: full steps with per-sample trust-region damping
    (lambda up on a cost increase, down on a decrease). Converges where fixed-step GN
    drifts/diverges, so the result is a true fixed point (-> implicit backward valid)."""
    B = pose0.shape[0]
    theta0 = torch.cat([pose0, trans0], dim=-1)
    theta = theta0.clone()
    eye = torch.eye(51, device=theta.device, dtype=theta.dtype)

    def _prior_cost(th):
        if prior is not None:
            prec, ref, anchor = prior
            rfd = ref.to(th.dtype) if torch.is_tensor(ref) else ref
            r = th - rfd
            c = 0.5 * torch.einsum("bi,ij,bj->b", r, prec.to(th.dtype), r)
            return c + 0.5 * (anchor.to(th.dtype) * (th - theta0) ** 2).sum(-1)
        return 0.5 * lam * ((th - theta0) ** 2).sum(-1)

    def _build(th, anat_on=True, sdf_on=True, temporal_on=True):
        ps, tr = th[:, :48], th[:, 48:]
        rv, Jm, Om, j3, cdr, janat = _build_system_unc(mano_layer, J_regressor, ps, tr, betas, K, w2c, dovf,
                                                       chol, num_joints, H, W, jac_mode, anc_mask, center_idx,
                                                       radius, view_weight, gate, corr, mu3d, omega3, anat_on,
                                                       sdf_on, temporal_on)
        return rv, Jm, Om, j3, janat, cdr + _prior_cost(th)

    # last-k schedule: run the (autograd) barriers / temporal anchor only in the final k
    # iterations (the cheap data/vertex term converges first). Independent schedules.
    anat_start = (max(0, max_iterations - ANATOMY_LAST_K)
                  if (ANATOMY_W is not None and ANATOMY_LAST_K is not None) else 0)
    sdf_start = (max(0, max_iterations - OBJECT_SDF_LAST_K)
                 if (OBJECT_SDF is not None and OBJECT_SDF_LAST_K is not None) else 0)
    temporal_start = (max(0, max_iterations - TEMPORAL_ANCHOR_LAST_K)
                      if ((TEMPORAL_ANCHOR is not None or TEMPORAL_NEIGHBOR_W is not None)
                          and TEMPORAL_ANCHOR_LAST_K is not None) else 0)
    cur_anat = (anat_start == 0)
    cur_sdf = (sdf_start == 0)
    cur_temporal = (temporal_start == 0)
    rv, Jm, Om, j3, janat, cost = _build(theta, cur_anat, cur_sdf, cur_temporal)
    lm0 = float(lm_damping if not torch.is_tensor(lm_damping) else float(lm_damping.mean()))
    lm = theta.new_full((B,), 1e-8 if GN_MODE == "pure" else max(lm0, 1e-4))
    adapt = (GN_MODE == "lm")                              # trust-region on only in LM mode
    if RECORD_TRAJ:
        TRAJ.append(float(cost.detach().mean()))
    if RECORD_POSE_TRAJ:
        POSE_TRAJ.append((theta[:, :48].detach().cpu().clone(), theta[:, 48:].detach().cpu().clone()))
    for it in range(max_iterations):
        if (((not cur_anat) and it >= anat_start) or ((not cur_sdf) and it >= sdf_start)
                or ((not cur_temporal) and it >= temporal_start)):
            cur_anat = cur_anat or (it >= anat_start)                 # entering a barrier phase:
            cur_sdf = cur_sdf or (it >= sdf_start)
            cur_temporal = cur_temporal or (it >= temporal_start)
            rv, Jm, Om, j3, janat, cost = _build(theta, cur_anat, cur_sdf, cur_temporal)  # re-cost θ
        A, g = _normal_eqs_unc(rv, Jm, Om, theta, theta0, lam, lm, prior, j3, janat)
        d = torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
        theta_new = theta - d                                         # full LM/GN step
        rv2, Jm2, Om2, j32, janat2, cost2 = _build(theta_new, cur_anat, cur_sdf, cur_temporal)
        acc = (cost2 < cost) if adapt else torch.ones_like(cost, dtype=torch.bool)  # GN: always step
        theta = torch.where(acc.view(B, 1), theta_new, theta)
        cost = torch.where(acc, cost2, cost)
        rv = torch.where(acc.view(B, 1, 1), rv2, rv)
        Om = torch.where(acc.view(B, 1, 1, 1), Om2, Om)
        Jm = torch.where(acc.view(B, 1, 1, 1), Jm2, Jm)
        if j3 is not None:
            j3 = (torch.where(acc.view(B, 1, 1), j32[0], j3[0]),
                  torch.where(acc.view(B, 1), j32[1], j3[1]))
        if janat is not None:
            janat = (torch.where(acc.view(B, 1, 1), janat2[0], janat[0]),
                     torch.where(acc.view(B, 1), janat2[1], janat[1]))
        if adapt:
            lm = torch.where(acc, (lm * 0.3).clamp(min=1e-6), (lm * 3.0).clamp(max=1e6))
        if RECORD_TRAJ:
            TRAJ.append(float(cost.detach().mean()))
        if RECORD_POSE_TRAJ:
            POSE_TRAJ.append((theta[:, :48].detach().cpu().clone(), theta[:, 48:].detach().cpu().clone()))
    if RECORD_FISHER:
        # Jm/Om/j3 correspond to the ACCEPTED theta (the trust-region where() above keeps
        # them in sync), so this is the linearization at the converged fixed point.
        with torch.no_grad():
            Jf = Jm.float(); Of = Om.float()
            A_data = torch.einsum("nmap,nmac,nmcq->npq", Jf, Of, Jf)
            A_full = A_data + (j3[0].float() if j3 is not None else 0.0)
            if janat is not None:
                A_full = A_full + janat[0].float()
            ps, tr = theta[:, :48], theta[:, 48:]
            out = mano_layer(ps, betas)
            if anc_mask is not None:
                dJc = mano_kinematic_jac(out, mano_layer, ps, betas, anc_mask, center_idx, num_joints)
            else:
                with torch.enable_grad():
                    pl = ps.detach().requires_grad_(True)
                    _, jcg = _mano_joints(mano_layer, J_regressor, pl, betas, tr.detach(), num_joints)
                    dJc = _joints_pose_jac(jcg, pl)
            FISHER_OUT.append({"A_data": A_data.detach().cpu(), "A_full": A_full.detach().cpu(),
                               "dJc": dJc.detach().cpu(), "Om": Of.detach().cpu(),
                               "Jm": Jf.detach().cpu()})
    return theta[:, :48], theta[:, 48:]


class _ImplicitGNUnc(torch.autograd.Function):
    """Implicit differentiation of the block-weighted GN fixed point. Same scheme
    as :class:`analytic_fitter._ImplicitGN`, extended to backprop into the
    uncertainty field `chol` and the consensus `gate`."""

    @staticmethod
    def forward(ctx, dovf, chol, gate, corr, pose0, trans0, betas, log_radius, K, w2c, view_weight,
                mano_layer, J_regressor, anc_mask, num_joints, H, W,
                max_iterations, step_size, lam, lm_damping, jac_mode, center_idx, prior,
                mu3d=None, omega3=None):
        radius = log_radius.exp().reshape(-1)
        with torch.no_grad():
            pose, trans = _gn_loop_unc(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf, chol,
                                       gate, radius, num_joints, H, W, jac_mode, anc_mask, center_idx,
                                       view_weight, max_iterations, step_size, lam, lm_damping, prior,
                                       corr, mu3d, omega3)
        z = dovf.new_zeros(0)
        ctx.save_for_backward(dovf, chol, gate if gate is not None else z,
                              corr if corr is not None else z, pose0, trans0, betas,
                              log_radius, K, w2c, view_weight if view_weight is not None else z,
                              anc_mask if anc_mask is not None else z, pose, trans,
                              mu3d if mu3d is not None else z, omega3 if omega3 is not None else z)
        ctx.cfg = (mano_layer, J_regressor, num_joints, H, W, lam, lm_damping, jac_mode, center_idx,
                   gate is not None, view_weight is not None, anc_mask is not None, corr is not None,
                   mu3d is not None)
        ctx.prior = prior
        return pose, trans

    @staticmethod
    def backward(ctx, g_pose, g_trans):
        (dovf, chol, gate_s, corr_s, pose0, trans0, betas, log_radius, K, w2c, vw_s, anc_s,
         pose, trans, mu3d_s, omega3_s) = ctx.saved_tensors
        (mano_layer, J_regressor, num_joints, H, W, lam, lm_damping, jac_mode, center_idx,
         has_gate, has_vw, has_anc, has_corr, has_mu3d) = ctx.cfg
        view_weight = vw_s if has_vw else None
        anc_mask = anc_s if has_anc else None

        dovf_ = dovf.detach().requires_grad_(True)
        chol_ = chol.detach().requires_grad_(True)
        gate_ = gate_s.detach().requires_grad_(True) if has_gate else None
        corr_ = corr_s.detach().requires_grad_(True) if has_corr else None
        pose0_ = pose0.detach().requires_grad_(True)
        trans0_ = trans0.detach().requires_grad_(True)
        betas_ = betas.detach().requires_grad_(True)
        lr_ = log_radius.detach().requires_grad_(True)
        mu3d_ = mu3d_s.detach().requires_grad_(True) if has_mu3d else None
        omega3_ = omega3_s.detach().requires_grad_(True) if has_mu3d else None
        with torch.enable_grad():
            radius = lr_.exp().reshape(-1)
            rv, Jm, Om, j3, _, janat = _build_system_unc(mano_layer, J_regressor, pose.detach(), trans.detach(),
                                                         betas_, K, w2c, dovf_, chol_, num_joints, H, W, jac_mode,
                                                         anc_mask, center_idx, radius, view_weight, gate_, corr_,
                                                         mu3d_, omega3_)
            theta = torch.cat([pose, trans], dim=-1).detach()
            theta0 = torch.cat([pose0_, trans0_], dim=-1)
            A, g = _normal_eqs_unc(rv, Jm, Om, theta, theta0, lam, lm_damping, ctx.prior, j3, janat)
        ghat = torch.cat([g_pose, g_trans], dim=-1)
        v = torch.linalg.solve(A.detach(), ghat.unsqueeze(-1)).squeeze(-1)
        inputs = [dovf_, chol_, pose0_, trans0_, betas_, lr_]
        if has_gate: inputs.append(gate_)
        if has_corr: inputs.append(corr_)
        if has_mu3d: inputs += [mu3d_, omega3_]
        grads = torch.autograd.grad(g, inputs, grad_outputs=v, allow_unused=True)
        gd = [(-x if x is not None else None) for x in grads]
        d_dovf, d_chol, d_p0, d_t0, d_betas, d_lr = gd[:6]
        k = 6
        d_gate = gd[k] if has_gate else None
        if has_gate: k += 1
        d_corr = gd[k] if has_corr else None
        if has_corr: k += 1
        d_mu3d = gd[k] if has_mu3d else None
        d_omega3 = gd[k + 1] if has_mu3d else None
        # order matches forward inputs (last two = mu3d, omega3):
        return (d_dovf, d_chol, d_gate, d_corr, d_p0, d_t0, d_betas, d_lr, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None,
                d_mu3d, d_omega3)


def analytic_fit_unc(
    mano_layer, J_regressor,
    pose0, trans0, betas, K, w2c, dovf, chol,
    gate: Optional[torch.Tensor] = None,
    corr: Optional[torch.Tensor] = None,
    mu3d: Optional[torch.Tensor] = None,
    omega3: Optional[torch.Tensor] = None,
    num_joints=21, H=64, W=64,
    max_iterations=10, step_size=0.5, pose_prior_weight=1.0,
    log_radius: Optional[torch.Tensor] = None, loss_radius_init=3.0,
    lm_damping=1e-3,
    view_weight: Optional[torch.Tensor] = None,
    jac_mode="autograd",
    center_idx=0,
    backward="unroll",
    prior_prec: Optional[torch.Tensor] = None,
    prior_ref: Optional[torch.Tensor] = None,
    prior_anchor: Optional[torch.Tensor] = None,
):
    """Block-weighted (2x2 Ω) damped Gauss-Newton DOVF fit. Returns (pose, trans).

    chol  (B,V,H,W,J,3): co-located Cholesky field -> 2x2 precision Ω per (view,joint).
    gate  (B,V,J) or None: per-(view,joint) cross-view consensus trust in (0,1].
    Reduces to ``analytic_fit`` when Ω = w·I and gate = 1.
    """
    device = pose0.device
    lam = float(pose_prior_weight)
    anc_mask = _ancestor_mask(mano_layer.kintree_parents, device=device) if jac_mode == "analytic" else None
    if log_radius is None:
        log_radius = torch.log(torch.tensor([loss_radius_init], device=device))

    prior = None
    if prior_prec is not None:
        prior = (prior_prec, prior_ref, prior_anchor)

    if backward == "implicit":
        return _ImplicitGNUnc.apply(dovf, chol, gate, corr, pose0, trans0, betas, log_radius, K, w2c,
                                    view_weight, mano_layer, J_regressor, anc_mask, num_joints, H, W,
                                    max_iterations, step_size, lam, lm_damping, jac_mode, center_idx, prior,
                                    mu3d, omega3)

    radius = log_radius.exp().reshape(-1)
    return _gn_loop_unc(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf, chol, gate, radius,
                        num_joints, H, W, jac_mode, anc_mask, center_idx, view_weight,
                        max_iterations, step_size, lam, lm_damping, prior, corr, mu3d, omega3)
