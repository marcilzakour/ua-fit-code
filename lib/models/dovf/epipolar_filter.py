"""
Epipolar information-filter refiner (Phases A + C of the efficiency-line design).
====================================================================================

Replaces the inert ``MixtureMCHead`` sigma-point scorer (which mean-pools over views
and is provably a no-op) with two principled, mostly param-free pieces:

* **Phase A — precision-weighted voting** (:func:`precision_consensus_2d`): every
  pixel votes ``V = grid + offset`` with an *anisotropic per-vote precision* ``Λ``
  (from a Cholesky field). The consensus is the precision-weighted MLE of the votes,
  and the fused 2D precision ``Ω2`` falls out of the *same* normal matrix
  (``A = Σ wΛ``, ``cons = A⁻¹ b``, ``Ω2 = A·τ``). Self-consistent; replaces the old
  hm-weighted vote + the sample-at-consensus ``mc_omega_scale`` fudge.

* **Phase C — recurrent epipolar information filter** (:func:`run_epipolar_filter`
  + :class:`EpipolarRefiner`): state is the 3D Gaussian belief ``(μ, Ω3=Σ3⁻¹)`` from
  triangulation. For ``T`` iterations over random view-pairs ``(v, vp)`` it samples
  in-plane features around ``cons[v]`` (window sized by the reprojected ``Σ3``) and
  depth features ALONG the epipolar line in ``vp`` (beam extent = depth-σ from
  ``Σ3``) — *no mean-pool*. A small SHARED refiner localizes the joint along the beam
  → a 2D measurement ``c_meas = cons[vp] + Δc`` with precision ``Ωm`` and a gate. The
  belief is updated with the EXACT additive information-filter step using the
  reprojection Jacobian:  ``Ω3 += gate·JᵀΩm J``,  ``μ += Σ3·(gate·JᵀΩm r)``.

The 3D fusion (voting, triangulation, info update) is parameter-free; the only
learned pieces are the 2D front-end (offset + per-vote precision) and the shared
refiner. Camera convention follows the rest of the repo: ``w2c = [R|t]`` world->cam,
``K`` and 2D points in HEATMAP pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .analytic_fitter_unc import chol_to_prec
from .triangulate_resample import _pad_by_scene, triangulate_omega2


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers (param-free) — projection + Jacobian, back-projected ray
# ──────────────────────────────────────────────────────────────────────────────

def proj_and_jac(mu, K, w2c):
    """Project world points ``mu`` (B,J,3) into a view and return the projection +
    its Jacobian wrt the world point.

    K (B,3,3) heatmap-pixel intrinsics, w2c (B,4,4) world->cam.
    Returns uv (B,J,2) and J = d(uv)/d(X_world) (B,J,2,3).  (Same linearization the
    GN triangulator uses, so the info update below is consistent with Phase B.)
    """
    R = w2c[:, :3, :3]; t = w2c[:, :3, 3]
    fx = K[:, 0, 0, None]; fy = K[:, 1, 1, None]
    cx = K[:, 0, 2, None]; cy = K[:, 1, 2, None]
    Xc = torch.einsum("bij,bkj->bki", R, mu) + t[:, None, :]            # (B,J,3)
    x, y = Xc[..., 0], Xc[..., 1]; z = Xc[..., 2].clamp(min=1e-4)
    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)        # (B,J,2)
    B, J = uv.shape[:2]
    Juvc = uv.new_zeros(B, J, 2, 3)                                     # d uv / d Xc
    Juvc[..., 0, 0] = fx / z; Juvc[..., 0, 2] = -fx * x / (z * z)
    Juvc[..., 1, 1] = fy / z; Juvc[..., 1, 2] = -fy * y / (z * z)
    Jac = torch.einsum("bjac,bcd->bjad", Juvc, R)                      # d uv / d Xworld
    return uv, Jac


def ray_dir(uv, K, w2c):
    """World-space unit ray direction through heatmap-pixel ``uv`` (B,J,2)."""
    R = w2c[:, :3, :3]; Rt = R.transpose(-1, -2)
    uvh = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)         # (B,J,3)
    Kinv = torch.linalg.inv(K)
    d_cam = torch.einsum("bij,bkj->bki", Kinv, uvh)                     # (B,J,3) cam
    d = torch.einsum("bij,bkj->bki", Rt, d_cam)                         # (B,J,3) world
    return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _pixel_grid(h, w, device, dtype):
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)                               # (h,w,2)


def _grid_sample_pts(feat_map, uv, h, w):
    """Bilinear-sample feat_map (N,C,H,W) at uv (N,Q,2) heatmap px -> (N,Q,C)."""
    N, C = feat_map.shape[:2]
    gx = 2.0 * uv[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * uv[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(N, -1, 1, 2)             # (N,Q,1,2)
    samp = F.grid_sample(feat_map, grid, mode="bilinear",
                         align_corners=True, padding_mode="border")    # (N,C,Q,1)
    return samp.squeeze(-1).permute(0, 2, 1).contiguous()              # (N,Q,C)


# ──────────────────────────────────────────────────────────────────────────────
# Phase A — precision-weighted voting -> consensus + fused Ω2
# ──────────────────────────────────────────────────────────────────────────────

def precision_consensus_2d(hm_probs, dovf_field, cholP, log_tau, ess=False, eps=1e-4):
    """Precision-weighted MLE consensus of the dense votes.

    hm_probs   (BN,J,h,w)    softmax heatmap = support weight w_p
    dovf_field (BN,J,2,h,w)  per-pixel offset o_p  (vote V_p = p + o_p)
    cholP      (BN,J,3,h,w)  per-VOTE Cholesky -> anisotropic precision Λ_p
    log_tau    scalar param  log of the ESS/temperature τ (Ω2 = A·τ)

    Returns cons (BN,J,2) and Omega2 (BN,J,2,2) — both from the SAME fusion.
    """
    BN, J, h, w = hm_probs.shape
    grid = _pixel_grid(h, w, hm_probs.device, hm_probs.dtype)          # (h,w,2)
    off = dovf_field.permute(0, 1, 3, 4, 2)                            # (BN,J,h,w,2)
    V = grid.view(1, 1, h, w, 2) + off                                # votes (BN,J,h,w,2)
    Lp = cholP.permute(0, 1, 3, 4, 2)                                  # (BN,J,h,w,3)
    Lam = chol_to_prec(Lp)                                             # (BN,J,h,w,2,2)
    wt = hm_probs                                                      # (BN,J,h,w)

    A = (Lam * wt[..., None, None]).sum(dim=(2, 3))                    # (BN,J,2,2)
    LamV = torch.einsum("bjhwac,bjhwc->bjhwa", Lam, V)                 # (BN,J,h,w,2)
    b = (LamV * wt[..., None]).sum(dim=(2, 3))                         # (BN,J,2)
    I2 = torch.eye(2, device=hm_probs.device, dtype=hm_probs.dtype)
    cons = torch.linalg.solve(A + eps * I2, b.unsqueeze(-1)).squeeze(-1)   # (BN,J,2)

    tau = torch.exp(log_tau)
    if ess:                                                           # effective sample size
        n_eff = 1.0 / (wt.pow(2).sum(dim=(2, 3)).clamp(min=1e-6))     # (BN,J), since Σw=1
        tau = tau * n_eff[..., None, None]
    Omega2 = A * tau
    return cons, Omega2


# ──────────────────────────────────────────────────────────────────────────────
# 3D belief NLL directly from a precision matrix (the −½log|Ω3| calibration term)
# ──────────────────────────────────────────────────────────────────────────────

def _sym(X):
    """Symmetrize (fp32 round-off in einsum/inv makes SPD matrices slightly asymmetric,
    which makes cholesky reject them as 'not positive-definite')."""
    return 0.5 * (X + X.transpose(-1, -2))


def gaussian_nll_3d_from_prec(delta, Omega3, eps=1e-4):
    """½ δᵀΩ3 δ − ½ log|Ω3| for a 3x3 SPD precision. delta (B,J,3), Omega3 (B,J,3,3).
    The −log|Ω3| term punishes the over-confidence that correlated random pairs cause.
    Robust: symmetrize + TRACE-SCALED jitter (precision can be ~1e3-1e6 in 1/m², so a fixed
    1e-6 floor is useless) so cholesky never sees a non-PD input.
    """
    Om = _sym(Omega3)
    I3 = torch.eye(3, device=Om.device, dtype=Om.dtype)
    tr = torch.diagonal(Om, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-6)        # (B,J)
    jit = (eps * tr / 3.0)[..., None, None] * I3                            # scale to matrix size
    L = torch.linalg.cholesky(Om + jit)
    quad = torch.einsum("bja,bjac,bjc->bj", delta, Om, delta).clamp(min=0)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)).sum(-1)
    return 0.5 * quad - 0.5 * logdet


# ──────────────────────────────────────────────────────────────────────────────
# The shared recurrent refiner — reads the epipolar beam (no pool)
# ──────────────────────────────────────────────────────────────────────────────

class EpipolarRefiner(nn.Module):
    """Shared over iterations AND view-pairs. Given the in-plane feature template in
    view v and the depth-beam features along the epipolar line in view vp, it emits a
    soft match distribution over the D beam positions (-> 2D measurement c_meas),
    a measurement precision (cholM -> Ωm) and a gate, carrying a GRU hidden state so
    repeated application implements a coarse-to-fine schedule.

    kind:
      'corr' — cosine correlation of template vs beam (cheapest, ~0.3M)
      'mlp'  — per-beam MLP score on [beam, template, depth-pe]
      'attn' — multi-layer 1D cross-attention: template query attends the beam (strong)
    """

    def __init__(self, feat_dim, kind="attn", d=128, heads=4, layers=2, d_hidden=128,
                 gate_bias=-2.0, cholM_diag_bias=-1.0):
        super().__init__()
        self.kind = kind
        self.d = d
        self.feat_proj = nn.Linear(feat_dim, d)
        self.tmpl_proj = nn.Linear(feat_dim, d)
        self.depth_pe = nn.Linear(1, d)
        if kind == "attn":
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                             batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
            self.score = nn.Linear(d, 1)
        elif kind == "mlp":
            self.mlp = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(),
                                     nn.Linear(2 * d, d), nn.GELU())
            self.score = nn.Linear(d, 1)
        elif kind == "corr":
            self.score = None
        else:
            raise ValueError(f"unknown refiner kind '{kind}'")
        self.gru = nn.GRUCell(d, d_hidden)
        self.to_gate = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                     nn.Linear(d_hidden, 1))
        self.to_cholM = nn.Linear(d_hidden, 3)
        # Safe warm-start: gate≈sigmoid(-2)≈0.12 (gentle), cholM diag small -> low
        # measurement precision, so μ_refined ≈ μ_triangulation at init.
        nn.init.zeros_(self.to_gate[-1].weight); nn.init.constant_(self.to_gate[-1].bias, gate_bias)
        nn.init.zeros_(self.to_cholM.weight)
        with torch.no_grad():
            self.to_cholM.bias[:] = torch.tensor([cholM_diag_bias, 0.0, cholM_diag_bias])

    def forward(self, inpl, beam, depth_idx, hidden):
        """inpl (G,M,C), beam (G,D,C), depth_idx (G,D) in [-1,1], hidden (G,d_hidden).
        G = B*J flattened. Returns softw (G,D), gate (G,1), cholM (G,3), hidden (G,dh)."""
        G, D, C = beam.shape
        tmpl = self.tmpl_proj(inpl.mean(dim=1))                       # (G,d) template
        bpe = self.depth_pe(depth_idx[..., None])                     # (G,D,d)
        bfe = self.feat_proj(beam) + bpe                              # (G,D,d)
        if self.kind == "corr":
            logits = torch.einsum("gd,gld->gl", F.normalize(tmpl, dim=-1),
                                  F.normalize(self.feat_proj(beam), dim=-1)) * 8.0
            ctx_tok = bfe
        elif self.kind == "mlp":
            texp = tmpl[:, None, :].expand(G, D, self.d)
            logits = self.score(self.mlp(torch.cat([bfe, texp, bpe], dim=-1))).squeeze(-1)
            ctx_tok = bfe
        else:  # attn
            enc = self.enc(bfe)                                       # (G,D,d) beam self-attn
            q = tmpl[:, None, :]                                      # (G,1,d)
            attn_out, _ = self.cross(q, enc, enc)                    # template attends beam
            logits = self.score(enc + attn_out.expand(G, D, self.d)).squeeze(-1)
            ctx_tok = enc
        softw = torch.softmax(logits, dim=-1)                        # (G,D) depth posterior
        ctx = (softw[..., None] * ctx_tok).sum(dim=1)               # (G,d) matched context
        hidden = self.gru(ctx, hidden)
        gate = torch.sigmoid(self.to_gate(hidden))                  # (G,1)
        cholM = self.to_cholM(hidden)                               # (G,3)
        return softw, gate, cholM, hidden


# ──────────────────────────────────────────────────────────────────────────────
# Phase C driver — the recurrent information filter over random view-pairs
# ──────────────────────────────────────────────────────────────────────────────

def _scene_pair_schedule(cvn, T, generator=None, device="cpu"):
    """For each of T iters, sample one ordered pair (v, vp) per scene as FLAT image
    indices into the BN layout. Returns a list of T (iv, ivp) LongTensors (each (B,)).
    Both directions arise naturally from ordered sampling. Scenes have >=2 views."""
    offs, B = [], len(cvn)
    o = 0
    for n in cvn:
        offs.append((o, int(n))); o += int(n)
    steps = []
    for _ in range(T):
        iv = torch.empty(B, dtype=torch.long)
        ivp = torch.empty(B, dtype=torch.long)
        for i, (off, n) in enumerate(offs):
            if n >= 2:
                perm = torch.randperm(n, generator=generator)[:2]
            else:
                perm = torch.zeros(2, dtype=torch.long)
            iv[i] = off + int(perm[0]); ivp[i] = off + int(perm[1])
        steps.append((iv.to(device), ivp.to(device)))
    return steps


def run_epipolar_filter(mu, Sigma3, cons, feat, K_hm, w2c, cvn, refiner,
                        T=3, D=16, M=9, kappa=2.0, meas="corr2d",
                        generator=None, detach_mu_iters=False):
    """Recurrent epipolar information filter. All inputs are in the FLAT (BN) layout
    except (mu, Sigma3) which are per-scene (B,...).

    mu (B,J,3), Sigma3 (B,J,3,3); cons (BN,J,2); feat (BN,C,h,w); K_hm (BN,3,3);
    w2c (BN,4,4); cvn list of per-scene view counts; refiner the shared module.

    Returns dict(mu, Sigma3, mu_iters, gates, cmeas, vp_idx) — cmeas/vp_idx kept for
    the bidirectional-consistency loss.
    """
    B, J, _ = mu.shape
    C, h, w = feat.shape[1:]
    dev, dt = mu.device, mu.dtype
    I3 = torch.eye(3, device=dev, dtype=dt)
    Sigma3 = _sym(Sigma3)
    Omega3 = _sym(torch.linalg.inv(Sigma3 + 1e-6 * I3))
    G = B * J
    hidden = mu.new_zeros(G, refiner.gru.hidden_size)
    depth_idx = torch.linspace(-1.0, 1.0, D, device=dev, dtype=dt)[None, :].expand(G, D)
    # small in-plane unit window (m x m), scaled per-joint by reprojected Σ3 below
    m = int(round(M ** 0.5)); M = m * m
    lin = torch.linspace(-1.0, 1.0, m, device=dev, dtype=dt)
    wy, wx = torch.meshgrid(lin, lin, indexing="ij")
    win = torch.stack([wx.reshape(-1), wy.reshape(-1)], dim=-1)        # (M,2) unit grid

    steps = _scene_pair_schedule(cvn, T, generator=generator, device=dev)
    mu_iters = [mu]
    gates_all, cmeas_all, vpidx_all, softw_all, pz_all = [], [], [], [], []
    for (iv, ivp) in steps:
        cons_v = cons[iv]; cons_vp = cons[ivp]                        # (B,J,2)
        feat_v = feat[iv]; feat_vp = feat[ivp]                       # (B,C,h,w)
        Kv = K_hm[iv]; Kvp = K_hm[ivp]; w2cv = w2c[iv]; w2cvp = w2c[ivp]

        # in-plane window in v sized by reprojected Σ3 (Σ2v = Jv Σ3 Jvᵀ). Axis-aligned window
        # from the clamped diagonal — robust (no eigh, which diverges on ill-conditioned Σ2v).
        _, Jv = proj_and_jac(mu, Kv, w2cv)                            # (B,J,2,3)
        Sig2v = torch.einsum("bjac,bjcd,bjed->bjae", Jv, Sigma3, Jv)  # (B,J,2,2)
        var_in = torch.stack([Sig2v[..., 0, 0], Sig2v[..., 1, 1]], -1).clamp(min=1e-4, max=256.0)
        rad = var_in.sqrt().clamp(max=16.0)                          # (B,J,2) per-axis px std (capped)
        off_in = win[None, None] * rad[:, :, None, :]                # (B,J,M,2) window offsets
        uv_in = cons_v[:, :, None, :] + off_in                       # (B,J,M,2)
        inpl = _grid_sample_pts(feat_v, uv_in.reshape(B, J * M, 2), h, w).reshape(B, J, M, C)

        # depth beam in vp along the epipolar line (NO pool)
        d_v = ray_dir(cons_v, Kv, w2cv)                              # (B,J,3) world
        sig_z = torch.einsum("bja,bjac,bjc->bj", d_v, Sigma3, d_v).clamp(min=1e-8, max=0.25).sqrt()
        steps_d = torch.linspace(-kappa, kappa, D, device=dev, dtype=dt)
        Pz = mu[:, :, None, :] + steps_d[None, None, :, None] * (sig_z[:, :, None, None]
                                                                 * d_v[:, :, None, :])
        # project beam points into vp
        Rp = w2cvp[:, :3, :3]; tp = w2cvp[:, :3, 3]
        Xcp = torch.einsum("bij,bkmj->bkmi", Rp, Pz) + tp[:, None, None, :]   # (B,J,D,3)
        zc = Xcp[..., 2:3].clamp(min=1e-4)
        beam_uv = torch.einsum("bij,bkmj->bkmi", Kvp, Xcp)[..., :2] / zc      # (B,J,D,2)
        beam = _grid_sample_pts(feat_vp, beam_uv.reshape(B, J * D, 2), h, w).reshape(B, J, D, C)

        softw, gate, cholM, hidden = refiner(inpl.reshape(G, M, C), beam.reshape(G, D, C),
                                             depth_idx, hidden)
        softw = softw.reshape(B, J, D); gate = gate.reshape(B, J, 1)
        Omega_m = chol_to_prec(cholM).reshape(B, J, 2, 2)

        # measurement = expected 2D location along the beam in vp (2D-correction form)
        c_meas = (softw[..., None] * beam_uv).sum(dim=2)             # (B,J,2)
        uv_pred, Jvp = proj_and_jac(mu, Kvp, w2cvp)                  # (B,J,2),(B,J,2,3)
        r = c_meas - uv_pred                                        # (B,J,2)

        gJ = gate[..., None] * Jvp                                  # gate·Jvp (B,J,2,3)
        Omega3 = _sym(Omega3 + torch.einsum("bjca,bjcd,bjde->bjae", gJ, Omega_m, Jvp))
        Sigma3 = _sym(torch.linalg.inv(Omega3 + 1e-6 * I3))
        gJtOr = torch.einsum("bjca,bjcd,bjd->bja", gJ, Omega_m, r)  # (B,J,3)
        mu = mu + torch.einsum("bjac,bjc->bja", Sigma3, gJtOr)

        mu_iters.append(mu.detach() if detach_mu_iters else mu)
        gates_all.append(gate.reshape(B, J))
        cmeas_all.append(c_meas); vpidx_all.append(ivp)
        softw_all.append(softw); pz_all.append(Pz)               # for beam-matching CE

    return dict(mu=mu, Sigma3=Sigma3, Omega3=Omega3, mu_iters=mu_iters,
                softw=softw_all, beam_pz=pz_all,
                gates=torch.stack(gates_all, 0) if gates_all else None,
                cmeas=cmeas_all, vp_idx=vpidx_all)


def _sample_perjoint(maps, uv, h, w):
    """maps (B,J,H,W), uv (B,J,Q,2) hm px -> (B,J,Q) per-joint bilinear samples."""
    B, J, H, W = maps.shape; Q = uv.shape[2]
    out = _grid_sample_pts(maps.reshape(B * J, 1, H, W), uv.reshape(B * J, Q, 2), h, w)
    return out.reshape(B, J, Q)


def run_heatmap_filter(mu, Sigma3, cons, hmp, omega2, K_hm, w2c, cvn,
                       T=3, D=16, kappa=2.0, om_scale=0.3, gate_min=0.05,
                       generator=None, gate_mode="peak"):
    """PARAM-FREE epipolar-constrained refinement using vp's per-joint HEATMAP.

    For joint j and pair (v,vp): build depth hypotheses along v's ray (through cons[v]),
    project onto vp's epipolar line, and weight them by vp's HEATMAP channel j sampled
    there. The heatmap-weighted location c_meas is vp's epipolar-consistent estimate of j
    (sharper than the global consensus when the heatmap is multimodal). Fold it in with the
    same param-free information update. A confidence gate from the heatmap peakiness limits
    flat-heatmap (no-information) updates. No learned parameters.

    hmp (BN,J,h,w) softmax heatmap; omega2 (BN,J,2,2) per-view 2D precision (measurement prec).
    """
    B, J, _ = mu.shape
    h, w = hmp.shape[-2:]
    dev, dt = mu.device, mu.dtype
    I3 = torch.eye(3, device=dev, dtype=dt)
    Sigma3 = _sym(Sigma3)
    Omega3 = _sym(torch.linalg.inv(Sigma3 + 1e-6 * I3))
    steps = _scene_pair_schedule(cvn, T, generator=generator, device=dev)
    mu_iters = [mu]; gates_all = []
    for (iv, ivp) in steps:
        cons_v = cons[iv]; Kv = K_hm[iv]; w2cv = w2c[iv]
        Kvp = K_hm[ivp]; w2cvp = w2c[ivp]
        d_v = ray_dir(cons_v, Kv, w2cv)
        sig_z = torch.einsum("bja,bjac,bjc->bj", d_v, Sigma3, d_v).clamp(min=1e-8, max=0.25).sqrt()
        sd = torch.linspace(-kappa, kappa, D, device=dev, dtype=dt)
        Pz = mu[:, :, None, :] + sd[None, None, :, None] * (sig_z[:, :, None, None] * d_v[:, :, None, :])
        Rp = w2cvp[:, :3, :3]; tp = w2cvp[:, :3, 3]
        Xcp = torch.einsum("bij,bkmj->bkmi", Rp, Pz) + tp[:, None, None, :]
        zc = Xcp[..., 2:3].clamp(min=1e-4)
        beam_uv = torch.einsum("bij,bkmj->bkmi", Kvp, Xcp)[..., :2] / zc      # (B,J,D,2)
        h_d = _sample_perjoint(hmp[ivp], beam_uv, h, w).clamp(min=0)          # (B,J,D)
        softw = h_d / h_d.sum(-1, keepdim=True).clamp(min=1e-12)              # heatmap weights along line
        c_meas = (softw[..., None] * beam_uv).sum(dim=2)                      # (B,J,2)
        # confidence gate: peaky heatmap-along-line -> trust; flat -> ~0 (no info)
        if gate_mode == "peak":
            gate = (softw.max(-1).values - 1.0 / D).clamp(min=0) / (1.0 - 1.0 / D)
        else:
            gate = torch.ones(B, J, device=dev, dtype=dt)
        gate = gate.clamp(min=gate_min)[..., None]                           # (B,J,1)
        uv_pred, Jvp = proj_and_jac(mu, Kvp, w2cvp)
        r = c_meas - uv_pred
        Omega_m = omega2[ivp] * om_scale                                     # (B,J,2,2)
        gJ = gate[..., None] * Jvp
        Omega3 = _sym(Omega3 + torch.einsum("bjca,bjcd,bjde->bjae", gJ, Omega_m, Jvp))
        Sigma3 = _sym(torch.linalg.inv(Omega3 + 1e-6 * I3))
        mu = mu + torch.einsum("bjac,bjc->bja",
                               Sigma3, torch.einsum("bjca,bjcd,bjd->bja", gJ, Omega_m, r))
        mu_iters.append(mu); gates_all.append(gate.reshape(B, J))
    return dict(mu=mu, Sigma3=Sigma3, mu_iters=mu_iters,
                gates=torch.stack(gates_all, 0) if gates_all else None)
