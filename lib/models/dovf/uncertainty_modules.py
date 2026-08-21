"""
Uncertainty + cross-view modules for the uncertainty-aware DOVF fitter.
=======================================================================

Two small, self-contained nn.Modules (total ~2-3M params, dominated by the
transformer; see the design notes / model docstring):

* ``UncertaintyFieldHead`` — a few convs on the high-res neck feature producing a
  per-pixel, per-joint Cholesky FIELD (J×3 channels), co-located with the DOVF
  field. Sampled at the projected joint each GN iteration -> a 2x2 precision Ω.
  This is the ALEATORIC (per-view, location-dependent) uncertainty.

* ``CrossViewGate`` — a masked cross-view transformer over per-(view,joint)
  tokens (feature sampled at the consensus 2D + a relative-camera encoding). It
  attends across views (per joint) and emits a per-(view,joint) scalar gate in
  (0,1] — the EPISTEMIC (consensus / agreement) trust. Computed ONCE (front-end),
  then multiplied into Ω inside the loop. Variable view counts are handled by a
  key-padding mask on the padded (B, N_max, ...) layout the fitter already builds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyFieldHead(nn.Module):
    """High-res neck feature -> per-joint Cholesky field (B, J, 3, h, w)."""

    def __init__(self, feat_dim, n_pts, hidden=None, groups=8):
        super().__init__()
        hidden = hidden or feat_dim
        self.n_pts = n_pts
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 3, padding=1),
            nn.GroupNorm(min(groups, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_pts * 3, 1),
        )
        # Bias init: l11,l22 raw -> softplus(0)=~0.69 -> moderate precision ~0.5;
        # off-diagonal l21 -> 0 (start isotropic). Keeps early Ω well-conditioned.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, x):
        B = x.shape[0]
        o = self.net(x)                                   # (B, J*3, h, w)
        h, w = o.shape[-2:]
        return o.view(B, self.n_pts, 3, h, w)


class CrossViewGate(nn.Module):
    """Per-(view,joint) consensus gate via masked cross-view attention.

    forward(tok, cam_enc, vmask):
      tok     (B, N, J, d_in)  feature sampled at the consensus 2D per (view,joint)
      cam_enc (B, N, cam_dim)  relative-camera encoding per view (center + axis)
      vmask   (B, N)           1.0 = real view, 0.0 = padded
    returns gate (B, N, J) in (0,1], zeroed on padded views.
    """

    def __init__(self, d_in, d=128, heads=4, layers=2, cam_dim=6):
        super().__init__()
        self.in_proj = nn.Linear(d_in + cam_dim, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.out = nn.Linear(d, 1)
        # Start near gate≈1 (trust all views) so the gate is a gentle modulation
        # of the existing solver, not an initial information bottleneck.
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, 2.0)             # sigmoid(2)≈0.88

    def forward(self, tok, cam_enc, vmask):
        B, N, J, _ = tok.shape
        cam = cam_enc.unsqueeze(2).expand(B, N, J, -1)
        x = self.in_proj(torch.cat([tok, cam], dim=-1))   # (B, N, J, d)
        # attention ACROSS views (N) per joint -> fold (B,J) into batch, N is the seq
        x = x.permute(0, 2, 1, 3).reshape(B * J, N, -1)   # (B*J, N, d)
        kpm = (vmask < 0.5)                               # (B, N) True = pad
        kpm = kpm.unsqueeze(1).expand(B, J, N).reshape(B * J, N)
        # every scene has >=1 real view, so no query row is fully masked.
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=kpm)
        g = torch.sigmoid(self.out(x)).reshape(B, J, N).permute(0, 2, 1)  # (B, N, J)
        return g * vmask.unsqueeze(-1)


_M3_MAX = 50.0   # bound on Cholesky-factor entries -> Ω3 eigenvalues bounded
_M3_EPS = 1e-2

def _chol3_entries(L):
    """Bounded lower-tri Cholesky factor entries from raw params (...,6).
    Diagonal in (eps, eps+MAX] via sigmoid (so log|Ω3| is bounded -> NLL can't
    collapse to -inf); off-diagonal in [-MAX, MAX] via tanh (so a pathological
    precision can't dominate/destabilize the GN solve)."""
    s, t = torch.sigmoid, torch.tanh
    m00 = _M3_EPS + _M3_MAX * s(L[..., 0]); m10 = _M3_MAX * t(L[..., 1])
    m11 = _M3_EPS + _M3_MAX * s(L[..., 2]); m20 = _M3_MAX * t(L[..., 3])
    m21 = _M3_MAX * t(L[..., 4]);           m22 = _M3_EPS + _M3_MAX * s(L[..., 5])
    return m00, m10, m11, m20, m21, m22


def chol3_to_prec(L):
    """(...,6) raw Cholesky params -> (...,3,3) SPD precision Ω3 = M·Mᵀ, M bounded.
    Lets a joint be anisotropically (un)certain (large covariance along depth)."""
    m00, m10, m11, m20, m21, m22 = _chol3_entries(L)
    z = torch.zeros_like(m00)
    M = torch.stack([torch.stack([m00, z, z], -1),
                     torch.stack([m10, m11, z], -1),
                     torch.stack([m20, m21, m22], -1)], -2)          # (...,3,3)
    return M @ M.transpose(-1, -2)


def gaussian_nll_3d(delta, L):
    """Per-joint 3D Gaussian NLL ½ δᵀΩ3 δ − ½ log|Ω3| with Ω3 = M·Mᵀ (bounded M).
    Trains the learned 3D mean (toward GT) and calibrates its 3x3 covariance."""
    m00, m10, m11, m20, m21, m22 = _chol3_entries(L)
    d0, d1, d2 = delta[..., 0], delta[..., 1], delta[..., 2]
    a = m00 * d0 + m10 * d1 + m20 * d2        # ‖Mᵀ δ‖² (Mᵀ upper-tri)
    b = m11 * d1 + m21 * d2
    c = m22 * d2
    quad = a * a + b * b + c * c
    logdet = 2.0 * (torch.log(m00) + torch.log(m11) + torch.log(m22))
    return 0.5 * quad - 0.5 * logdet


class CrossViewRefiner(nn.Module):
    """Cross-view transformer that outputs, per (view, joint), BOTH:
      - ``gate``  scalar in (0,1]      — epistemic trust (as :class:`CrossViewGate`),
      - ``corr``  2D delta (hm px)     — a bounded correction ADDED to the DOVF
                                          residual before the GN fit.

    The 2D correction is what lets the (scaled) transformer actually USE capacity:
    it does cross-view (epipolar) reasoning to refine the 2D evidence the fitter
    consumes, instead of only re-weighting it. ``out_corr`` is zero-initialised, so
    ``corr`` starts at 0 and the model begins identical to the gate-only variant
    (safe warm-start); the bound (``corr_max`` px via tanh) keeps early corrections
    from destabilising the solver. Same attention backbone as CrossViewGate, so it
    scales identically with ``d``/``layers``/``heads``.
    """

    def __init__(self, d_in, d=128, heads=4, layers=2, cam_dim=6, corr_max=6.0,
                 predict_trans=False, dtrans_max=0.10, predict_j3d=False, dmu_max=0.20):
        super().__init__()
        self.corr_max = float(corr_max)
        self.predict_trans = bool(predict_trans)
        self.dtrans_max = float(dtrans_max)                 # bound on the 3D root correction (m)
        self.predict_j3d = bool(predict_j3d)
        self.dmu_max = float(dmu_max)                       # bound on per-joint mean correction (m)
        self.in_proj = nn.Linear(d_in + cam_dim, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.out_gate = nn.Linear(d, 1)
        self.out_corr = nn.Linear(d, 2)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.constant_(self.out_gate.bias, 2.0)          # sigmoid(2)≈0.88 -> trust all
        nn.init.zeros_(self.out_corr.weight)
        nn.init.zeros_(self.out_corr.bias)                  # corr starts at 0 (identity)
        if self.predict_trans:
            self.out_trans = nn.Linear(d, 3)
            nn.init.zeros_(self.out_trans.weight); nn.init.zeros_(self.out_trans.bias)
        if self.predict_j3d:
            # per-joint learned 3D GAUSSIAN: a mean CORRECTION dmu (added to the
            # geometric init joints) + a 3x3 Cholesky precision. dmu zero-init ->
            # mean starts at the init joints; chol bias -> moderate initial precision.
            self.out_mu3d = nn.Linear(d, 3)
            self.out_chol3 = nn.Linear(d, 6)
            nn.init.zeros_(self.out_mu3d.weight); nn.init.zeros_(self.out_mu3d.bias)
            nn.init.zeros_(self.out_chol3.weight); nn.init.zeros_(self.out_chol3.bias)
            # start with LOW precision (weak 3D prior) so the still-bad initial mean
            # can't dominate the geometric fit; NLL sharpens it as the mean improves.
            with torch.no_grad():
                self.out_chol3.bias[0] = self.out_chol3.bias[2] = self.out_chol3.bias[5] = -3.0

    def forward(self, tok, cam_enc, vmask):
        B, N, J, _ = tok.shape
        cam = cam_enc.unsqueeze(2).expand(B, N, J, -1)
        x = self.in_proj(torch.cat([tok, cam], dim=-1))     # (B, N, J, d)
        x = x.permute(0, 2, 1, 3).reshape(B * J, N, -1)     # (B*J, N, d): attend across views
        kpm = (vmask < 0.5).unsqueeze(1).expand(B, J, N).reshape(B * J, N)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=kpm)
        x = x.reshape(B, J, N, -1).permute(0, 2, 1, 3)      # (B, N, J, d)
        gate = torch.sigmoid(self.out_gate(x)).squeeze(-1)  # (B, N, J)
        corr = self.corr_max * torch.tanh(self.out_corr(x))  # (B, N, J, 2) bounded px
        vm = vmask.unsqueeze(-1)
        aux = {}
        if self.predict_trans or self.predict_j3d:
            vm4 = vmask.view(B, N, 1, 1)
            pj = (x * vm4).sum(dim=1) / vm4.sum(dim=1).clamp(min=1.0)   # (B, J, d) per-joint pool
        if self.predict_trans:
            aux["dtrans"] = self.dtrans_max * torch.tanh(self.out_trans(pj.mean(dim=1)))  # (B,3)
        if self.predict_j3d:
            aux["dmu3d"] = self.dmu_max * torch.tanh(self.out_mu3d(pj))   # (B,J,3) bounded corr
            aux["L3d"] = self.out_chol3(pj)                               # (B,J,6) chol params
        return gate * vm, corr * vm.unsqueeze(-1), aux


def sample_tokens(feat_map, pts2d, h, w):
    """Bilinear-sample feat_map (BN, C, H, W) at pts2d (BN, J, 2) in (h,w)-grid px
    -> tokens (BN, J, C). Normalised grid for F.grid_sample."""
    BN, C = feat_map.shape[:2]
    J = pts2d.shape[1]
    gx = 2.0 * pts2d[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * pts2d[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(BN, J, 1, 2)            # (BN, J, 1, 2)
    samp = F.grid_sample(feat_map, grid, mode="bilinear",
                         align_corners=True, padding_mode="border")    # (BN, C, J, 1)
    return samp.squeeze(-1).permute(0, 2, 1).contiguous()             # (BN, J, C)


def camera_encoding(w2c):
    """w2c (BN, 4, 4) world->cam -> per-view encoding (BN, 6): camera centre +
    optical axis, both in world coords (the relative-camera signal for attention)."""
    R = w2c[:, :3, :3]                                   # (BN,3,3)
    t = w2c[:, :3, 3]                                    # (BN,3)
    Rt = R.transpose(1, 2)                               # cam->world rot
    centre = -torch.einsum("bij,bj->bi", Rt, t)          # (BN,3) camera centre in world
    axis = Rt[:, :, 2]                                   # (BN,3) optical axis in world
    return torch.cat([centre, axis], dim=-1)             # (BN,6)
