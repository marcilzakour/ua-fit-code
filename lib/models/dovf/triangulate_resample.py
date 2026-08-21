"""Monte-Carlo 3D head: DOVF uplift -> per-joint 3D Gaussian (single component =
the 2-view mixture); draw MC particles that DESCRIBE it; resample backbone features
at each particle (projected into every view); a small transformer scores the
particles by appearance -> POSTERIOR weights. Output = the weighted particle set
(no collapse). Trained with loss_mc = Σ_k w_k ‖X_k − GT‖².

Camera convention: w2c = [R|t] (world->cam); consensus & K are in heatmap pixels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .uncertainty_modules import camera_encoding


def _pad_by_scene(x, cvn, Nmax):
    B = len(cvn); out = x.new_zeros(B, Nmax, *x.shape[1:])
    vmask = torch.zeros(B, Nmax, device=x.device)
    for i in range(B):
        s = int(sum(cvn[:i])); e = int(sum(cvn[:i + 1])); n = e - s
        out[i, :n] = x[s:e]; vmask[i, :n] = 1.0
    return out, vmask


def triangulate(cons_pad, K_pad, w2c_pad, vmask, eps=1e-3, sigma_px=None):
    """Multi-view ray-intersection per joint. Returns mu (B,J,3), Sigma3 (B,J,3,3).

    If ``sigma_px`` (B,N,J) — the per-(view,joint) 2D localization std in heatmap
    px — is given, each ray is weighted by 1/sigma_perp_v^2 with the *metric*
    perpendicular uncertainty sigma_perp_v = (sigma_px_v / f_v) * depth_v. Then
    Sigma3 = A^-1 is a real-metric covariance (m^2), correctly depth-anisotropic.
    Without it, rays are unit-weighted (Sigma3 is geometric shape only, no scale).
    """
    B, N, J, _ = cons_pad.shape
    R = w2c_pad[..., :3, :3]; t = w2c_pad[..., :3, 3]
    Rt = R.transpose(-1, -2)
    C = -torch.einsum("bnij,bnj->bni", Rt, t)                          # cam centre (B,N,3)
    uvh = torch.cat([cons_pad, torch.ones_like(cons_pad[..., :1])], -1)
    # Variable view counts pad missing slots with ZERO K (singular). Replace padded slots with
    # identity so inv() is well-defined; their rays are masked out by vmask in the sums below.
    eye3 = torch.eye(3, device=K_pad.device, dtype=K_pad.dtype)
    K_safe = torch.where(vmask[..., None, None] > 0, K_pad, eye3)
    d = torch.einsum("bnij,bnkj->bnki", Rt, torch.einsum("bnij,bnkj->bnki",
                     torch.linalg.inv(K_safe), uvh))                   # ray dir (world) (B,N,J,3)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    M = torch.eye(3, device=d.device) - d.unsqueeze(-1) * d.unsqueeze(-2)   # (B,N,J,3,3)
    eye = torch.eye(3, device=d.device)

    def _solve(weight):                                                # weight (B,N,J) or None
        if weight is None:
            wM = M; vmM = vmask.view(B, N, 1, 1, 1)
        else:
            wM = M * weight[..., None, None]; vmM = vmask.view(B, N, 1, 1, 1)
        A = (vmM * wM).sum(1) + eps * eye                              # (B,J,3,3)
        MC = torch.einsum("bnjac,bnc->bnja", wM, C)
        bvec = (vmask.view(B, N, 1, 1) * MC).sum(1)                    # (B,J,3)
        Sig = torch.linalg.inv(A)
        return torch.einsum("bjac,bjc->bja", Sig, bvec), Sig

    mu, Sigma3 = _solve(None)                                          # unweighted first pass
    if sigma_px is not None:
        # metric per-ray weight from a depth estimate at the unweighted mu
        Xc = torch.einsum("bnij,bkj->bnki", R, mu) + t[:, :, None, :]   # (B,N,J,3) cam coords
        Xc_z = Xc[..., 2]                                              # (B,N,J) depth
        f = K_pad[..., 0, 0].clamp(min=1.0)[:, :, None]               # focal hm-px (B,N,1)
        sig_perp = (sigma_px / f) * Xc_z.abs().clamp(min=1e-3)         # (B,N,J) metres
        lam = 1.0 / sig_perp.clamp(min=1e-4).pow(2)                    # (B,N,J) 1/m^2
        mu, Sigma3 = _solve(lam)                                       # metric solve
    return mu, Sigma3


def triangulate_omega2(cons_pad, K_pad, w2c_pad, omega2_pad, vmask, n_iter=2, eps=1e-4,
                       robust=False, robust_iter=4, robust_c2=25.0, robust_kind="cauchy",
                       master_w=3.0):
    """Uncertainty-weighted Gauss-Newton triangulation using the model's predicted
    anisotropic 2D precision Omega2 (B,N,J,2,2). Minimizes Σ_v r_vᵀ Ω2_v r_v over the
    reprojection residual r_v. Returns mu (B,J,3) and a metric, anisotropic
    Sigma3 (B,J,3,3) = (Σ_v Jᵀ_v Ω2_v J_v)^-1 — the proper triangulation covariance.

    robust: after ``n_iter`` plain GN steps, run ``robust_iter`` IRLS steps that
    reweight each (view,joint) by a redescending function of its Mahalanobis
    reprojection residual ρ = rᵀΩ2 r (≈χ²₂ for an inlier). A wrong-hand / occluded
    view (InterHand two-hand ambiguity) has ρ orders of magnitude larger → its weight
    →0, so the consistent views win (needs ≥3 views; can't disambiguate 2v). Inference
    only; no new params. ``robust_c2`` is the cutoff in Mahalanobis² units.
    """
    B, N, J, _ = cons_pad.shape
    R = w2c_pad[..., :3, :3]; t = w2c_pad[..., :3, 3]
    fx = K_pad[..., 0, 0, None]; fy = K_pad[..., 1, 1, None]                   # (B,N,1)
    cx = K_pad[..., 0, 2, None]; cy = K_pad[..., 1, 2, None]
    eye = torch.eye(3, device=cons_pad.device)
    mu, _ = triangulate(cons_pad, K_pad, w2c_pad, vmask)                       # unweighted init
    vm = vmask.view(B, N, 1, 1, 1)
    total_iter = n_iter + (robust_iter if robust else 0)
    Sigma3 = None
    for it in range(total_iter):
        Xc = torch.einsum("bnij,bkj->bnki", R, mu) + t[:, :, None, :]          # (B,N,J,3)
        x, y = Xc[..., 0], Xc[..., 1]; z = Xc[..., 2].clamp(min=1e-4)
        uvp = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)              # (B,N,J,2)
        Juvc = torch.zeros(B, N, J, 2, 3, device=cons_pad.device)             # d uv / d Xc
        Juvc[..., 0, 0] = fx / z; Juvc[..., 0, 2] = -fx * x / (z * z)
        Juvc[..., 1, 1] = fy / z; Juvc[..., 1, 2] = -fy * y / (z * z)
        Jac = torch.einsum("bnjac,bncd->bnjad", Juvc, R)                       # d uv / d X (B,N,J,2,3)
        r = cons_pad - uvp                                                    # (B,N,J,2)
        Om = omega2_pad
        if robust and it >= n_iter:
            rho = torch.einsum("bnja,bnjac,bnjc->bnj", r, omega2_pad, r).clamp(min=0.0)  # (B,N,J)
            if robust_kind == "master":
                # WHOLE-HAND, master-anchored rejection (InterHand two-hand fix). A wrong-hand view
                # has ALL 21 joints offset -> its per-VIEW aggregate residual is huge -> reject the
                # whole view. View 0 is the master (target_cam_extr=I, crop centred on target hand)
                # -> pin it to a fixed high weight so mu anchors to the TRUE hand even when a
                # self-consistent pair of views agrees on the wrong one (which per-joint IRLS can't fix).
                rho_v = rho.mean(dim=2, keepdim=True)                          # (B,N,1) per-view
                wr = (robust_c2 / (robust_c2 + rho_v)).expand(-1, -1, rho.shape[2]).clone()
                wr[:, 0, :] = master_w                                         # master = fixed anchor
            elif robust_kind == "huber":
                wr = torch.clamp(robust_c2 / rho.clamp(min=1e-6), max=1.0)     # min(1, c²/ρ)
            else:  # cauchy/lorentzian — smooth, stable, strongly downweights gross outliers
                wr = robust_c2 / (robust_c2 + rho)
            Om = omega2_pad * wr[..., None, None]
        JtO = torch.einsum("bnjca,bnjcd->bnjad", Jac, Om)                     # Jᵀ Ω (B,N,J,3,2)
        H = torch.einsum("bnjad,bnjde->bnjae", JtO, Jac)                       # (B,N,J,3,3)
        g = torch.einsum("bnjad,bnjd->bnja", JtO, r)                          # (B,N,J,3)
        # N=1 / extreme-seed robustness: inf/NaN in Jac/Omega defeats the +eps*I guard
        # (a NaN row is singular regardless), and at 1 view H is rank-2 by construction.
        # Sanitize + scale-aware Tikhonov so the solve NEVER hard-fails; degenerate scenes
        # yield a finite (depth-ambiguous) mu that Sigma3 flags via a huge along-ray var.
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        A = (vm * H).sum(1)                                                    # (B,J,3,3)
        trA = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).clamp(min=0.0)       # (B,J)
        A = A + (eps + 1e-4 * trA[..., None, None] / 3.0) * eye
        gg = (vmask.view(B, N, 1, 1) * g).sum(1)                              # (B,J,3)
        Sigma3 = torch.linalg.inv(A)
        mu = mu + torch.einsum("bjac,bjc->bja", Sigma3, gg)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    return mu, Sigma3


class MixtureMCHead(nn.Module):
    def __init__(self, feat_dim, d=256, heads=8, layers=3, KS=16, num_joints=21,
                 sigma_min=0.001, sigma_max=0.05, spread_scale=2.0, consistency=False):
        super().__init__()
        self.KS = 7; self.J = num_joints                          # 7 = 2·3+1 sigma-points
        self.sigma_min = sigma_min; self.sigma_max = sigma_max    # spread bounds (m)
        self.spread_scale = spread_scale
        # consistency: append the cross-view feature-variance at each sigma-point (high=views agree
        # this is the joint) — the depth-disambiguation cue the mean-pooled feature throws away.
        self.consistency = consistency
        self.in_proj = nn.Linear(feat_dim + (1 if consistency else 0) + 3 + 1, d)  # feat[+cons]+ΔX(3)+logπ(1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True) for _ in range(layers)])
        self.score = nn.Linear(d, 1)
        nn.init.normal_(self.score.weight, std=0.01); nn.init.zeros_(self.score.bias)  # near-uniform but grad flows

    def _resample(self, X, feat_map, K_hm, w2c, cvn, Nmax, vmask):
        B, J, KS, _ = X.shape; BN, C, h, w = feat_map.shape
        X_bn = torch.cat([X[i].unsqueeze(0).expand(int(cvn[i]), -1, -1, -1)
                          for i in range(B)], 0)                        # (BN,J,KS,3)
        R = w2c[:, :3, :3]; t = w2c[:, :3, 3]
        Xc = torch.einsum("bij,bkmj->bkmi", R, X_bn) + t[:, None, None, :]   # (BN,J,KS,3) cam
        z = Xc[..., 2:3].clamp(min=1e-4)
        uv = torch.einsum("bij,bkmj->bkmi", K_hm, Xc)[..., :2] / z      # (BN,J,KS,2) hm-px
        grid = torch.stack([2 * uv[..., 0] / (w - 1) - 1,
                            2 * uv[..., 1] / (h - 1) - 1], -1)          # (BN,J,KS,2)
        samp = F.grid_sample(feat_map, grid.reshape(BN, J * KS, 1, 2),
                             mode="bilinear", align_corners=True, padding_mode="border")
        feat_bn = samp.squeeze(-1).reshape(BN, C, J, KS).permute(0, 2, 3, 1)   # (BN,J,KS,C)
        feat_pad, _ = _pad_by_scene(feat_bn, cvn, Nmax)                 # (B,Nmax,J,KS,C)
        vm = vmask.view(B, Nmax, 1, 1, 1)
        nv = vm.sum(1).clamp(min=1.0)
        mean_f = (feat_pad * vm).sum(1) / nv                            # (B,J,KS,C)
        if not self.consistency:
            return mean_f
        var_f = ((feat_pad - mean_f[:, None]) ** 2 * vm).sum(1) / nv    # (B,J,KS,C) cross-view var
        cons = -var_f.mean(-1, keepdim=True)                            # (B,J,KS,1) high = views agree
        return torch.cat([mean_f, cons], -1)                           # (B,J,KS,C+1)

    def forward(self, cons2d, feat_map, K_hm, w2c, cvn, sigma2d=None, omega2=None):
        B = len(cvn); Nmax = int(max(cvn)); J = self.J; KS = self.KS
        cons_pad, vmask = _pad_by_scene(cons2d, cvn, Nmax)
        K_pad, _ = _pad_by_scene(K_hm, cvn, Nmax)
        w2c_pad, _ = _pad_by_scene(w2c, cvn, Nmax)

        if omega2 is not None:                                         # preferred: learned 2D precision
            om_pad, _ = _pad_by_scene(omega2, cvn, Nmax)               # (B,Nmax,J,2,2)
            mu, Sigma3 = triangulate_omega2(cons_pad, K_pad, w2c_pad, om_pad, vmask)
        else:
            sig_pad = None
            if sigma2d is not None:
                sig_pad, _ = _pad_by_scene(sigma2d, cvn, Nmax)
            mu, Sigma3 = triangulate(cons_pad, K_pad, w2c_pad, vmask, sigma_px=sig_pad)
        # DETERMINISTIC sigma-points (unscented): symmetric -> uniform weights give
        # exactly mu (no MC readout noise), and the same points every forward (no
        # train/eval mismatch). 7 points = {mu, mu ± s·√λ_i·q_i}.
        ev, Q = torch.linalg.eigh(Sigma3)                              # (B,J,3),(B,J,3,3)
        ev = ev.clamp(self.sigma_min ** 2, self.sigma_max ** 2)
        sd = ev.sqrt() * self.spread_scale                             # (B,J,3) per-axis spread
        O = (Q * sd.unsqueeze(-2)).transpose(-1, -2)                   # (B,J,3,3) offset vectors (rows)
        mu0 = mu[:, :, None, :]
        X = torch.cat([mu0, mu0 + O, mu0 - O], dim=2)                 # (B,J,7,3)
        dX = X - mu0                                                   # (B,J,7,3)
        # mahalanobis prior: center 0, the 6 axis points = -0.5·scale²
        Sinv = torch.linalg.inv(Sigma3)
        logpi = -0.5 * torch.einsum("bjka,bjac,bjkc->bjk", dX, Sinv, dX)   # (B,J,7)
        KS = X.shape[2]

        feat = self._resample(X, feat_map, K_hm, w2c, cvn, Nmax, vmask)   # (B,J,KS,C)
        tok = self.in_proj(torch.cat([dX, feat, logpi[..., None]], -1))  # (B,J,KS,d)
        x = tok.reshape(B * J, KS, -1)
        for layer in self.layers:
            x = layer(x)
        logit = self.score(x).reshape(B, J, KS)
        w = torch.softmax(logit + logpi, dim=-1)                       # posterior weights
        mu_hat = (w[..., None] * X).sum(2)                             # (B,J,3) readout mean
        return mu_hat, X, w, mu                                        # X,w kept (no collapse); mu=triangulation


class CostVolumeHead(nn.Module):
    """Depth cost-volume head (drop-in for MixtureMCHead). Instead of 7 moment-matched
    sigma-points, samples D points DENSELY along the depth-uncertain axis (top eigvec of
    Sigma3) and scores each by (a) appearance and (b) CROSS-VIEW CONSISTENCY (the classic
    plane-sweep cost). Captures multi-modal depth the single Gaussian can't. The scorer's
    softmax over D depths is the depth posterior; readout = Σ w X (no collapse)."""
    def __init__(self, feat_dim, d=256, heads=8, layers=3, KS=48, num_joints=21,
                 depth_sigmas=3.0, sigma_min=0.004, sigma_max=0.06):
        super().__init__()
        self.D = KS; self.J = num_joints
        self.depth_sigmas = depth_sigmas                  # sample ±depth_sigmas·σ along depth
        self.sigma_min = sigma_min; self.sigma_max = sigma_max
        self.in_proj = nn.Linear(feat_dim + 1 + 1 + 1, d)  # mean_feat + Δdepth + consistency + logπ
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True) for _ in range(layers)])
        self.score = nn.Linear(d, 1)
        nn.init.normal_(self.score.weight, std=0.01); nn.init.zeros_(self.score.bias)

    def _resample_perview(self, X, feat_map, K_hm, w2c, cvn, Nmax, vmask):
        """Return per-view features (B,Nmax,J,D,C) + padded mask -> for mean & consistency."""
        B, J, D, _ = X.shape; BN, C, h, w = feat_map.shape
        X_bn = torch.cat([X[i].unsqueeze(0).expand(int(cvn[i]), -1, -1, -1) for i in range(B)], 0)
        R = w2c[:, :3, :3]; t = w2c[:, :3, 3]
        Xc = torch.einsum("bij,bkmj->bkmi", R, X_bn) + t[:, None, None, :]
        z = Xc[..., 2:3].clamp(min=1e-4)
        uv = torch.einsum("bij,bkmj->bkmi", K_hm, Xc)[..., :2] / z
        grid = torch.stack([2 * uv[..., 0] / (w - 1) - 1, 2 * uv[..., 1] / (h - 1) - 1], -1)
        samp = F.grid_sample(feat_map, grid.reshape(BN, J * D, 1, 2), mode="bilinear",
                             align_corners=True, padding_mode="border")
        feat_bn = samp.squeeze(-1).reshape(BN, C, J, D).permute(0, 2, 3, 1)   # (BN,J,D,C)
        feat_pad, _ = _pad_by_scene(feat_bn, cvn, Nmax)                       # (B,Nmax,J,D,C)
        return feat_pad

    def forward(self, cons2d, feat_map, K_hm, w2c, cvn, sigma2d=None, omega2=None):
        B = len(cvn); Nmax = int(max(cvn)); J = self.J; D = self.D
        cons_pad, vmask = _pad_by_scene(cons2d, cvn, Nmax)
        K_pad, _ = _pad_by_scene(K_hm, cvn, Nmax)
        w2c_pad, _ = _pad_by_scene(w2c, cvn, Nmax)
        om_pad, _ = _pad_by_scene(omega2, cvn, Nmax)
        mu, Sigma3 = triangulate_omega2(cons_pad, K_pad, w2c_pad, om_pad, vmask)   # (B,J,3),(B,J,3,3)

        # depth axis = largest-variance eigenvector of Sigma3; sample D points along it
        ev, Q = torch.linalg.eigh(Sigma3)                              # ascending
        sig_d = ev[..., -1].clamp(self.sigma_min ** 2, self.sigma_max ** 2).sqrt()   # (B,J) depth std
        q_d = Q[..., -1]                                              # (B,J,3) depth direction
        zc = torch.linspace(-self.depth_sigmas, self.depth_sigmas, D, device=mu.device)  # (D,)
        dz = zc[None, None, :] * sig_d[:, :, None]                    # (B,J,D) metric depth offset
        X = mu[:, :, None, :] + dz[..., None] * q_d[:, :, None, :]    # (B,J,D,3)
        logpi = -0.5 * (zc ** 2)[None, None, :].expand(B, J, D)       # gaussian depth prior

        fpv = self._resample_perview(X, feat_map, K_hm, w2c, cvn, Nmax, vmask)   # (B,Nmax,J,D,C)
        vm = vmask.view(B, Nmax, 1, 1, 1)
        nv = vm.sum(1).clamp(min=1.0)
        mean_f = (fpv * vm).sum(1) / nv                              # (B,J,D,C)
        var_f = ((fpv - mean_f[:, None]) ** 2 * vm).sum(1) / nv      # (B,J,D,C)
        consistency = -var_f.mean(-1)                                # (B,J,D) high = views agree
        tok = self.in_proj(torch.cat([mean_f, dz[..., None], consistency[..., None],
                                      logpi[..., None]], -1))         # (B,J,D,d)
        x = tok.reshape(B * J, D, -1)
        for layer in self.layers:
            x = layer(x)
        logit = self.score(x).reshape(B, J, D)
        w = torch.softmax(logit + logpi, dim=-1)                     # depth posterior
        mu_hat = (w[..., None] * X).sum(2)                           # (B,J,3)
        return mu_hat, X, w, mu
