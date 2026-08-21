"""
Learned 3D hand-structure prior (Phase D).
==========================================

The cross-view APPEARANCE signal for depth refinement isn't realizable in-scope (oracle ≈ chance
for features and raw image; param-free heatmap gain = 0). But the other source of information at
low views — the **hand-pose manifold** — is strong and learnable. At 2 views the triangulated joints
have wrong DEPTHS → an implausible hand shape; a network that knows hand structure can pull the
depth-uncertain joints back toward a plausible configuration, while leaving confident (high-view)
joints alone.

``PoseRefiner3D`` is a small transformer over the J=21 joints (learned per-joint embeddings encode
skeleton identity), conditioned on the triangulated mean AND the per-joint uncertainty Σ3 (so it
knows WHICH joints to trust vs. fix). It emits a bounded residual on μ, trained end-to-end with the
3D loss. Uncertainty-conditioning makes it view-count-robust by construction: low Σ3 (many views) →
small correction; high Σ3 (2 views, depth-ambiguous) → larger structure-guided correction. The 3D
fusion (triangulation) stays parameter-free; this is a learned READOUT prior, not a fusion change.
"""

import torch
import torch.nn as nn


class PoseRefiner3D(nn.Module):
    def __init__(self, num_joints=21, center_idx=0, d=256, heads=8, layers=4,
                 max_corr=0.05, iters=1, unc_gate=False):
        super().__init__()
        self.J = num_joints; self.center_idx = center_idx
        self.max_corr = float(max_corr); self.iters = int(iters)
        # unc_gate: scale each joint's correction by its OWN uncertainty σ (≤ max_corr), so a
        # confident (high-view) joint can only be nudged within its error bar -> no over-correction.
        self.unc_gate = bool(unc_gate)
        # input per joint: abs μ (3) + root-relative μ (3) + log per-joint uncertainty (1)
        self.in_proj = nn.Linear(7, d)
        self.joint_emb = nn.Parameter(torch.zeros(1, num_joints, d))
        nn.init.normal_(self.joint_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity

    def forward(self, mu, Sigma3, return_iters=False):
        """mu (B,J,3), Sigma3 (B,J,3,3) -> refined mu (B,J,3) [+ list of per-iter μ]."""
        mu_iters = [mu]
        cur = mu
        # per-joint scalar uncertainty = total variance (trace Σ3)
        var = torch.diagonal(Sigma3, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)  # (B,J)
        unc = (var.log() + 6.0) * 0.2                                            # ~O(1) input feature
        sigma = (var / 3.0).sqrt()                                              # (B,J) per-joint std (m)
        for _ in range(self.iters):
            root = cur[:, self.center_idx:self.center_idx + 1]
            rel = cur - root                                                     # root-relative
            tok = self.in_proj(torch.cat([cur, rel, unc[..., None]], dim=-1)) + self.joint_emb
            x = self.enc(tok)
            if self.unc_gate:                                                   # σ-scaled: ≤ each joint's own error bar
                bound = sigma.clamp(max=self.max_corr)[..., None]              # (B,J,1)
            else:
                bound = self.max_corr
            dmu = bound * torch.tanh(self.out(x))                               # bounded residual
            cur = cur + dmu
            mu_iters.append(cur)
        if return_iters:
            return cur, mu_iters
        return cur


class PoseRefiner3DMV(nn.Module):
    """POEM-lite: learned cross-view fusion refiner.

    PoseRefiner3D only sees the triangulation *summary* (μ, Σ3). POEM's edge is that its 94M head
    fuses the per-view evidence with a learned network instead of a closed-form solve. This module
    adds exactly that — but using GEOMETRY ONLY (cross-view appearance matching is proven dead:
    epipolar oracle ≈ chance). For every (view, joint) it builds a token from purely geometric /
    statistical cues — ray direction, camera offset from μ, the *reprojection residual* of the
    current μ in that view (the per-view disagreement signal), and the 2D log-precision — then
    cross-view-attends them per joint into the joint-structure transformer. It learns the nonlinear
    corrections + low-view depth priors that closed-form triangulation can't, at a few M params.
    Recomputes evidence each iter so the net sees how its own correction changes cross-view agreement.
    """

    def __init__(self, num_joints=21, center_idx=0, d=256, heads=8, layers=4,
                 max_corr=0.05, iters=2, unc_gate=False, ev_dim=64):
        super().__init__()
        self.J = num_joints; self.center_idx = center_idx
        self.max_corr = float(max_corr); self.iters = int(iters); self.unc_gate = bool(unc_gate)
        # per-(view,joint) evidence token: ray(3)+camoff_dir(3)+camdist(1)+reproj_r(2)+log2Dprec(2)=11
        self.ev_in = nn.Sequential(nn.Linear(11, ev_dim), nn.GELU(), nn.Linear(ev_dim, d))
        # zero the evidence projection -> ev tokens (the attn VALUES) start at 0 -> ev=0 -> the MV
        # refiner begins EXACTLY as the base PoseRefiner3D (warm-start from poseauxfull is identity),
        # then learns the cross-view correction. Still trainable (grad flows via the attn values).
        nn.init.zeros_(self.ev_in[-1].weight); nn.init.zeros_(self.ev_in[-1].bias)
        self.ev_query = nn.Parameter(torch.zeros(1, num_joints, d)); nn.init.normal_(self.ev_query, std=0.02)
        self.ev_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.in_proj = nn.Linear(7, d)
        self.joint_emb = nn.Parameter(torch.zeros(1, num_joints, d)); nn.init.normal_(self.joint_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity

    def _evidence(self, cur, cons_pad, K_pad, w2c_pad, om_pad, vmask):
        B, N, J, _ = cons_pad.shape
        R = w2c_pad[..., :3, :3]; t = w2c_pad[..., :3, 3]
        Rt = R.transpose(-1, -2)
        C = -torch.einsum("bnij,bnj->bni", Rt, t)                                   # cam centres (B,N,3)
        eye3 = torch.eye(3, device=K_pad.device, dtype=K_pad.dtype)
        K_safe = torch.where(vmask[..., None, None] > 0, K_pad, eye3)
        uvh = torch.cat([cons_pad, torch.ones_like(cons_pad[..., :1])], -1)         # (B,N,J,3)
        d_cam = torch.einsum("bnij,bnkj->bnki", torch.linalg.inv(K_safe), uvh)
        d_world = torch.einsum("bnij,bnkj->bnki", Rt, d_cam)
        d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp(min=1e-6)       # ray dir (B,N,J,3)
        camoff = C[:, :, None, :] - cur[:, None, :, :]                              # (B,N,J,3)
        camdist = camoff.norm(dim=-1, keepdim=True)                                 # (B,N,J,1)
        camoff_n = camoff / camdist.clamp(min=1e-6)
        Xc = torch.einsum("bnij,bkj->bnki", R, cur) + t[:, :, None, :]              # (B,N,J,3)
        z = Xc[..., 2].clamp(min=1e-4)
        fx = K_pad[..., 0, 0, None]; fy = K_pad[..., 1, 1, None]                    # (B,N,1)
        cx = K_pad[..., 0, 2, None]; cy = K_pad[..., 1, 2, None]
        uvp = torch.stack([fx * Xc[..., 0] / z + cx, fy * Xc[..., 1] / z + cy], -1)  # (B,N,J,2)
        r = (cons_pad - uvp) * 0.01                                                 # reproj residual (scaled)
        logprec = torch.diagonal(om_pad, dim1=-2, dim2=-1).clamp(min=1e-6).log() * 0.1  # (B,N,J,2)
        feat = torch.cat([d_world, camoff_n, camdist.clamp(max=2.0), r, logprec], -1)   # (B,N,J,11)
        tok = self.ev_in(feat)                                                      # (B,N,J,d)
        tok = tok.permute(0, 2, 1, 3).reshape(B * J, N, -1)                         # (B*J,N,d)
        q = self.ev_query.expand(B, -1, -1).reshape(B * J, 1, -1)                   # (B*J,1,d)
        keymask = (vmask[:, None, :].expand(B, J, N).reshape(B * J, N) < 0.5)       # True=pad
        ev, _ = self.ev_attn(q, tok, tok, key_padding_mask=keymask)                 # (B*J,1,d)
        return ev.reshape(B, J, -1)                                                 # (B,J,d)

    def forward(self, mu, Sigma3, cons_pad, K_pad, w2c_pad, om_pad, vmask, return_iters=False):
        mu_iters = [mu]; cur = mu
        var = torch.diagonal(Sigma3, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)     # (B,J)
        unc = (var.log() + 6.0) * 0.2
        sigma = (var / 3.0).sqrt()
        for _ in range(self.iters):
            ev = self._evidence(cur, cons_pad, K_pad, w2c_pad, om_pad, vmask)       # (B,J,d)
            root = cur[:, self.center_idx:self.center_idx + 1]
            rel = cur - root
            tok = self.in_proj(torch.cat([cur, rel, unc[..., None]], dim=-1)) + self.joint_emb + ev
            x = self.enc(tok)
            bound = sigma.clamp(max=self.max_corr)[..., None] if self.unc_gate else self.max_corr
            dmu = bound * torch.tanh(self.out(x))
            cur = cur + dmu
            mu_iters.append(cur)
        if return_iters:
            return cur, mu_iters
        return cur
