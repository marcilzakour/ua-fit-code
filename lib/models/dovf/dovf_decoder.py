"""
Neck + heads for Dense Offset Vector Field (DOVF) prediction.

Components
----------
MultiResNeck         Top-down FPN that fuses the backbone's multi-resolution
                     feature maps into a feat_dim pyramid (high-res -> low-res).
HeatmapHead          Per-point 2D heatmap head on the finest pyramid level
                     (soft-argmax point estimate + softmax probabilities).
CoarseToFineDOVFHead Predicts a dense offset field per point by coarse-to-fine
                     residual refinement across the pyramid: a coarse offset
                     field at the lowest resolution is upsampled and additively
                     corrected at each finer scale.  Offsets are in
                     heatmap-pixel units (resolution independent).
ManoInitHead         Regresses a coarse MANO axis-angle pose + shape from a
                     pooled global descriptor (the optimizer refines it).

Plus two functional helpers:
dovf_consensus_2d    Dense-voting consensus: every pixel votes for the point's
                     location (pixel + offset); votes are aggregated weighted by
                     heatmap probability -> a robust 2D point + confidence.
build_dovf_target    Ground-truth offset field for a given spatial resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Neck
# ─────────────────────────────────────────────────────────────────────────────

class MultiResNeck(nn.Module):
    """Top-down FPN over backbone features (ordered high-res -> low-res).

    Args:
        in_channels_list: channels of each backbone scale, high-res first.
        feat_dim:         unified output channel count for every level.
    """

    def __init__(self, in_channels_list, feat_dim=128, extra_fine=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, feat_dim, kernel_size=1) for c in in_channels_list]
        )
        self.smooth = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, feat_dim), feat_dim),
                nn.ReLU(inplace=True),
            ) for _ in in_channels_list]
        )
        # Optional genuine higher-res level: learnably upsample the finest fused
        # level by 2× (e.g. stride-4 64² -> stride-2 128²) and PREPEND it as the new
        # finest pyramid level, so the heatmap/DOVF heads localize on a finer grid.
        self.extra_fine = extra_fine
        if extra_fine:
            self.fine_up = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, feat_dim), feat_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        """feats: list high-res->low-res. Returns pyramid high-res->low-res."""
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        # Top-down: start at coarsest, add upsampled into each finer level.
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i] = laterals[i] + up
        out = [self.smooth[i](laterals[i]) for i in range(len(laterals))]
        if self.extra_fine:
            up = F.interpolate(out[0], scale_factor=2, mode="bilinear", align_corners=False)
            out = [self.fine_up(up)] + out                          # new finest level (2× res)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapHead(nn.Module):
    """Per-point heatmap head -> softmax probs + soft-argmax coords (heatmap px)."""

    def __init__(self, feat_dim, num_points, hm_size):
        super().__init__()
        self.num_points = num_points
        self.hm_h, self.hm_w = hm_size
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, num_points, kernel_size=1),
        )

    def forward(self, feat):
        logits = self.conv(feat)                                  # (B, P, h, w)
        if logits.shape[-2:] != (self.hm_h, self.hm_w):
            logits = F.interpolate(logits, size=(self.hm_h, self.hm_w),
                                   mode="bilinear", align_corners=False)
        b, p, h, w = logits.shape
        probs = F.softmax(logits.flatten(2), dim=-1).view(b, p, h, w)
        xs = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)                      # (h, w, 2)
        coords = torch.einsum("bphw,hwc->bpc", probs, grid)       # (B, P, 2)
        return logits, probs, coords


class _DilatedContext(nn.Module):
    """Widen the receptive field with a residual dilated-conv stack so a pixel's
    offset prediction can gather long-range context (where the joint is)."""

    def __init__(self, dim, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(min(32, dim), dim), nn.ReLU(inplace=True),
            ) for d in dilations])

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


def _sincos_2d(h, w, dim, device, dtype):
    """Parameter-free 2D sin-cos positional embedding -> (h*w, dim)."""
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    d4 = max(dim // 4, 1)
    omega = 1.0 / (10000 ** (torch.arange(d4, device=device, dtype=dtype) / d4))

    def emb(p):
        o = p.flatten()[:, None] * omega[None, :]
        return torch.cat([o.sin(), o.cos()], dim=1)

    pe = torch.cat([emb(gx), emb(gy)], dim=1)
    if pe.shape[1] < dim:
        pe = torch.cat([pe, pe.new_zeros(pe.shape[0], dim - pe.shape[1])], dim=1)
    return pe[:, :dim]


class _AttnContext(nn.Module):
    """Global self-attention over a (small) grid: every location attends everywhere,
    so far pixels can see the joint. Cheap only at coarse resolutions."""

    def __init__(self, dim, heads=4, layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                                           batch_first=True, dropout=0.0, activation="gelu",
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        b, c, h, w = x.shape
        pe = _sincos_2d(h, w, c, x.device, x.dtype).unsqueeze(0)
        t = x.flatten(2).transpose(1, 2) + pe                # (B, HW, C)
        t = self.tr(t)
        return t.transpose(1, 2).reshape(b, c, h, w) + x


def _build_dovf_contexts(dim, n_levels, kind):
    """Per-level context modules (index 0 = coarsest). kind:
       none/c2f  : identity (current head)
       dilated   : dilated-conv context at every level (wide receptive field)
       attn      : self-attention at the 2 coarsest grids, dilated at finer (cheap)"""
    if kind in (None, "none", "c2f"):
        return nn.ModuleList([nn.Identity() for _ in range(n_levels)])
    if kind in ("dilated", "wide"):
        return nn.ModuleList([_DilatedContext(dim) for _ in range(n_levels)])
    if kind == "attn":
        return nn.ModuleList([_AttnContext(dim) if lvl < 2 else _DilatedContext(dim)
                              for lvl in range(n_levels)])
    raise ValueError(f"unknown DOVF_HEAD_CONTEXT '{kind}' (none|dilated|attn)")


class CoarseToFineDOVFHead(nn.Module):
    """Coarse-to-fine dense offset field, refined across the pyramid.

    Returns the final field at heatmap resolution plus the per-scale fields
    (lowest-res first) for optional deep supervision.  All offsets are in
    heatmap-pixel units. `context` adds a per-level receptive-field/attention
    module to gather long-range info before each offset head.
    """

    def __init__(self, feat_dim, num_points, hm_size, n_levels, context="none"):
        super().__init__()
        self.num_points = num_points
        self.hm_h, self.hm_w = hm_size
        self.out_ch = num_points * 2
        self.feat_dim = feat_dim
        self._n_levels = n_levels

        # Per-level context (identity by default; dilated/attention to widen RF).
        self.contexts = _build_dovf_contexts(feat_dim, n_levels, context)

        # Coarse head (applied at the lowest-resolution pyramid level).
        self.coarse = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, self.out_ch, kernel_size=1),
        )
        # One residual head per finer level (all but the coarsest). Built
        # eagerly so every parameter exists before DDP wrap / optimizer build.
        self.res_heads = nn.ModuleList()
        for _ in range(n_levels - 1):
            head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dim, self.out_ch, kernel_size=1),
            )
            # Zero-init the residual output so initial field == coarse upsample.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.res_heads.append(head)

    def forward(self, pyramid):
        """pyramid: list high-res->low-res of (B, feat_dim, h, w)."""
        assert len(pyramid) == self._n_levels, \
            f"expected {self._n_levels} pyramid levels, got {len(pyramid)}"

        # coarsest -> finest
        levels = pyramid[::-1]
        field = self.coarse(self.contexts[0](levels[0]))    # (B, out_ch, h0, w0)
        per_scale = [field]
        for i in range(1, len(levels)):
            field_up = F.interpolate(field, size=levels[i].shape[-2:],
                                     mode="bilinear", align_corners=False)
            field = field_up + self.res_heads[i - 1](self.contexts[i](levels[i]))
            per_scale.append(field)

        if field.shape[-2:] != (self.hm_h, self.hm_w):
            field = F.interpolate(field, size=(self.hm_h, self.hm_w),
                                  mode="bilinear", align_corners=False)

        b = field.shape[0]
        field = field.view(b, self.num_points, 2, self.hm_h, self.hm_w)
        return field, per_scale


class ManoInitHead(nn.Module):
    """Regress a coarse MANO axis-angle pose (48) + shape (10) from a descriptor."""

    def __init__(self, feat_dim, n_pose=48, n_betas=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pose_head = nn.Linear(feat_dim, n_pose)
        self.betas_head = nn.Linear(feat_dim, n_betas)
        # Start near the flat mean hand (small pose, mean shape) for stable init.
        nn.init.zeros_(self.pose_head.weight); nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.betas_head.weight); nn.init.zeros_(self.betas_head.bias)

    def forward(self, desc):
        x = self.mlp(desc)
        return self.pose_head(x), self.betas_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Functional helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_grid(h, w, device, dtype):
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)                          # (h, w, 2)


def dovf_consensus_2d(heatmap_probs, dovf_field):
    """Dense-voting consensus 2D point estimate.

    Every pixel p votes for the point location ``p + offset(p)``; votes are
    aggregated weighted by the heatmap probability (which sums to 1 over space).

    Args:
        heatmap_probs: (B, P, H, W) softmax probabilities.
        dovf_field:    (B, P, 2, H, W) offsets in heatmap-pixel units.

    Returns:
        coords:     (B, P, 2) consensus point (heatmap px).
        confidence: (B, P)    peak heatmap probability.
    """
    b, p, h, w = heatmap_probs.shape
    grid = _pixel_grid(h, w, heatmap_probs.device, heatmap_probs.dtype)   # (H, W, 2)
    offsets = dovf_field.permute(0, 1, 3, 4, 2)                            # (B, P, H, W, 2)
    votes = grid.view(1, 1, h, w, 2) + offsets                            # (B, P, H, W, 2)
    coords = torch.einsum("bphw,bphwc->bpc", heatmap_probs, votes)        # (B, P, 2)
    confidence = heatmap_probs.flatten(2).max(dim=-1).values              # (B, P)
    return coords, confidence


def dovf_vote_cov(heatmap_probs, dovf_field, consensus):
    """2nd moment of the DOVF vote distribution (its spread = 2D localization
    uncertainty). Every pixel votes p+offset(p) with weight heatmap_probs(p);
    this returns the heatmap-prob-weighted covariance of those votes about the
    consensus, per (B,P).

    Returns sigma_cov (B,P,2,2) and sigma_px (B,P) = sqrt(0.5*trace) scalar std.
    All in heatmap-pixel units.
    """
    b, p, h, w = heatmap_probs.shape
    grid = _pixel_grid(h, w, heatmap_probs.device, heatmap_probs.dtype)   # (H,W,2)
    offsets = dovf_field.permute(0, 1, 3, 4, 2)                            # (B,P,H,W,2)
    votes = grid.view(1, 1, h, w, 2) + offsets                            # (B,P,H,W,2)
    diff = votes - consensus.view(b, p, 1, 1, 2)                          # (B,P,H,W,2)
    cov = torch.einsum("bphw,bphwc,bphwd->bpcd", heatmap_probs, diff, diff)   # (B,P,2,2)
    sigma_px = (0.5 * (cov[..., 0, 0] + cov[..., 1, 1])).clamp(min=1e-6).sqrt()
    return cov, sigma_px


def build_dovf_target(gt_2d_hm, h, w):
    """Ground-truth dense offset field at resolution (h, w).

    Args:
        gt_2d_hm: (B, P, 2) GT point locations in heatmap-pixel units.
        h, w:     target spatial resolution.

    Returns:
        (B, P, 2, h, w) target field = gt - pixel_grid.
    """
    grid = _pixel_grid(h, w, gt_2d_hm.device, gt_2d_hm.dtype)             # (h, w, 2)
    gt = gt_2d_hm.view(*gt_2d_hm.shape[:2], 1, 1, 2)                      # (B, P, 1, 1, 2)
    field = gt - grid.view(1, 1, h, w, 2)                                # (B, P, h, w, 2)
    return field.permute(0, 1, 4, 2, 3).contiguous()                     # (B, P, 2, h, w)
