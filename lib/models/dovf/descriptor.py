"""
Learned cross-view DESCRIPTORS for epipolar matching (Phase A′).
================================================================

The epipolar refiner (Phase C) was inert because the raw HRNet features carry no
depth signal along the epipolar line (oracle match ≈ chance, feat_var ≈ 0). This
module trains the features to BE correspondence-discriminative, using the free,
exact cross-view correspondence supervision from the registered MANO mesh.

* ``DescriptorHead`` — a small multi-scale head on the neck pyramid -> a per-pixel,
  L2-normalised D-dim descriptor map. Multi-scale gives it context (coarse RF =
  "where in the hand am I") so it can be position-discriminative even on textureless
  skin.
* ``correspondence_loss`` — for random co-visible MANO vertices in a view pair (v, vp):
    (1) EPIPOLAR BEAM-MATCHING CE: build D depth hypotheses along the vertex's ray in
        v (beam RANDOMLY recentred so the true depth lands at a random index -> the
        model can't cheat by always picking the middle), project into vp, and the
        descriptor at the v-template must pick the GT-closest hypothesis. This loss IS
        the ``oracle%`` objective the diagnostic measures.
    (2) global InfoNCE: the v-descriptor of a vertex must match its vp-descriptor over
        all sampled vertices -> globally discriminative descriptors.
  Co-visibility is a front-facing test from MANO vertex normals (cheap occlusion proxy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .epipolar_filter import _grid_sample_pts


def compute_vertex_normals(verts, faces):
    """verts (B,V,3) world, faces (F,3) long -> per-vertex unit normals (B,V,3)."""
    B, V, _ = verts.shape
    v0 = verts[:, faces[:, 0]]; v1 = verts[:, faces[:, 1]]; v2 = verts[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)                 # area-weighted face normal
    n = verts.new_zeros(B, V, 3)
    for k in range(3):
        n.index_add_(1, faces[:, k], fn)
    return F.normalize(n, dim=-1)


def _project(P, K3, w2c4):
    """P (...,3) world -> uv (...,2) heatmap px under single-view K3 (3,3), w2c4 (4,4)."""
    R = w2c4[:3, :3]; t = w2c4[:3, 3]
    Xc = torch.einsum("ij,...j->...i", R, P) + t
    z = Xc[..., 2:3].clamp(min=1e-4)
    return torch.einsum("ij,...j->...i", K3, Xc)[..., :2] / z


def _samp(desc2d, uv, H, W):
    """desc2d (Dd,H,W), uv (...,2) px -> (...,Dd) bilinear descriptors."""
    Dd = desc2d.shape[0]
    flat = uv.reshape(-1, 2)
    out = _grid_sample_pts(desc2d[None], flat[None], H, W)[0]   # (M,Dd)
    return out.reshape(*uv.shape[:-1], Dd)


class DescriptorHead(nn.Module):
    """Neck pyramid -> L2-normalised D-dim descriptor map at the finest resolution."""

    def __init__(self, feat_dim, n_levels, d_desc=64, hidden=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(feat_dim, hidden, 1) for _ in range(n_levels)])
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(min(32, hidden), hidden), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(min(32, hidden), hidden), nn.ReLU(True),
            nn.Conv2d(hidden, d_desc, 1))

    def forward(self, pyramid):
        H, W = pyramid[0].shape[-2:]
        x = self.proj[0](pyramid[0])
        for i in range(1, len(pyramid)):
            x = x + F.interpolate(self.proj[i](pyramid[i]), (H, W), mode="bilinear", align_corners=False)
        return F.normalize(self.head(x), dim=1)                # (B, d_desc, H, W)


def correspondence_loss(desc, verts, normals, K_hm, w2c, cvn, H, W,
                        n_samp=128, D=12, rrange=0.18, tau=0.07, pairs_per_scene=2,
                        nce_weight=0.5, generator=None):
    """Epipolar beam-matching CE + global InfoNCE over co-visible MANO vertices.
    desc (BN,Dd,H,W) L2-normalised; verts/normals (B,Vv,3); K_hm/w2c flat per-image.
    Returns (loss, stats) with stats['acc'] = train beam-oracle accuracy."""
    BN, Dd = desc.shape[:2]
    B = len(cvn); dev = desc.device
    C = -torch.einsum("bji,bj->bi", w2c[:, :3, :3], w2c[:, :3, 3])     # cam centres (BN,3)
    steps = torch.linspace(-rrange, rrange, D, device=dev)
    ce_list, nce_list, acc_list = [], [], []
    off = 0
    offsets = [int(sum(cvn[:i])) for i in range(B)]
    for i in range(B):
        nvi = int(cvn[i]); o = offsets[i]
        if nvi < 2:
            continue
        Pv = verts[i]; nrm = normals[i]                                # (Vv,3)
        for _ in range(pairs_per_scene):
            perm = torch.randperm(nvi, generator=generator)[:2].tolist()
            a = o + perm[0]; b = o + perm[1]
            ffa = ((nrm * (C[a] - Pv)).sum(-1) > 0)                    # front-facing in a
            ffb = ((nrm * (C[b] - Pv)).sum(-1) > 0)
            uva = _project(Pv, K_hm[a], w2c[a]); uvb = _project(Pv, K_hm[b], w2c[b])
            ina = (uva[:, 0] >= 0) & (uva[:, 0] < W) & (uva[:, 1] >= 0) & (uva[:, 1] < H)
            inb = (uvb[:, 0] >= 0) & (uvb[:, 0] < W) & (uvb[:, 1] >= 0) & (uvb[:, 1] < H)
            ok = (ffa & ffb & ina & inb).nonzero().squeeze(-1)
            if ok.numel() < 8:
                continue
            sel = ok[torch.randperm(ok.numel(), generator=generator)[:n_samp]]
            P = Pv[sel]                                                # (N,3)
            t = _samp(desc[a], uva[sel], H, W)                        # (N,Dd) template in a
            ptrue = _samp(desc[b], uvb[sel], H, W)                    # (N,Dd) true positive in b
            # beam along ray-in-a, RANDOMLY recentred so true depth lands at a random index
            dC = P - C[a]                                              # ray param (N,3); t=1 at P
            center = 1.0 + (torch.rand(P.shape[0], device=dev, generator=generator) * 2 - 1) * (rrange * 0.5)
            ts = center[:, None] + steps[None, :]                      # (N,D)
            Pd = C[a][None, None, :] + ts[..., None] * dC[:, None, :]  # (N,D,3)
            uvb_beam = _project(Pd, K_hm[b], w2c[b])                  # (N,D,2)
            beam = _samp(desc[b], uvb_beam, H, W)                     # (N,D,Dd)
            logits = torch.einsum("nc,ndc->nd", t, beam) / tau        # (N,D)
            target = (Pd - P[:, None, :]).norm(dim=-1).argmin(-1)     # GT-closest hypothesis
            ce_list.append(F.cross_entropy(logits, target))
            acc_list.append((logits.argmax(-1) == target).float().mean())
            g = (t @ ptrue.t()) / tau                                 # global InfoNCE
            nce_list.append(F.cross_entropy(g, torch.arange(P.shape[0], device=dev)))
    if not ce_list:
        z = desc.sum() * 0.0
        return z, {"ce": 0.0, "nce": 0.0, "acc": 0.0, "npair": 0}
    ce = torch.stack(ce_list).mean(); nce = torch.stack(nce_list).mean()
    return ce + nce_weight * nce, {"ce": float(ce.detach()), "nce": float(nce.detach()),
                                   "acc": float(torch.stack(acc_list).mean()), "npair": len(ce_list)}
