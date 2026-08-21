"""
Occlusion-weighted DLT triangulation for OccViTMV.

Stage 2 of OccViTMV:  for each sample with N camera views, project WiLOR's
per-view camera-space vertices to 2-D pixel coordinates, weight each view's
DLT constraints by its per-vertex visibility confidence (predicted by the
occlusion head, or from GT labels during curriculum warm-up), and solve for
the world-space vertex positions via SVD.

    P_n = K_n @ T_m2c_n[:3, :]          # world → pixel
    A_n = [[u_n*p2_n - p0_n] * w_n,     # weighted constraint rows
            [v_n*p2_n - p1_n] * w_n]
    solve  min_{||X||=1}  ||A X||  via SVD → homogeneous world coords

Single-view fallback: use the Stage-1 per-view world prediction directly.
"""

import torch


def weighted_dlt(
    u2d:        torch.Tensor,   # (B, N, V, 2)  pixel coords in crop space
    K:          torch.Tensor,   # (B, N, 3, 3)  K_crop
    T_m2c:      torch.Tensor,   # (B, N, 4, 4)  world → camera
    weights:    torch.Tensor,   # (B, N, V)     per-view per-vertex confidence ≥ 0
    return_svd: bool = False,   # if True, also return (B, V, 4) smallest singular values
) -> torch.Tensor:
    """
    Batch occlusion-weighted DLT triangulation.

    Each view's DLT constraint rows are scaled by its weight before the SVD,
    so views with low visibility (occluded vertices) contribute less to the
    solution.

    Returns
    -------
    pts3d : (B, V, 3)  world-space vertex positions.
    svd_eigs : (B, V, 4)  four smallest singular values (only when return_svd=True).
        These encode the degeneracy of the triangulation per vertex and are used
        by the Stage-2 Vertex Refinement Transformer as a quality signal.
    """
    B, N, V, _ = u2d.shape

    # ── Build projection matrices  P = K @ T_m2c[:3, :] ──────────────
    Pmat = torch.bmm(
        K.reshape(B * N, 3, 3),
        T_m2c.reshape(B * N, 4, 4)[:, :3, :],
    ).reshape(B, N, 3, 4)                      # (B, N, 3, 4)

    p0 = Pmat[:, :, 0, :]                      # (B, N, 4) — row 0
    p1 = Pmat[:, :, 1, :]                      # (B, N, 4) — row 1
    p2 = Pmat[:, :, 2, :]                      # (B, N, 4) — row 2

    u = u2d[..., 0]                            # (B, N, V)
    v = u2d[..., 1]                            # (B, N, V)
    w = weights.clamp(min=1e-6)                # (B, N, V)

    # ── Weighted constraint rows ─────────────────────────────────────
    # row_u_n = w_n * (u_n * P2_n − P0_n)   shape: (B, N, V, 4)
    row_u = (u.unsqueeze(-1) * p2.unsqueeze(2) - p0.unsqueeze(2)) * w.unsqueeze(-1)
    row_v = (v.unsqueeze(-1) * p2.unsqueeze(2) - p1.unsqueeze(2)) * w.unsqueeze(-1)

    # A: (B, V, 2N, 4)
    A = torch.cat(
        [row_u.permute(0, 2, 1, 3),
         row_v.permute(0, 2, 1, 3)],
        dim=2,
    )

    # ── SVD in fp32 for numerical stability ───────────────────────────
    dtype_in = A.dtype
    with torch.amp.autocast(device_type="cuda", enabled=False):
        _, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    # Last right-singular vector → null-space solution (homogeneous world coords)
    Xh    = Vh[:, :, -1, :]                    # (B, V, 4)
    pts3d = Xh[:, :, :3] / (Xh[:, :, 3:4] + 1e-8)
    if return_svd:
        # Return the 4 smallest singular values per vertex as a degeneracy signal.
        # S has shape (B, V, min(2N, 4)); we want the last 4 (or fewer if 2N < 4).
        n_sv = S.shape[-1]
        k    = min(4, n_sv)
        svd_eigs = S[:, :, n_sv - k:].to(dtype_in)  # (B, V, k≤4)
        # Zero-pad to exactly 4 columns so downstream code always sees [B, V, 4]
        if k < 4:
            pad = torch.zeros(S.shape[0], S.shape[1], 4 - k, dtype=dtype_in, device=S.device)
            svd_eigs = torch.cat([pad, svd_eigs], dim=-1)
        return pts3d.to(dtype_in), svd_eigs        # (B, V, 3), (B, V, 4)
    return pts3d.to(dtype_in)                  # (B, V, 3)
