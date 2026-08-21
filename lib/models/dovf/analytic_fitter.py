"""
Analytic-Jacobian batched MANO fitter (pure torch, no Theseus autograd-Jacobian).

Replaces the Theseus DENSE-autograd inner loop. The reprojection residual is
    r[v,j] = DOVF_v( pi_v( MANO_joints(pose,trans) ) )                 (V*J*2,)
and the Jacobian wrt theta=[pose(48), trans(3)] is assembled analytically by the
chain rule:
    dr/dtheta = (dDOVF/duv) @ (duv/dXcam) @ (dXcam/dXworld) @ (dXworld/dtheta)
where only the last term (the MANO kinematic Jacobian dXworld/dpose) uses
autograd -- and via a single *batched* vector-Jacobian product
(`is_grads_batched=True`), not Theseus's per-residual-row backprop. The
projection Jacobian and the bilinear-DOVF spatial gradient are closed form, so
the expensive `nonzero`/`index_put` autograd of the gather disappears.

Objective (matches the Theseus optimizer in mano_optimizer.py):
    0.5 * || w_huber * r_dovf ||^2  +  0.5*lambda*||pose-pose0||^2
                                     +  0.5*lambda*||trans-trans0||^2
solved by damped Gauss-Newton (Levenberg-Marquardt) with `step_size`.

Per-element Huber IRLS weights are used (w = 1 if |e|<=delta else sqrt(delta/|e|)).
With a large radius this reduces to plain least squares (used for the
equivalence test against Theseus).
"""

from typing import Optional
import torch
import torch.nn.functional as F

from lib.utils.transform import mano_to_openpose
from lib.utils.misc import CONST

# fingertip vertex ids used by mano_to_openpose (must match exactly)
_TIP_VERTS = [v[0] for v in CONST.MANO_KPID_2_VERTICES.values()]   # [744,320,443,555,672]


def _mano_joints(mano_layer, J_regressor, pose, betas, trans, num_joints):
    """pose (B,48), betas (B,10), trans (B,3) -> (B,J,3) world; also returns centered."""
    verts = mano_layer(pose, betas).verts                      # (B,778,3) centered
    jc = mano_to_openpose(J_regressor, verts)[:, :num_joints]  # (B,J,3) centered
    return jc + trans.unsqueeze(1), jc                         # world, centered


# OpenPose joint order (mano_to_openpose): stack = [16 MANO joints, 5 tips]; reorder by this perm.
_OPENPOSE_PERM = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# stack tips 16..20 -> the distal MANO joint each fingertip rides on (thumb,index,middle,ring,pinky)
_TIP_DISTAL = [15, 3, 6, 12, 9]


def _ancestor_mask(parents, n=16, device="cpu"):
    """A[k,j]=1 if joint j is on the kinematic path root..k (inclusive). (n,n)."""
    A = torch.zeros(n, n)
    for k in range(n):
        j = k
        while True:
            A[k, j] = 1.0
            p = int(parents[j])
            if j == 0 or p < 0 or p >= n:
                break
            j = p
        A[k, 0] = 1.0
    return A.to(device)


def _skew(v):
    """(...,3) -> (...,3,3) skew-symmetric."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def _so3_right_jacobian(theta):
    """Right Jacobian of SO(3) for axis-angle theta (...,3) -> (...,3,3).
    exp(theta+dtheta) ~= exp(theta) exp(Jr(theta) dtheta)."""
    phi = theta.norm(dim=-1, keepdim=True).clamp(min=1e-8)        # (...,1)
    K = _skew(theta)                                              # (...,3,3)
    K2 = K @ K
    phi_ = phi.unsqueeze(-1)                                      # (...,1,1)
    a = ((1 - torch.cos(phi)) / phi**2).unsqueeze(-1)            # (...,1,1)
    b = ((phi - torch.sin(phi)) / phi**3).unsqueeze(-1)
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).expand_as(K)
    Jr = eye - a * K + b * K2
    small = (phi.squeeze(-1) < 1e-5).unsqueeze(-1).unsqueeze(-1)
    return torch.where(small, eye, Jr)


def mano_kinematic_jac(mano_out, mano_layer, pose, betas, anc_mask, center_idx, num_joints):
    """Analytic dJc/dpose (B, num_joints, 3, 48) for the fitter's joints jc = M @ verts.

    Cross-product articulated Jacobian with the SO(3) right-Jacobian correction
    (exact for kinematic points around the current pose). The 16 body joints use
    the kinematic bone-end positions; the 5 fingertips use their TRUE positions
    p_distal + R_distal (v_rest_tip - J_rest_distal) (tips are skinned ~rigidly to
    the distal joint), riding the distal joint's kinematic chain. The
    pose-blendshape term is omitted (small; the Jacobian only needs to be a good
    GN direction and the outer gradient detaches it). Requires flat_hand_mean."""
    T = mano_out.transforms_abs                                  # (B,16,4,4) uncentered
    Rg = T[:, :, :3, :3]                                         # (B,16,3,3)
    pg = T[:, :, :3, 3]                                          # (B,16,3)
    B = pose.shape[0]
    theta = pose.reshape(B, 16, 3)
    Jr = _so3_right_jacobian(theta)                              # (B,16,3,3)
    axes = Rg @ Jr                                               # (B,16,3,3): col m = world axis for dtheta_{j,m}

    # ---- true fingertip positions in the (uncentered) transform frame ----
    bS = torch.matmul(mano_layer.th_shapedirs, betas.transpose(0, 1)).permute(2, 0, 1)  # (B,778,3)
    v_rest = mano_layer.th_v_template + bS                       # (B,778,3) rest verts
    J_rest = torch.matmul(mano_layer.th_J_regressor, v_rest)     # (B,16,3) rest joints
    tips_rest = v_rest[:, _TIP_VERTS]                            # (B,5,3)
    d = _TIP_DISTAL
    tip_pos = pg[:, d] + torch.einsum("bdij,bdj->bdi", Rg[:, d], tips_rest - J_rest[:, d])  # (B,5,3)

    # ---- unified stack of 21 points: 16 bone-ends + 5 true tips ----
    P = torch.cat([pg, tip_pos], dim=1)                          # (B,21,3)
    A = torch.cat([anc_mask, anc_mask[d]], dim=0)                # (21,16): tip rides distal's chain
    S = P.shape[1]
    diff = P.unsqueeze(2) - pg.unsqueeze(1)                      # (B,S,j,3) = p_s - p_j
    a_e = axes.unsqueeze(1).expand(B, S, 16, 3, 3)              # (B,S,j,xyz,m)
    d_e = diff.unsqueeze(-1).expand(B, S, 16, 3, 3)            # (B,S,j,xyz,m)
    cr = torch.cross(a_e, d_e, dim=3) * A.view(1, S, 16, 1, 1)
    dP_stack = cr.permute(0, 1, 3, 2, 4).reshape(B, S, 3, 48)    # (B,21,3,48)
    dJc = dP_stack[:, _OPENPOSE_PERM]                            # to openpose order
    dJc = dJc - dJc[:, center_idx:center_idx + 1]               # center at center_idx
    return dJc[:, :num_joints]


def _joints_pose_jac(jc, pose):
    """Batched dJc/dpose via one vectorized vjp. jc (B,J,3) needs grad wrt pose (B,48).
    Returns (B, J, 3, 48)."""
    B, J, _ = jc.shape
    out = jc.reshape(B, J * 3)
    # identity grad_outputs batched over the J*3 output dims
    eye = torch.eye(J * 3, device=jc.device, dtype=jc.dtype).unsqueeze(1).expand(J * 3, B, J * 3)
    (grad,) = torch.autograd.grad(out, pose, grad_outputs=eye, is_grads_batched=True,
                                  retain_graph=True, create_graph=False)
    # grad: (J*3, B, 48) -> (B, J, 3, 48)
    return grad.permute(1, 0, 2).reshape(B, J, 3, 48)


def _project_and_jac(joints_world, K, w2c):
    """joints_world (B,J,3) -> uv (B,V,J,2), duv_dXworld (B,V,J,2,3)."""
    B, J, _ = joints_world.shape
    V = w2c.shape[1]
    R = w2c[:, :, :3, :3]                         # (B,V,3,3)
    t = w2c[:, :, :3, 3]                          # (B,V,3)
    jw = joints_world.unsqueeze(1).expand(-1, V, -1, -1)               # (B,V,J,3)
    Xc = torch.einsum("bvik,bvjk->bvji", R, jw) + t.unsqueeze(2)       # (B,V,J,3)
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2].clamp(min=1e-3)       # (B,V,J)
    fx = K[:, :, 0, 0].unsqueeze(-1); fy = K[:, :, 1, 1].unsqueeze(-1)
    cx = K[:, :, 0, 2].unsqueeze(-1); cy = K[:, :, 1, 2].unsqueeze(-1)
    u = fx * x / z + cx
    v = fy * y / z + cy
    uv = torch.stack([u, v], dim=-1)                                  # (B,V,J,2)
    # duv/dXcam : (B,V,J,2,3)
    zero = torch.zeros_like(x)
    duv_dXc = torch.stack([
        torch.stack([fx / z, zero, -fx * x / z**2], dim=-1),
        torch.stack([zero, fy / z, -fy * y / z**2], dim=-1),
    ], dim=-2)                                                        # (B,V,J,2,3)
    # dXcam/dXworld = R  ->  duv/dXworld = duv_dXc @ R
    duv_dXw = torch.einsum("bvjik,bvkl->bvjil", duv_dXc, R)           # (B,V,J,2,3)
    return uv, duv_dXw


def _sample_dovf_and_grad(uv, dovf, H, W):
    """Bilinear sample + analytic spatial gradient.
    uv (B,V,J,2) px; dovf (B,V,H,W,J,2) -> r (B,V,J,2), dr_duv (B,V,J,2,2)."""
    B, V, J = uv.shape[:3]
    px = uv[..., 0].clamp(0.0, W - 1.0)
    py = uv[..., 1].clamp(0.0, H - 1.0)
    x0 = px.detach().floor().long().clamp(0, W - 2)
    y0 = py.detach().floor().long().clamp(0, H - 2)
    x1 = x0 + 1; y1 = y0 + 1
    wx = (px - x0.float()).unsqueeze(-1)                              # (B,V,J,1)
    wy = (py - y0.float()).unsqueeze(-1)

    bb = torch.arange(B, device=uv.device).view(B, 1, 1).expand(B, V, J).reshape(-1)
    bv = torch.arange(V, device=uv.device).view(1, V, 1).expand(B, V, J).reshape(-1)
    bj = torch.arange(J, device=uv.device).view(1, 1, J).expand(B, V, J).reshape(-1)

    def corner(xi, yi):
        return dovf[bb, bv, yi.reshape(-1), xi.reshape(-1), bj, :].reshape(B, V, J, 2)

    f00 = corner(x0, y0); f10 = corner(x1, y0)
    f01 = corner(x0, y1); f11 = corner(x1, y1)
    r = (f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy)
         + f01 * (1 - wx) * wy + f11 * wx * wy)                       # (B,V,J,2)
    # analytic bilinear spatial gradient (per output component c):
    dr_du = (f10 - f00) * (1 - wy) + (f11 - f01) * wy                 # (B,V,J,2)
    dr_dv = (f01 - f00) * (1 - wx) + (f11 - f10) * wx                 # (B,V,J,2)
    dr_duv = torch.stack([dr_du, dr_dv], dim=-1)                      # (B,V,J,2,2)  [c, (u,v)]
    return r, dr_duv


def _build_system(mano_layer, J_regressor, pose, trans, betas, K, w2c, dovf,
                  num_joints, H, W, jac_mode, anc_mask, center_idx, radius, view_weight,
                  joint_conf=None):
    """One linearization: returns (rv (B,R) diff wrt dovf, Jm (B,R,51) detached,
    w (B,R) IRLS x view x per-view-joint-confidence). Jacobian detached (GN constant)."""
    B = pose.shape[0]; P = 51
    if jac_mode == "analytic":
        out = mano_layer(pose, betas)
        jc = mano_to_openpose(J_regressor, out.verts)[:, :num_joints]
        jw = jc + trans.unsqueeze(1)
        dJc = mano_kinematic_jac(out, mano_layer, pose.detach(), betas, anc_mask,
                                 center_idx, num_joints).detach()
    else:
        with torch.enable_grad():
            pl = pose.detach().requires_grad_(True)
            _, jcg = _mano_joints(mano_layer, J_regressor, pl, betas, trans.detach(), num_joints)
            dJc = _joints_pose_jac(jcg, pl).detach()
        jw, _ = _mano_joints(mano_layer, J_regressor, pose, betas, trans, num_joints)
    uv, duv_dXw = _project_and_jac(jw, K, w2c)
    r, dr_duv = _sample_dovf_and_grad(uv, dovf, H, W)
    Vn = uv.shape[1]
    dr_dXw = torch.einsum("bvjci,bvjik->bvjck", dr_duv.detach(), duv_dXw.detach())
    dr_dpose = torch.einsum("bvjck,bvjkp->bvjcp", dr_dXw, dJc.unsqueeze(1).expand(B, Vn, -1, -1, -1))
    Jm = torch.cat([dr_dpose.reshape(B, Vn, num_joints, 2, 48),
                    dr_dXw.reshape(B, Vn, num_joints, 2, 3)], dim=-1).reshape(B, Vn * num_joints * 2, P).detach()
    rv = r.reshape(B, Vn * num_joints * 2)
    absr = rv.detach().abs()
    delta = radius.reshape(B, 1) if radius.numel() == B else radius
    w = torch.where(absr <= delta, torch.ones_like(absr), torch.sqrt(delta / absr.clamp(min=1e-9)))
    if view_weight is not None:
        w = w * view_weight.view(B, Vn, 1, 1).expand(B, Vn, num_joints, 2).reshape(B, -1)
    if joint_conf is not None:
        # per-(view,joint) reliability weight (e.g. heatmap peak prob): trust
        # confident joint detections, down-weight occluded/uncertain ones.
        w = w * joint_conf.view(B, Vn, num_joints, 1).expand(B, Vn, num_joints, 2).reshape(B, -1)
    return rv, Jm, w


def _normal_eqs(rv, Jm, w, theta, theta0, lam, lm_damping, prior=None):
    """Gauss-Newton system: A = JᵀWJ + prior_prec + μI,  g = JᵀW r + prior_grad.

    Default prior (prior=None): isotropic L2 toward the init θ0, weight λ.
    Structured prior (prior=(prec, ref, anchor)):
      - prec   (P,P): PCA-Mahalanobis precision on the finger block (toward the mean
                      hand `ref`); 0 elsewhere. λ is already baked into `prec`.
      - ref    (P,)  : prior mean (hands_mean on fingers, 0 elsewhere).
      - anchor (P,)  : per-DoF L2-toward-init weight (global orient + trans only).
    """
    B, R, P = Jm.shape
    WJ = w.unsqueeze(-1) * Jm
    eye = torch.eye(P, device=Jm.device, dtype=Jm.dtype).unsqueeze(0)
    A = torch.einsum("bri,brj->bij", WJ, Jm) + lm_damping * eye
    g = torch.einsum("bri,br->bi", WJ, rv)
    if prior is None:
        A = A + lam * eye
        g = g + lam * (theta - theta0)
    else:
        prec, ref, anchor = prior
        A = A + prec.unsqueeze(0) + torch.diag(anchor).unsqueeze(0)
        g = (g + torch.einsum("ij,bj->bi", prec, theta - ref)
             + anchor * (theta - theta0))
    return A, g


def _gn_loop(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf, radius,
             num_joints, H, W, jac_mode, anc_mask, center_idx, view_weight,
             max_iterations, step_size, lam, lm_damping, prior=None, joint_conf=None):
    pose, trans = pose0, trans0
    theta0 = torch.cat([pose0, trans0], dim=-1)
    for _ in range(max_iterations):
        rv, Jm, w = _build_system(mano_layer, J_regressor, pose, trans, betas, K, w2c, dovf,
                                  num_joints, H, W, jac_mode, anc_mask, center_idx, radius,
                                  view_weight, joint_conf)
        theta = torch.cat([pose, trans], dim=-1)
        A, g = _normal_eqs(rv, Jm, w, theta, theta0, lam, lm_damping, prior)
        d = torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
        theta = theta - step_size * d
        pose, trans = theta[:, :48], theta[:, 48:]
    return pose, trans


class _ImplicitGN(torch.autograd.Function):
    """Implicit differentiation of the GN fixed point. Forward runs the loop under
    no_grad (O(1) memory, iteration-independent). Backward: at θ* the optimality
    condition g(θ*,φ)=0, so dθ*/dφ = -A⁻¹ ∂g/∂φ; given upstream ḡ, dL/dφ =
    -(∂g/∂φ)ᵀ A⁻¹ ḡ — one extra solve with the cached GN Hessian + one vjp of g.
    Only 1st-order derivatives of the bilinear residual are needed."""
    @staticmethod
    def forward(ctx, dovf, pose0, trans0, betas, log_radius, K, w2c, view_weight,
                mano_layer, J_regressor, anc_mask, num_joints, H, W,
                max_iterations, step_size, lam, lm_damping, jac_mode, center_idx, prior, joint_conf):
        radius = log_radius.exp().reshape(-1)
        with torch.no_grad():
            pose, trans = _gn_loop(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf,
                                   radius, num_joints, H, W, jac_mode, anc_mask, center_idx,
                                   view_weight, max_iterations, step_size, lam, lm_damping, prior, joint_conf)
        ctx.save_for_backward(dovf, pose0, trans0, betas, log_radius, K, w2c,
                              view_weight if view_weight is not None else dovf.new_zeros(0),
                              anc_mask if anc_mask is not None else dovf.new_zeros(0), pose, trans)
        ctx.cfg = (mano_layer, J_regressor, num_joints, H, W, lam, lm_damping, jac_mode,
                   center_idx, view_weight is not None, anc_mask is not None)
        ctx.prior = prior            # constant tensors (no grad)
        ctx.joint_conf = joint_conf  # constant per-(view,joint) reliability weight
        return pose, trans

    @staticmethod
    def backward(ctx, g_pose, g_trans):
        dovf, pose0, trans0, betas, log_radius, K, w2c, vw_s, anc_s, pose, trans = ctx.saved_tensors
        (mano_layer, J_regressor, num_joints, H, W, lam, lm_damping, jac_mode,
         center_idx, has_vw, has_anc) = ctx.cfg
        view_weight = vw_s if has_vw else None
        anc_mask = anc_s if has_anc else None

        dovf_ = dovf.detach().requires_grad_(True)
        pose0_ = pose0.detach().requires_grad_(True)
        trans0_ = trans0.detach().requires_grad_(True)
        betas_ = betas.detach().requires_grad_(True)
        lr_ = log_radius.detach().requires_grad_(True)
        with torch.enable_grad():
            radius = lr_.exp().reshape(-1)
            rv, Jm, w = _build_system(mano_layer, J_regressor, pose.detach(), trans.detach(), betas_,
                                      K, w2c, dovf_, num_joints, H, W, jac_mode, anc_mask, center_idx,
                                      radius, view_weight, ctx.joint_conf)
            theta = torch.cat([pose, trans], dim=-1).detach()
            theta0 = torch.cat([pose0_, trans0_], dim=-1)
            A, g = _normal_eqs(rv, Jm, w, theta, theta0, lam, lm_damping, ctx.prior)  # optimality gradient
        ghat = torch.cat([g_pose, g_trans], dim=-1)                         # (B,51)
        v = torch.linalg.solve(A.detach(), ghat.unsqueeze(-1)).squeeze(-1)  # A⁻¹ ḡ
        grads = torch.autograd.grad(g, [dovf_, pose0_, trans0_, betas_, lr_],
                                    grad_outputs=v, allow_unused=True)
        d = [(-x if x is not None else None) for x in grads]
        # match forward inputs: dovf,pose0,trans0,betas,log_radius,K,w2c,view_weight, + non-diff + prior + joint_conf
        return (d[0], d[1], d[2], d[3], d[4], None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None)


def analytic_fit(
    mano_layer, J_regressor,
    pose0, trans0, betas, K, w2c, dovf,
    num_joints=21, H=64, W=64,
    max_iterations=10, step_size=0.5, pose_prior_weight=1.0,
    log_radius: Optional[torch.Tensor] = None, loss_radius_init=3.0,
    lm_damping=1e-3, detach_jac=False,
    view_weight: Optional[torch.Tensor] = None,   # (B,V) 0/1 — 0 zeroes padded views
    jac_mode="autograd",                          # "autograd" | "analytic"
    center_idx=0,
    backward="unroll",                            # "unroll" | "implicit"
    prior_prec: Optional[torch.Tensor] = None,    # (P,P) PCA-Mahalanobis precision (λ baked in)
    prior_ref: Optional[torch.Tensor] = None,     # (P,) prior mean (hands_mean on fingers)
    prior_anchor: Optional[torch.Tensor] = None,  # (P,) L2-toward-init weights (global+trans)
    joint_conf: Optional[torch.Tensor] = None,    # (B,V,J) per-view-joint reliability weight
):
    """Damped Gauss-Newton fit guided by joint DOVFs. Returns (pose, trans).

    backward="unroll":   plain autograd through the loop (memory O(iters)); the
                         Jacobian is detached so no MANO/grid 2nd-order is needed.
    backward="implicit": fixed-point implicit differentiation (O(1) memory,
                         iteration-independent) via :class:`_ImplicitGN`.
    Differentiable wrt dovf, pose0, trans0, betas, log_radius."""
    device = pose0.device
    lam = float(pose_prior_weight)
    anc_mask = _ancestor_mask(mano_layer.kintree_parents, device=device) if jac_mode == "analytic" else None
    if log_radius is None:
        log_radius = torch.log(torch.tensor([loss_radius_init], device=device))

    prior = None
    if prior_prec is not None:
        prior = (prior_prec, prior_ref, prior_anchor)

    if backward == "implicit":
        return _ImplicitGN.apply(dovf, pose0, trans0, betas, log_radius, K, w2c, view_weight,
                                 mano_layer, J_regressor, anc_mask, num_joints, H, W,
                                 max_iterations, step_size, lam, lm_damping, jac_mode, center_idx,
                                 prior, joint_conf)

    radius = log_radius.exp().reshape(-1)
    return _gn_loop(mano_layer, J_regressor, pose0, trans0, betas, K, w2c, dovf, radius,
                    num_joints, H, W, jac_mode, anc_mask, center_idx, view_weight,
                    max_iterations, step_size, lam, lm_damping, prior, joint_conf)
