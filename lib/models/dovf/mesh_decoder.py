"""Late, joints-anchored FREE-vertex mesh decoder (no MANO theta).

Takes the geometrically-refined 3D joints J_hat (from the MC head) and deforms a
MANO template surface around them: vertex queries cross-attend the joint anchors
(kinematic conditioning) and self-attend (surface smoothness), producing 778 free
vertices in the root-relative frame, then re-rooted at the wrist. Joints can be
read back via the fixed J_regressor (POEM-comparable: joints-from-vertices, no theta).

This is "late" on purpose: most vertices are not 2D-localizable, so they are a
learned deformation anchored to the triangulated joints rather than triangulated
themselves (see design discussion).
"""
import torch
import torch.nn as nn


class JointsPoseInit(nn.Module):
    """Learned IK warm-start: maps the (root-relative) triangulated joints mu_hat to a
    MANO-pose CORRECTION on top of init_head's pose. Zero-init -> starts at init_head's
    pose, learns to make the init mu_hat-consistent so the solver always lands in the
    same (correct) basin -> reduces the solver's non-convex basin-jumping."""
    def __init__(self, num_joints=21, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_joints * 3, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, 48))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # residual: start at 0

    def forward(self, joints, center_idx=0):
        root = joints[:, center_idx:center_idx + 1]
        return self.net((joints - root).flatten(1))      # (B,48) pose correction


class MeshDecoder(nn.Module):
    def __init__(self, template_verts_rel, template_joints_rel, num_joints=21, d=256,
                 heads=8, layers=4, center_idx=0, feat_dim=0):
        """template_verts_rel: (V,3) MANO zero-pose verts, root-relative (wrist at 0).
        template_joints_rel: (J,3) the template's joints in the same frame (for Kabsch).
        feat_dim>0: also accept a per-vertex IMAGE descriptor (fused across views) -> injects
        the per-subject SHAPE signal that joints alone can't carry (image-free shape floor)."""
        super().__init__()
        self.center_idx = center_idx
        self.register_buffer("Vt", template_verts_rel)        # (V,3)
        self.register_buffer("Jt", template_joints_rel)       # (J,3)
        self.NV = template_verts_rel.shape[0]
        self.vert_code = nn.Parameter(torch.randn(self.NV, d) * 0.02)   # per-vertex identity
        self.vin = nn.Linear(3, d)                            # template-position embed
        self.jin = nn.Linear(3, d)                            # joint-anchor embed
        self.feat_dim = int(feat_dim)
        if self.feat_dim > 0:
            self.fin = nn.Linear(self.feat_dim, d)            # per-vertex image-descriptor embed
            nn.init.zeros_(self.fin.weight); nn.init.zeros_(self.fin.bias)  # warm-start = image-free
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at posed template

    def place_template(self, J_hat):
        """Kabsch-place the canonical template at the predicted joints -> (root, Jrel, V0)."""
        B = J_hat.shape[0]
        root = J_hat[:, self.center_idx:self.center_idx + 1]            # (B,1,3)
        Jrel = J_hat - root                                            # (B,J,3) root-relative
        H = torch.einsum("jk,bjl->bkl", self.Jt, Jrel)                 # (B,3,3) cross-cov
        U, S, Vh = torch.linalg.svd(H)
        dsign = torch.sign(torch.linalg.det(torch.einsum("bij,bkj->bik", Vh, U)))
        D = torch.eye(3, device=J_hat.device).repeat(B, 1, 1)
        D[:, 2, 2] = dsign
        R = torch.einsum("bij,bjk,blk->bil", Vh.transpose(-1, -2), D, U)   # (B,3,3) Jt->Jrel
        s = (S * D.diagonal(dim1=-2, dim2=-1)).sum(-1) / (self.Jt ** 2).sum().clamp(min=1e-6)
        V0 = s[:, None, None] * torch.einsum("vk,bjk->bvj", self.Vt, R)    # (B,V,3) posed template
        return root, Jrel, V0.detach()   # geometric placement only — don't backprop through SVD

    def forward(self, J_hat, vfeat=None):
        """J_hat: (B,J,3) world joints [+ vfeat (B,V,feat_dim) per-vertex image descriptors]
        -> V_hat: (B,V,3) world vertices."""
        B = J_hat.shape[0]
        root, Jrel, V0 = self.place_template(J_hat)
        vq = self.vert_code[None].expand(B, -1, -1) + self.vin(V0)         # (B,V,d)
        if self.feat_dim > 0 and vfeat is not None:
            vq = vq + self.fin(vfeat)                                     # image-conditioned shape
        mem = self.jin(Jrel)                                          # (B,J,d) joint anchors
        for layer in self.layers:
            vq = layer(vq, mem)                  # self-attn(verts) + cross-attn(verts<-joints)
        Vrel = V0 + self.out(vq)                                      # (B,V,3) deformed
        return root + Vrel                                            # (B,V,3) world
