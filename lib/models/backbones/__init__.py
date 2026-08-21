# Pruned for the UA-Fit release: HRNet-W40 is the only backbone used by the
# released checkpoint.
from ...utils.builder import BACKBONE, build_from_cfg
from .hrnet import HRNet


def build_backbone(cfg, **kwargs):
    return build_from_cfg(cfg, BACKBONE, **kwargs)


__all__ = ["HRNet", "build_backbone"]
