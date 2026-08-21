"""Dense Offset Vector Field (DOVF) hand-pose components.

This package implements the front-end and differentiable back-end of the
DOVF-guided multi-view MANO estimator:

  * :mod:`mano_optimizer` -- a Theseus Levenberg-Marquardt solver that fits
    world-space MANO pose/translation by sampling predicted dense joint offset
    fields at the projected joint locations (the pipeline from
    ``iterative_mano_fit_theseus_dovf/method.py``, re-implemented to use the
    POEM-v2 ``manotorch.ManoLayer`` and a plain-tensor API).
  * :mod:`dovf_decoder` -- a multi-resolution neck plus heatmap and
    coarse-to-fine DOVF heads, and the dense-voting consensus + MANO-init logic.
"""

from .dovf_decoder import (  # noqa: F401
    MultiResNeck,
    HeatmapHead,
    CoarseToFineDOVFHead,
    ManoInitHead,
    dovf_consensus_2d,
    dovf_vote_cov,
    build_dovf_target,
)

__all__ = [
    "MultiResNeck",
    "HeatmapHead",
    "CoarseToFineDOVFHead",
    "ManoInitHead",
    "dovf_consensus_2d",
    "dovf_vote_cov",
    "build_dovf_target",
]
