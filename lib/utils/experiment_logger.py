"""Unified experiment logger over TensorBoard and/or Weights & Biases.

Drop-in replacement for ``DDPSummaryWriter`` — exposes the same
``add_scalar`` / ``add_scalars`` / ``add_image`` signatures the model already
calls, plus ``add_figure`` for the DOVF visualisation panels. The default
backend is **wandb** (so a run can be monitored remotely) with a graceful
fall-back to TensorBoard if wandb is unavailable or login fails.

All logging is master-only (rank 0). WandB requires a globally non-decreasing
step, so we clamp every wandb step to ``max(seen, requested)`` — this lets
per-step train scalars and per-epoch val scalars share one timeline without
wandb dropping the out-of-order epoch points (they land at the latest step,
which is chronologically correct since val runs after that train step).
"""
import os

from .logger import logger
from .summary_writer import DDPSummaryWriter


def _cfg_to_dict(cfg):
    if cfg is None:
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg.dump())
    except Exception:
        try:
            return dict(cfg)
        except Exception:
            return {}


def _to_hwc(img):
    """Accept CHW / HWC torch or numpy -> HxWxC uint8/float numpy for wandb.Image."""
    import numpy as np
    import torch
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))   # CHW -> HWC
    return img


class ExperimentLogger:
    """TensorBoard + WandB unified logger. backend in {wandb, tensorboard, both}."""

    def __init__(self, log_dir, rank, cfg=None, exp_id=None, backend="wandb",
                 project="poem-dovf", entity=None, mode=None, tags=None, notes=None):
        self.rank = rank
        self.backend = (backend or "wandb").lower()
        self.tb = None
        self.wandb = None
        self._wb_step = -1

        want_tb = self.backend in ("tensorboard", "tb", "both")
        want_wandb = self.backend in ("wandb", "wb", "both")

        if want_tb:
            self.tb = DDPSummaryWriter(log_dir=log_dir, rank=rank)

        if want_wandb and rank == 0:
            try:
                import wandb
                run_dir = os.path.dirname(log_dir.rstrip("/")) or "."
                self.wandb = wandb
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", project),
                    entity=entity or os.environ.get("WANDB_ENTITY"),
                    name=exp_id,
                    dir=run_dir,
                    config=_cfg_to_dict(cfg),
                    mode=mode or os.environ.get("WANDB_MODE", "online"),
                    tags=tags,
                    notes=notes,
                    resume="allow",
                )
                logger.info(f"[ExperimentLogger] wandb run: {wandb.run.url if wandb.run else '(offline)'}")
            except Exception as e:  # not installed / not logged in / offline
                logger.warning(f"[ExperimentLogger] wandb unavailable ({e}); falling back to TensorBoard.")
                self.wandb = None
                if self.tb is None:
                    self.tb = DDPSummaryWriter(log_dir=log_dir, rank=rank)

    # ── step bookkeeping ────────────────────────────────────────────────────
    def _wb_log(self, payload, global_step):
        if self.wandb is None or self.rank != 0:
            return
        step = self._wb_step if global_step is None else max(int(global_step), self._wb_step)
        self._wb_step = step
        try:
            self.wandb.log(payload, step=step)
        except Exception as e:
            logger.warning(f"[ExperimentLogger] wandb.log failed: {e}")

    # ── scalar API (matches DDPSummaryWriter) ───────────────────────────────
    def add_scalar(self, tag, value, global_step=None, walltime=None):
        if self.rank != 0:
            return
        try:
            value = float(value.item() if hasattr(value, "item") else value)
        except Exception:
            return
        if self.tb is not None:
            self.tb.add_scalar(tag, value, global_step=global_step, walltime=walltime)
        self._wb_log({tag: value}, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.rank != 0:
            return
        if self.tb is not None:
            self.tb.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)
        self._wb_log({f"{main_tag}/{k}": float(v) for k, v in tag_scalar_dict.items()}, global_step)

    # ── image / figure API ──────────────────────────────────────────────────
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"):
        if self.rank != 0:
            return
        if self.tb is not None:
            self.tb.add_image(tag, img_tensor, global_step=global_step,
                              walltime=walltime, dataformats=dataformats)
        if self.wandb is not None:
            img = _to_hwc(img_tensor) if dataformats.upper() == "CHW" else img_tensor
            self._wb_log({tag: self.wandb.Image(img)}, global_step)

    def add_figure(self, tag, figure, global_step=None, close=True):
        """Log a matplotlib Figure to both backends (rasterised for wandb)."""
        if self.rank != 0:
            return
        if self.tb is not None:
            try:
                self.tb.add_figure(tag, figure, global_step=global_step, close=False)
            except Exception:
                pass
        if self.wandb is not None:
            self._wb_log({tag: self.wandb.Image(figure)}, global_step)
        if close:
            try:
                import matplotlib.pyplot as plt
                plt.close(figure)
            except Exception:
                pass

    def finish(self):
        if self.rank != 0:
            return
        if self.tb is not None:
            try:
                self.tb.flush(); self.tb.close()
            except Exception:
                pass
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass
