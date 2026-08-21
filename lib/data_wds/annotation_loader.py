"""
Occlusion annotation loader for WebDataset-based training pipelines.

Annotation files are produced by ``scripts/annotate_occlusion.py`` and stored as::

    {ann_dir}/{dataset}_{split}/{safe_key}.npy

Each ``.npy`` file contains an ``(N_views, 778)`` uint8 array with 2-bit labels:
    - bit 0 (value & 1) : self-occlusion  (rasterization-based)
    - bit 1 (value & 2) : other-occlusion (depth or SAM3-based)
    - 0 = fully visible, 1 = self-occ, 2 = other-occ, 3 = both

Typical usage inside a WebDataset pipeline::

    from lib.data_wds.annotation_loader import OcclusionAnnotationLoader

    loader = OcclusionAnnotationLoader(
        ann_dir="data/annotations/occ_labels",
        dataset_name="HO3D",
        split="train",
    )

    # In process_data_item (or any map fn), after computing indices_keep:
    occ = loader.load(key, indices=indices_keep)  # (N_kept, 778) uint8 or None
    if occ is not None:
        res["occ_labels"] = occ
"""

import os

import numpy as np


def _key_to_filename(key: str) -> str:
    """Convert a WebDataset ``__key__`` string to a safe filename stem."""
    return key.replace("/", "_").replace("\\", "_")


class OcclusionAnnotationLoader:
    """
    Memory-mapped loader for pre-computed per-vertex occlusion labels.

    Parameters
    ----------
    ann_dir : str
        Root annotation directory (e.g. ``"data/annotations/occ_labels"``).
    dataset_name : str
        Dataset name matching the subdirectory (e.g. ``"HO3D"``).
    split : str
        Split name (``"train"`` / ``"test"``).
    """

    def __init__(self, ann_dir: str, dataset_name: str, split: str):
        self.ann_subdir = os.path.join(ann_dir, f"{dataset_name}_{split}")

    def available(self, key: str) -> bool:
        """Return True if the annotation file for *key* exists on disk."""
        stem = _key_to_filename(key)
        return os.path.isfile(os.path.join(self.ann_subdir, f"{stem}.npy"))

    def load(self, key: str, indices=None) -> "np.ndarray | None":
        """
        Load the ``(N_views, 778)`` uint8 occlusion-label array for *key*.

        Parameters
        ----------
        key : str
            WebDataset sample key (e.g. ``"s01/capsulemachine_use_01/00012"``).
        indices : array-like of int | None
            If given, return only the rows corresponding to the selected views
            (e.g. the ``indices_keep`` list from ``MultiviewWebDataset``).
            The returned array has shape ``(len(indices), 778)``.

        Returns
        -------
        np.ndarray of shape ``(N_views, 778)`` uint8, or ``None`` if the
        annotation file is not found.
        """
        stem = _key_to_filename(key)
        path = os.path.join(self.ann_subdir, f"{stem}.npy")
        if not os.path.isfile(path):
            return None
        # mmap_mode='r' gives near-instant loading via OS page cache
        ann = np.load(path, mmap_mode="r").astype(np.uint8)
        if indices is not None:
            return ann[list(indices)]
        return ann

    def load_bits(self, key: str, indices=None):
        """
        Convenience method that unpacks the 2-bit labels into two separate
        boolean arrays.

        Returns
        -------
        self_occ  : (N_views, 778) bool
        other_occ : (N_views, 778) bool
        Or ``(None, None)`` if the annotation file is not found.
        """
        ann = self.load(key, indices)
        if ann is None:
            return None, None
        self_occ  = (ann & 1).astype(bool)
        other_occ = ((ann >> 1) & 1).astype(bool)
        return self_occ, other_occ
