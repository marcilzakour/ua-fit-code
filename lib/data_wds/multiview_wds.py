import webdataset as wds
import braceexpand
import os
import torch
import numpy as np
from typing import List, Union
from torch import distributed as dist
from lib.utils.config import CN
from lib.utils.builder import build_transform
from ..utils.logger import logger
import random
import cv2
from torchvision.io import decode_jpeg, ImageReadMode
from .annotation_loader import OcclusionAnnotationLoader


def _tv_jpeg_decode(key, data):
    """Fast libjpeg-turbo JPEG decode via torchvision.io (~0.70ms vs PIL 1.51ms on a
    512 crop = 2.2x). Returns HWC uint8 RGB numpy — exactly what SimpleTransform3DMultiView
    expects (same as the old '.decode(rgb8)'). Fork-safe (unlike the cv2 decode handler,
    which dead-locked worker forks). Only handles .jpg/.jpeg; PNG falls through to PIL.
    GPU (nvJPEG) is slower here — these crops are small, so launch+H2D overhead dominates."""
    if key.endswith(".jpg") or key.endswith(".jpeg"):
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        img = decode_jpeg(t, mode=ImageReadMode.RGB)        # (3,H,W) uint8
        return np.ascontiguousarray(img.permute(1, 2, 0).numpy())
    return None

# NOTE: 'Interhand' and 'Oakink' restored 2026-06-10 — they were removed in an
# uncommitted edit, which silently un-inverts their extrinsics and corrupts all
# Interhand/OakInk multi-view geometry (POEM-large InterHand 8v: 7.1mm -> 212mm).
# The May-2026 baseline reproductions matched the paper only WITH them present.
INV_EXTR_DATASETS = ['Interhand', 'Arctic', 'Oakink', 'Oakink2']


def expand(s):
    s = os.path.expanduser(os.path.expandvars(s))
    base = os.environ.get("TAR_HTTP_BASE")   # stream tars over HTTP (e.g. tailscale) instead of local disk
    if base and "data/dataset_tars/" in s:
        s = base.rstrip("/") + "/" + s.split("data/dataset_tars/", 1)[1]
    return s


def expand_urls(urls: Union[str, List[str]]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls


class MultiviewWebDataset():

    def __init__(self, cfg, data_preset=None, is_train=True, curriculum_cfg=None):
        self.cfg = cfg
        self.data_split = cfg.DATA_SPLIT
        self.epoch_size = cfg.get("EPOCH_SIZE", None)
        self.data_preset = data_preset if data_preset is not None else cfg.DATA_PRESET
        self.urls = cfg.URLS
        self.name = cfg.URLS.split("/")[-1].split("_")[0]
        self.inv_extr = self.name in INV_EXTR_DATASETS
        self.random_n_views = cfg.get("RANDOM_N_VIEWS", False)
        self.view_range = cfg.get("VIEW_RANGE", None)
        # Efficiency knob: use a FIXED number of views per scene instead of the
        # Gauss(4,2) draw. Variable view counts make each DDP rank's batch a
        # different total #images -> the faster rank idles at the gradient
        # all-reduce barrier (sync bubbles). A constant n makes every batch
        # bs*n images on both ranks -> balanced. Still clamped to view_range and
        # the scene's camera count. None (default) = original random behaviour.
        self.fixed_n_views = cfg.get("FIXED_N_VIEWS", None)
        self.mode = "train" if is_train else "val"
        self.transform = build_transform(cfg=cfg.TRANSFORM, data_preset=self.data_preset, is_train=is_train)
        # Opt-in: recompute the crop box from the (correct) joints_2d instead of the stored
        # bbox_center/bbox_scale. InterHand val tars have corrupt bbox_scale (~10-50x too large)
        # for ~21% of samples -> hand shrinks to a dot -> catastrophic failure. Robust to occluded
        # outlier joints via joints_vis. Default off (no effect on training / other datasets).
        self.recrop_from_joints = cfg.get("RECROP_FROM_JOINTS", False)

        self.curriculum_cfg = curriculum_cfg
        import multiprocessing
        self.current_epoch = multiprocessing.Value('i', 0)

        # Optional: load pre-computed occlusion annotations.
        # Set ANN_DIR in the dataset config to enable.
        ann_dir = cfg.get("ANN_DIR", None)
        self.ann_loader = (
            OcclusionAnnotationLoader(ann_dir, self.name, self.data_split)
            if ann_dir is not None
            else None
        )

        if self.random_n_views:
            assert self.view_range is not None and self.view_range[0] >= 1

        dataset = wds.WebDataset(urls=expand_urls(self.urls),
                                 nodesplitter=wds.split_by_node,
                                 shardshuffle=True,
                                 resampled=False,
                                 empty_check=False)  # allow workers > num_shards
        if is_train:
            dataset = dataset.shuffle(1000)

        dataset = dataset.decode(_tv_jpeg_decode, 'rgb8')  # fast turbo-JPEG; PIL fallback for PNG
        dataset = dataset.map(self.process_data_item)
        self.dataset = dataset
        if self.epoch_size is not None:
            logger.info("Initialized MultiviewWebDataset {}_MV with epoch size {}.".format(self.name, self.epoch_size))
        else:
            logger.info("Initialized MultiviewWebDataset {}_MV for MixedWebDataset with mode {}.".format(
                self.name, self.mode))
        return None

    def get_curriculum_min_views(self, epoch):
        if not self.curriculum_cfg:
            return None
        stages = sorted(
            self.curriculum_cfg,
            key=lambda x: x.get("start_epoch", x.get("START_EPOCH", 0)),
            reverse=True
        )
        for stage in stages:
            stage_epoch = stage.get("start_epoch", stage.get("START_EPOCH", 0))
            if epoch >= stage_epoch:
                return stage.get("max_min_views", stage.get("MAX_MIN_VIEWS", None))
        return None

    def set_epoch(self, epoch):
        self.current_epoch.value = epoch
        min_views = self.get_curriculum_min_views(epoch)
        if not hasattr(self, '_last_logged_min_views') or self._last_logged_min_views != min_views:
            self._last_logged_min_views = min_views
            if min_views is not None:
                logger.info(f"[{self.name} Dataset] Curriculum Active: min_views = {min_views} at epoch {epoch}")

    def get_current_min_views(self):
        return self.get_curriculum_min_views(self.current_epoch.value)

    def process_data_item(self, item):
        n_view_imgs = {}
        for k in item.keys():
            if k.startswith("image"):
                img_type = "png" if "png" in k else "jpg"
                n_view_imgs[k] = item[k]

        n_cams = len(n_view_imgs)
        # dict of list
        # eg: key: [value_0, value_1, ...]
        key = item["__key__"]
        labels = item["label.pyd"]

        if "mano_pose" in labels:
            labels["mano_pose"] = [labels["mano_pose"][i].reshape(-1)[:48].reshape(16, 3) for i in range(n_cams)]
        else:
            # This is used as a temporary solution that deals with the case of Oakink dataset, should be fixed soon once the dataset is dumped properly
            labels["mano_pose"] = [np.zeros((16, 3)) for i in range(n_cams)]
            labels["mano_shape"] = [np.zeros(10) for _ in range(n_cams)]
        if self.inv_extr:
            labels['cam_extr'] = [np.linalg.inv(labels['cam_extr'][i]) for i in range(n_cams)]

        # random shuffle the camera idx
        indices = [i for i in range(0, n_cams)]
        if self.random_n_views:
            random.shuffle(indices)
            # randomly select from 1 to n ind from indices
            if self.fixed_n_views is not None:
                n = int(self.fixed_n_views)          # balanced-batch mode (constant views/scene)
            else:
                n = int(round(random.gauss(4, 2)))

            # Apply curriculum views constraint
            min_view_limit = self.view_range[0]
            curriculum_min = self.get_current_min_views()
            if curriculum_min is not None:
                min_view_limit = max(min_view_limit, curriculum_min)
                
            n = min(max(min_view_limit, n), self.view_range[1])
            n = min(n, n_cams)  # in case the range is larger than the number of cameras
            indices_keep = indices[:n]
        else:
            indices_keep = indices

        new_master_id = indices_keep[0]
        new_master_serial = labels["cam_serial"][new_master_id]
        T_master_2_new_master = labels["cam_extr"][new_master_id]
        master_joints_3d = labels["joints_3d"][new_master_id]
        master_verts_3d = labels["verts_3d"][new_master_id]

        res = {}
        for ind in indices_keep:
            img = n_view_imgs[f"image_{ind}.{img_type}"]
            # print(labels["request_flip"])
            if labels.get("request_flip", False):
                cam_intr = labels["cam_intr"][ind]
                raw_size = labels["raw_size"][ind]
                cam_center = np.array([cam_intr[0, 2], cam_intr[1, 2]])
                M = np.array([[-1, 0, 2 * cam_center[0]], [0, 1, 0]], dtype=np.float32)
                # Use warpAffine to apply the reflection
                img = cv2.warpAffine(img, M, raw_size)

            lab = {k: v[ind] for k, v in labels.items() if k not in ["request_flip"]}
            # Fix corrupt stored bbox: recompute crop center/scale from visible joints_2d.
            if self.recrop_from_joints and "joints_2d" in lab:
                j2 = np.asarray(lab["joints_2d"], dtype=np.float32).reshape(-1, 2)
                vis = np.asarray(lab.get("joints_vis", np.ones(len(j2)))).reshape(-1)
                sel = j2[vis > 0.5] if int((vis > 0.5).sum()) >= 3 else j2
                lo = sel.min(0); hi = sel.max(0)
                lab["bbox_center"] = ((lo + hi) / 2.0).astype(np.float32)
                lab["bbox_scale"] = np.float32(max(float((hi - lo).max()) * 2.0, 60.0))
            # data aug
            tgt = self.transform(img, lab, no_rot=ind == new_master_id)

            # deal with camera extr
            T_m2c = lab["cam_extr"]
            T_new_master_2_cam = np.linalg.inv(T_master_2_new_master) @ T_m2c
            extr_prerot = tgt["extr_prerot"]  # (3, 3)
            extr_pre_transf = np.concatenate([extr_prerot, np.zeros((3, 1))], axis=1)
            extr_pre_transf = np.concatenate([extr_pre_transf, np.array([[0, 0, 0, 1]])], axis=0)
            T_new_master_2_cam = np.linalg.inv(extr_pre_transf @ np.linalg.inv(T_new_master_2_cam))
            tgt["target_cam_extr"] = T_new_master_2_cam.astype(np.float32)

            tgt.update(lab)
            for k, v in tgt.items():
                if k not in res:
                    res[k] = []
                res[k].append(v)

        for query in res.keys():
            if isinstance(res[query][0], (int, float, np.ndarray, torch.Tensor)):
                res[query] = np.stack(res[query])

        res["master_id"] = 0
        res["master_serial"] = new_master_serial
        res["master_joints_3d"] = master_joints_3d
        res["master_verts_3d"] = master_verts_3d
        res["__key__"] = key
        # Per-sample source-dataset tag (e.g. "HO3D"/"DexYCB"/"Arctic"). Routed by
        # collation_random_n_views into a per-sample list so the model can bucket
        # validation metrics per dataset (the mixed val loader interleaves them).
        res["dataset_name"] = self.name

        # Load occlusion annotation if an annotation directory is configured.
        # occ_labels: (N_kept_views, 778) uint8  — values 0-3
        #   bit 0 (& 1) : self-occlusion
        #   bit 1 (& 2) : other-occlusion
        if self.ann_loader is not None:
            occ = self.ann_loader.load(key, indices=indices_keep)
            if occ is not None:
                res["occ_labels"] = occ   # (N_views, 778) uint8
            else:
                # Placeholder so the key is ALWAYS present when ANN_DIR is configured —
                # collation_random_n_views concatenates batch[0]'s keys and would KeyError
                # on mixed presence. 255 = "unknown", masked out of any occ-supervised loss.
                res["occ_labels"] = np.full((len(indices_keep), 778), 255, dtype=np.uint8)

        return res

    def get_dataset(self):
        return self.dataset
