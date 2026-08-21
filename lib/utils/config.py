from argparse import Namespace
import os
import yaml

from .logger import logger
from yacs.config import CfgNode as _CN
from copy import deepcopy


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.  override wins on conflicts."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml_with_base(config_file: str) -> dict:
    """Load a YAML file and resolve an optional ``_BASE_`` key.

    ``_BASE_`` supports three forms:

    1. **String** – whole-file merge (original behaviour)::

         _BASE_: path/to/base.yaml

    2. **Dict** – section-scoped imports.  Each key is a top-level config
       section; the value is a path whose file provides that section::

         _BASE_:
           DATASET: data/ho3d_dexycb.yaml
           MODEL:   models/hrnet.yaml

       The base file is loaded and, if it contains the named key at its own
       top level, that sub-tree is extracted; otherwise the whole file is used
       as the section content.  This lets base files be either *wrapped*
       (``DATASET: {TRAIN: …}``) or *flat* (``{TRAIN: …}``).

    3. **List of strings** – ordered whole-file merges (earlier entries are
       overridden by later ones, then by the current file)::

         _BASE_:
           - defaults/common.yaml
           - data/ho3d_dexycb.yaml

    All paths are resolved relative to the directory of *config_file*.
    Base chains are supported at every level (each base file may itself
    have a ``_BASE_`` key).
    """
    config_file = os.path.abspath(config_file)
    base_dir = os.path.dirname(config_file)
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("_BASE_", None)
    if base is None:
        return d

    if isinstance(base, str):
        # Form 1: whole-file merge
        base_d = _load_yaml_with_base(os.path.join(base_dir, base))
        d = _deep_merge_dicts(base_d, d)

    elif isinstance(base, list):
        # Form 3: ordered whole-file merges
        merged: dict = {}
        for entry in base:
            entry_d = _load_yaml_with_base(os.path.join(base_dir, entry))
            merged = _deep_merge_dicts(merged, entry_d)
        d = _deep_merge_dicts(merged, d)

    elif isinstance(base, dict):
        # Form 2: section-scoped imports
        merged_base: dict = {}
        for section_key, file_path in base.items():
            base_d = _load_yaml_with_base(os.path.join(base_dir, file_path))
            # If the base file wraps content under section_key at its top
            # level, unwrap it; otherwise use the whole file as the content.
            section_content = base_d.get(section_key, base_d)
            merged_base[section_key] = section_content
        d = _deep_merge_dicts(merged_base, d)

    else:
        raise ValueError(
            f"_BASE_ must be a str, list[str], or dict[str, str]; got {type(base)}"
        )
    return d


class CN(_CN):

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.recursive_cfg_update()

    def recursive_cfg_update(self):

        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        # self.freeze()
        return self

    def dump(self, *args, **kwargs):

        def change_back(cfg: CN) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return _CN(cfg).dump(*args, **kwargs)


_C = CN(new_allowed=True)

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.MANUAL_SEED = 1
_C.TRAIN.CONV_REPEATABLE = True
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCH = 100
_C.TRAIN.OPTIMIZER = "Adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.SCHEDULER = "StepLR"
_C.TRAIN.LR_DECAY_GAMMA = 0.1
_C.TRAIN.LR_DECAY_STEP = [70]
_C.TRAIN.LOG_INTERVAL = 50
_C.TRAIN.FIND_UNUSED_PARAMETERS = False
_C.TRAIN.GRAD_CLIP_ENABLED = True
_C.TRAIN.GRAD_CLIP = CN(new_allowed=True)
_C.TRAIN.GRAD_CLIP.TYPE = 2
_C.TRAIN.GRAD_CLIP.NORM = 0.001
_C.TRAIN.MIXED_PRECISION = False
_C.TRAIN.LAUNCH_TENSORBOARD = False
_C.TRAIN.AUTOSCALE_LR = True


_C.GPU_ID = 1


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, arg: Namespace = None, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    d = _load_yaml_with_base(config_file)
    cfg.merge_from_other_cfg(CN(d, new_allowed=True))

    if arg is not None:
        # if arg.batch_size is given, it always have higher priority
        # if arg has a field batch_size:
        if arg.batch_size is not None:
            if arg.resume is None:
                logger.warning(f"cfg's batch_size {cfg.TRAIN.BATCH_SIZE} reset to arg.batch_size: {arg.batch_size}")
            cfg.TRAIN.BATCH_SIZE = arg.batch_size
        else:
            arg.batch_size = cfg.TRAIN.BATCH_SIZE

        # Honor TRAIN.VAL_BATCH_SIZE from the YAML when the user left the CLI flag
        # at its default (argparse default is 2). Without this the config key is
        # silently ignored and validation always runs at batch size 2.
        cfg_vbs = getattr(cfg.TRAIN, "VAL_BATCH_SIZE", None)
        if cfg_vbs is not None and getattr(arg, "val_batch_size", None) in (None, 2):
            arg.val_batch_size = int(cfg_vbs)

        # if arg.reload is given, it always have higher priority.
        if arg.reload is not None:
            logger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
            cfg.MODEL.PRETRAINED = arg.reload

        # if arg.gpu_id is not given via CLI, use the YAML value.
        if hasattr(arg, 'gpu_id') and arg.gpu_id is None:
            arg.gpu_id = str(cfg.GPU_ID)

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    cfg: CN = get_config("config/train_bihand2d_fh_pl.yml")
    print(cfg)
    cfg_str = cfg.dump(sort_keys=False)
    with open("tmp/test_dump_cfg.yaml", "w") as f:
        f.write(cfg_str)
