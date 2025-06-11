# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
try:
    from .coco import load_coco_json
except Exception:  # pragma: no cover - coco dataset is optional
    load_coco_json = None

try:
    from .lvis import load_lvis_json, register_lvis_instances
except Exception:  # pragma: no cover - lvis dataset is optional
    pass

try:
    from .register_coco import register_coco_instances
except Exception:  # pragma: no cover - coco dataset is optional
    pass

from . import builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
