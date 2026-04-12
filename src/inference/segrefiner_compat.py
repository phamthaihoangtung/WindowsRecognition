"""
mmcv 1.x → mmcv 2.x + mmengine compatibility shims for SegRefiner.

SegRefiner's bundled mmdet was written against mmcv 1.x. In mmcv 2.x, most of
mmcv.runner and mmcv.utils moved to mmengine. This module patches the missing
symbols into mmcv's namespace before any SegRefiner import.

Import this module ONCE at the top of segrefiner_processor.py, before anything
from segrefiner/ or mmdet is imported.
"""
import os
import sys
import types
import mmcv

# ── 1. Version spoof ─────────────────────────────────────────────────────────
# SegRefiner's mmdet/__init__.py asserts mmcv <= 1.8.0.
mmcv.__version__ = "1.7.1"

# ── 2. mmengine imports ───────────────────────────────────────────────────────
from mmengine.model import BaseModule, Sequential, ModuleList
from mmengine.runner import load_checkpoint
from mmengine.dist import get_dist_info
from mmengine.registry import Registry, build_from_cfg, MODELS as _MMENGINE_MODELS
from mmengine.logging import print_log
from mmengine.utils import digit_version

# ── 3. mmcv.runner ───────────────────────────────────────────────────────────
def _noop_decorator(*args, **kwargs):
    """No-op replacement for auto_fp16 / force_fp32 decorators."""
    def decorator(fn):
        return fn
    # Handle both @auto_fp16 and @auto_fp16(apply_to=(...)) call patterns
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator

_runner = types.ModuleType("mmcv.runner")
_runner.BaseModule = BaseModule
_runner.Sequential = Sequential
_runner.ModuleList = ModuleList
_runner.load_checkpoint = load_checkpoint
_runner.get_dist_info = get_dist_info
_runner.auto_fp16 = _noop_decorator
_runner.force_fp32 = _noop_decorator

# Stub Hook / HOOKS / EvalHook used by mmdet.core but not called at inference
class _Hook:
    pass

_HOOKS = Registry("hooks")

_runner.Hook = _Hook
_runner.HOOKS = _HOOKS
_runner.EvalHook = _Hook
_runner.DistEvalHook = _Hook

# Stub hooks sub-package
_runner_hooks = types.ModuleType("mmcv.runner.hooks")
_runner_hooks.Hook = _Hook
_runner_hooks.HOOKS = _HOOKS

_runner_hooks_lr = types.ModuleType("mmcv.runner.hooks.lr_updater")
_runner_hooks_lr.CosineAnnealingLrUpdaterHook = _Hook
_runner_hooks_lr.LinearAnnealingLrUpdaterHook = _Hook

_runner_hooks_ckpt = types.ModuleType("mmcv.runner.hooks.checkpoint")
_runner_hooks_ckpt.CheckpointHook = _Hook

_runner_hooks_logger = types.ModuleType("mmcv.runner.hooks.logger")
_runner_hooks_logger_wandb = types.ModuleType("mmcv.runner.hooks.logger.wandb")
_runner_hooks_logger_wandb.WandbLoggerHook = _Hook

_runner_dist = types.ModuleType("mmcv.runner.dist_utils")
_runner_dist.master_only = lambda fn: fn

sys.modules["mmcv.runner"] = _runner
sys.modules["mmcv.runner.hooks"] = _runner_hooks
sys.modules["mmcv.runner.hooks.lr_updater"] = _runner_hooks_lr
sys.modules["mmcv.runner.hooks.checkpoint"] = _runner_hooks_ckpt
sys.modules["mmcv.runner.hooks.logger"] = _runner_hooks_logger
sys.modules["mmcv.runner.hooks.logger.wandb"] = _runner_hooks_logger_wandb
sys.modules["mmcv.runner.dist_utils"] = _runner_dist
mmcv.runner = _runner

# ── 4. mmcv.utils ────────────────────────────────────────────────────────────
import mmcv.utils as _mmcv_utils
_mmcv_utils.Registry = Registry
_mmcv_utils.build_from_cfg = build_from_cfg
_mmcv_utils.print_log = print_log
_mmcv_utils.digit_version = digit_version

# ── 5. mmcv misc stubs ───────────────────────────────────────────────────────
# mmcv.jit was a TorchScript/coderize decorator removed in 2.x — make it a no-op
mmcv.jit = lambda *a, **kw: (lambda fn: fn) if not (len(a) == 1 and callable(a[0]) and not kw) else a[0]

# ── 5b. mmcv.cnn ─────────────────────────────────────────────────────────────
from mmengine.model import (
    bias_init_with_prob, caffe2_xavier_init, constant_init,
    normal_init, trunc_normal_init, xavier_init,
)

import mmcv.cnn as _mmcv_cnn
_mmcv_cnn.MODELS = _MMENGINE_MODELS
_mmcv_cnn.CONV_LAYERS = Registry("conv_layers")
_mmcv_cnn.PLUGIN_LAYERS = Registry("plugin_layers")
_mmcv_cnn.bias_init_with_prob = bias_init_with_prob
_mmcv_cnn.caffe2_xavier_init = caffe2_xavier_init
_mmcv_cnn.constant_init = constant_init
_mmcv_cnn.normal_init = normal_init
_mmcv_cnn.trunc_normal_init = trunc_normal_init
_mmcv_cnn.xavier_init = xavier_init

# ── 6. mmdet heavy sub-package stubs ────────────────────────────────────────
# SegRefiner only uses: mmdet.models.detectors (SegRefinerSemantic/Base),
# mmdet.models.dense_heads (DenoiseUNet), and mmdet.models.builder.
# Everything else (backbones, necks, roi_heads, losses, seg_heads, plugins,
# mmdet.core, mmdet.utils) has mmcv 1.x-only imports we don't need at
# inference time. Stub them all out BEFORE mmdet.models is imported.
for _stub_name in [
    "mmdet.models.backbones",
    "mmdet.models.utils",
    "mmdet.models.dense_heads",
    "mmdet.models.detectors",
    "mmdet.models.necks",
    "mmdet.models.losses",
    "mmdet.models.plugins",
    "mmdet.models.roi_heads",
    "mmdet.models.seg_heads",
    "mmdet.core",
    "mmdet.core.bbox",
    "mmdet.core.anchor",
    "mmdet.core.mask",
    "mmdet.core.evaluation",
    "mmdet.core.export",
    "mmdet.core.hook",
    "mmdet.core.post_processing",
    "mmdet.core.utils",
    "mmdet.utils",
]:
    _m = types.ModuleType(_stub_name)
    _m.__all__ = []
    sys.modules[_stub_name] = _m

# BitmapMasks is needed at inference — load directly from file before stubs
import importlib.util as _ilu
_SEGREFINER_ROOT_COMPAT = os.path.join(os.path.dirname(__file__), "../../segrefiner")
_bm_spec = _ilu.spec_from_file_location(
    "mmdet.core.mask.structures",
    os.path.join(_SEGREFINER_ROOT_COMPAT, "mmdet/core/mask/structures.py"),
)
_bm_mod = _ilu.module_from_spec(_bm_spec)
_bm_spec.loader.exec_module(_bm_mod)
_BitmapMasks = _bm_mod.BitmapMasks
sys.modules["mmdet.core.mask"].BitmapMasks = _BitmapMasks

# DenoiseUNet lives in dense_heads but only uses mmcv.runner.BaseModule + HEADS.
# Import it directly from its file so it gets registered without pulling in the
# rest of dense_heads (which has many mmcv 1.x dependencies).
_du_spec = _ilu.spec_from_file_location(
    "mmdet.models.dense_heads.diffusion_unet_head",
    os.path.join(_SEGREFINER_ROOT_COMPAT, "mmdet/models/dense_heads/diffusion_unet_head.py"),
)
_du_mod = _ilu.module_from_spec(_du_spec)
sys.modules["mmdet.models.dense_heads.diffusion_unet_head"] = _du_mod
_du_spec.loader.exec_module(_du_mod)  # triggers @HEADS.register_module()

# Losses: SegRefiner uses CrossEntropyLoss — load it + its utils directly.
for _loss_file, _loss_mod_name in [
    ("mmdet/models/losses/utils.py", "mmdet.models.losses.utils"),
    ("mmdet/models/losses/cross_entropy_loss.py", "mmdet.models.losses.cross_entropy_loss"),
    ("mmdet/models/losses/textrue_l1_loss.py", "mmdet.models.losses.textrue_l1_loss"),
]:
    _spec = _ilu.spec_from_file_location(
        _loss_mod_name, os.path.join(_SEGREFINER_ROOT_COMPAT, _loss_file)
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_loss_mod_name] = _mod
    _spec.loader.exec_module(_mod)

# SegRefiner detectors: load base then semantic directly (same reason as above).
for _det_file, _det_mod_name in [
    ("mmdet/models/detectors/segrefiner_base.py", "mmdet.models.detectors.segrefiner_base"),
    ("mmdet/models/detectors/segrefiner_semantic.py", "mmdet.models.detectors.segrefiner_semantic"),
    ("mmdet/models/detectors/segrefiner_instance.py", "mmdet.models.detectors.segrefiner_instance"),
]:
    _spec = _ilu.spec_from_file_location(
        _det_mod_name, os.path.join(_SEGREFINER_ROOT_COMPAT, _det_file)
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_det_mod_name] = _mod
    _spec.loader.exec_module(_mod)  # triggers @DETECTORS.register_module()

# mmdet.utils needs get_root_logger used by some model files
import logging as _logging
sys.modules["mmdet.utils"].get_root_logger = lambda name="mmdet": _logging.getLogger(name)
sys.modules["mmdet.utils"].util_mixins = types.ModuleType("mmdet.utils.util_mixins")
sys.modules["mmdet.utils"].register_all_modules = lambda init_default_scope=True: None

# ── 7. mmcv.parallel ─────────────────────────────────────────────────────────
import torch.nn as _nn

def _is_module_wrapper(module):
    return isinstance(module, (_nn.DataParallel, _nn.parallel.DistributedDataParallel))

_parallel = types.ModuleType("mmcv.parallel")
_parallel.is_module_wrapper = _is_module_wrapper
sys.modules["mmcv.parallel"] = _parallel
mmcv.parallel = _parallel
