# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Inject QuaRot-style online inverse Hadamard rotation into GLM-5.3-Flash
#     without touching upstream vLLM sources.
#
#     Two insertion points (per the internal spec):
#
#       1. Main trunk  — Glm5NextModel: the residual stream is *already summed*
#          into `hidden_states` when the final norm runs (mHC layers fuse the
#          residual add).  Upstream forward ends with a single-argument call:
#          `hidden_states = self.norm(hidden_states)`.
#          → rotate `args[0]` before that norm.
#
#       2. MTP draft   — Glm5NextMultiTokenPredictorLayer: the draft bypasses
#          `SharedHead.forward` and calls
#          `self.shared_head.norm(hidden_states, residual=residual)` directly,
#          i.e. the residual add happens INSIDE RMSNorm.
#          → rotate `hidden_states + residual` before that norm, passing the
#            already-summed value in with residual=None.
#
#     Why a forward *pre-hook* instead of replacing `self.norm`:
#       `nn.Module.__setattr__` refuses to assign a non-Module callable onto a
#       name already registered in `self._modules` (raises TypeError).  A
#       pre-hook mutates neither the module tree nor the state_dict, so
#       checkpoint keys such as `...norm.weight` stay byte-for-byte intact.
#
#     The rotation module is registered as a submodule under the exact name the
#     config and checkpoint use, so `sign` / `hk` load as ordinary weights with
#     no bespoke loader:
#        trunk : <prefix>.prenorm_hadamard_rotation
#        MTP   : <prefix>.shared_head.prenorm_hadamard_rotation
#
# Affected versions: vLLM 0.27.2rc1.dev22 on MetaX MACA, reviewed 2026-10-15.
#
# Prerequisites (already present in the installed vllm_metax):
#     - vllm_metax.model_executor.layers.online_rotation
#       (build_rotation, get_online_rotation_module_config)
#     - vllm_metax.model_executor.kernels.hadamard
#
# Remove at: upstream GLM supports online rotation natively.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_ROTATION_LEAF = "prenorm_hadamard_rotation"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _maybe_prefix(prefix: str, leaf: str) -> str:
    from vllm.model_executor.models.utils import maybe_prefix
    return maybe_prefix(prefix, leaf)


def _build_rotation_if_present(config: Any, module_name: str) -> nn.Module | None:
    """Build the rotation mounted at `module_name` in config.online_rotation."""
    from vllm_metax.model_executor.layers.online_rotation import (
        build_rotation,
        get_online_rotation_module_config,
    )
    rot_cfg = get_online_rotation_module_config(config, module_name)
    if rot_cfg is None:
        return None
    logger.info("GLM5Next online rotation enabled at %s", module_name)
    return build_rotation(rot_cfg)


def _get_original(patched_func: Any, full_target_name: str) -> Any:
    """Fetch the original method that the patch decorator recorded before replacing."""
    marker = getattr(patched_func, "__vllm_metax_patch__", None)
    if not isinstance(marker, dict):
        raise RuntimeError(
            f"patch marker missing on {patched_func!r} — did patch() run?"
        )
    original = marker.get(full_target_name)
    if original is None:
        raise KeyError(
            f"original for {full_target_name!r} not recorded; keys={list(marker)}"
        )
    return original


# ---------------------------------------------------------------------------
# pre-hook factories
# ---------------------------------------------------------------------------

def _trunk_pre_hook(rotation: nn.Module) -> Callable:
    """Rotate the single argument of the trunk's final norm.

    The trunk's residual stream is already summed when norm runs, and the call
    carries no residual argument, so we rotate args[0] as-is.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict):
        if not args:
            return None
        return (rotation(args[0]),) + tuple(args[1:]), kwargs

    return hook


def _mtp_pre_hook(rotation: nn.Module) -> Callable:
    """Rotate `hidden_states + residual` before the MTP shared_head norm.

    RMSNorm performs the hidden+residual sum internally when `residual` is
    supplied, so we fold it here and hand the norm a pre-summed tensor with
    `residual=None`.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict):
        if not args:
            return None
        x = args[0]
        rest = tuple(args[1:])
        residual = kwargs.get("residual", None)
        if residual is None and rest:
            # residual passed positionally
            residual = rest[0]
            rest = rest[1:]
        new_kwargs = dict(kwargs)
        if residual is not None:
            x = x + residual
            new_kwargs["residual"] = None
        return (rotation(x),) + rest, new_kwargs

    return hook


def _install_hook(norm: Any, rotation: nn.Module, factory: Callable) -> None:
    if not isinstance(norm, nn.Module):
        logger.warning(
            "glm5next_rotation: expected an nn.Module at the norm slot, got %r; "
            "rotation not installed.",
            type(norm),
        )
        return
    if getattr(norm, "_glm5next_rotation_hooked", False):
        return
    norm.register_forward_pre_hook(factory(rotation), with_kwargs=True)
    norm._glm5next_rotation_hooked = True


def _resolve_rotation(
    configs: tuple[Any, ...],
    module_name: str,
    existing: Any,
) -> tuple[nn.Module | None, bool]:
    """Return (rotation, already_upstream_managed).

    If the target object already carries a rotation module, upstream owns it and
    we must not build a second one; otherwise we build from config.
    """
    if existing is not None:
        return existing, True
    for cfg in configs:
        if cfg is None:
            continue
        rotation = _build_rotation_if_present(cfg, module_name)
        if rotation is not None:
            return rotation, False
    return None, False


# ---------------------------------------------------------------------------
# Patch 1 — Glm5NextModel (main trunk)
# ---------------------------------------------------------------------------
# vLLM layouts seen in this image: nvidia/ is the active namespace; common/
# exists as a fallback.  Each candidate is attempted independently.

_TRUNK_CANDIDATES = [
    ("vllm.models.glm5next.nvidia.model", "Glm5NextModel"),
    ("vllm.models.glm5next.common.mtp", "Glm5NextModel"),
]

_MTP_CANDIDATES = [
    ("vllm.models.glm5next.nvidia.mtp", "Glm5NextMultiTokenPredictorLayer"),
    ("vllm.models.glm5next.nvidia.model", "Glm5NextMultiTokenPredictorLayer"),
    ("vllm.models.glm5next.common.mtp", "Glm5NextMultiTokenPredictorLayer"),
]


def _make_trunk_init(full_init: str) -> Callable:
    """Build the trunk __init__ replacement, binding its target name eagerly."""

    def _patched_trunk_init(self, vllm_config, prefix: str = "", *args, **kwargs):
        original_init = _get_original(_patched_trunk_init, full_init)
        original_init(self, vllm_config, prefix, *args, **kwargs)

        config = vllm_config.model_config.hf_config
        module_name = _maybe_prefix(prefix, _ROTATION_LEAF)
        rotation, upstream_managed = _resolve_rotation(
            (config,), module_name, getattr(self, _ROTATION_LEAF, None)
        )
        if rotation is None:
            return
        if upstream_managed:
            # Upstream already applies rotation inside forward; adding a hook
            # here would rotate twice.
            return

        # Register under the config/checkpoint name so sign/hk load as ordinary
        # weights with no bespoke loader.
        setattr(self, _ROTATION_LEAF, rotation)
        _install_hook(getattr(self, "norm", None), rotation, _trunk_pre_hook)

    return _patched_trunk_init


def _make_mtp_init(full_init: str) -> Callable:
    """Build the MTP __init__ replacement, binding its target name eagerly."""

    def _patched_mtp_init(self, vllm_config, prefix, *args, **kwargs):
        original_init = _get_original(_patched_mtp_init, full_init)
        original_init(self, vllm_config, prefix, *args, **kwargs)

        shared_head = getattr(self, "shared_head", None)
        if shared_head is None:
            logger.warning(
                "glm5next_rotation: MTP layer has no `shared_head`; "
                "rotation not installed."
            )
            return

        draft_cfg = None
        try:
            draft_cfg = (
                vllm_config.speculative_config.draft_model_config.hf_text_config
            )
        except AttributeError:
            pass
        main_cfg = vllm_config.model_config.hf_config

        module_name = f"{prefix}.shared_head.{_ROTATION_LEAF}"
        rotation, upstream_managed = _resolve_rotation(
            (draft_cfg, main_cfg),
            module_name,
            getattr(shared_head, _ROTATION_LEAF, None),
        )
        if rotation is None:
            return
        if not upstream_managed:
            setattr(shared_head, _ROTATION_LEAF, rotation)

        # The draft never calls SharedHead.forward, so the hook is required even
        # when upstream built the module.
        _install_hook(
            getattr(shared_head, "norm", None), rotation, _mtp_pre_hook
        )

    return _patched_mtp_init


def _install_candidates(candidates, factory, label: str) -> None:
    """Probe each candidate location and install at the first that resolves.

    Note: `patch()` performs its `import_module` when the returned decorator is
    *applied*, not when `patch()` is called — so the whole apply step has to sit
    inside the try block for a missing module to be caught as a skip.
    """
    from vllm_metax.patch.utils import patch

    for mod_path, cls_name in candidates:
        target = f"{cls_name}.__init__"
        try:
            decorator = patch(mod_path, target)
            decorator(factory(f"{mod_path}.{target}"))
        except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
            logger.debug(
                "glm5next_rotation: %s skip %s (%s: %s)",
                label, mod_path, type(exc).__name__, exc,
            )
            continue

        logger.info("glm5next_rotation: %s patched at %s", label, mod_path)


# ---------------------------------------------------------------------------
# Install (fires at import time, driven by patch/enhancement/__init__.py)
# ---------------------------------------------------------------------------

_install_candidates(_TRUNK_CANDIDATES, _make_trunk_init, "trunk")
_install_candidates(_MTP_CANDIDATES, _make_mtp_init, "mtp")
