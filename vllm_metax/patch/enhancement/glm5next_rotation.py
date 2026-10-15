# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: QuaRot-style online inverse rotation for GLM-5.3-Flash.  Injected at
#     runtime so upstream vLLM sources stay untouched.
#
#     Two insertion points (per the internal spec):
#
#       1. main trunk — `Glm5NextModel`.  Its forward ends with the residual
#          stream already summed (the mHC layers fuse the add) and the final
#          norm takes a single argument:
#              hidden_states = self.norm(hidden_states)          # model.py:721
#          → rotate the norm's input.
#
#       2. MTP draft — `Glm5NextMultiTokenPredictorLayer`.  The draft bypasses
#          `SharedHead.forward` and drives the fused norm directly:
#              hidden_states, _ = self.shared_head.norm(
#                  hidden_states, residual=residual)              # mtp.py:121
#          i.e. the residual add happens *inside* RMSNorm.  The rotation is
#          linear, so rotating `hidden_states` and `residual` separately is
#          equivalent to rotating their sum, while the fused kernel keeps its
#          own dtype handling and return values exactly as upstream.
#
#     Mechanism: a forward *pre-hook* on the final RMSNorm.  Replacing the
#     callable instead would trip `nn.Module.__setattr__`, which refuses to bind
#     a non-Module onto a name already present in `self._modules`; the pre-hook
#     leaves both the module tree and the state_dict keys untouched.
#
#     The rotation module is mounted under the exact name the config and the
#     checkpoint use, so `sign` / `hk` load as ordinary weights:
#         trunk : <prefix>.prenorm_hadamard_rotation
#         MTP   : <prefix>.shared_head.prenorm_hadamard_rotation
#
# Affected versions: vLLM 0.27.2rc1.dev22 on MetaX MACA.
# Requires: vllm_metax.model_executor.layers.online_rotation and
#     vllm_metax.model_executor.kernels.hadamard.
# Remove at: upstream GLM supports online rotation natively.
# -----------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_metax.model_executor.layers.online_rotation import build_rotation
from vllm_metax.patch.utils import PATCH_MARKER, patch

logger = init_logger(__name__)

# config / checkpoint leaf for the trunk, and its `shared_head`-anchored twin.
_ROTATION_LEAF = "prenorm_hadamard_rotation"
_TRUNK_SUFFIX = _ROTATION_LEAF
_MTP_SUFFIX = f"shared_head.{_ROTATION_LEAF}"

_HOOK_FLAG = "_glm5next_rotation_hooked"


# ---------------------------------------------------------------------------
# rotation lookup
# ---------------------------------------------------------------------------

def _rotation_modules(config: Any) -> Mapping[str, Any]:
    """Return `config.online_rotation.modules`, or an empty mapping."""
    rotation_config = getattr(config, "online_rotation", None)
    if not isinstance(rotation_config, Mapping):
        return {}
    modules = rotation_config.get("modules")
    return modules if isinstance(modules, Mapping) else {}


def _resolve_key(
    modules: Mapping[str, Any], prefix: str, leaf_suffix: str
) -> tuple[str | None, Any]:
    """Return `(config_key, spec)` for the rotation mounted at `prefix`.

    The runtime `prefix` and the exporter-facing key can disagree about the path
    in between, because a multimodal config anchors the language model one level
    deeper than vLLM's draft module does.  Concretely, for layer 45 this patch
    may see prefix `model.layers.45` while the config carries
    `model.language_model.layers.45.shared_head.prenorm_hadamard_rotation`.

    So the exact lookup is backed by two progressively weaker anchors, and only
    a unique best match is accepted:
       rank 1 — the key's trailing segments equal the runtime prefix;
       rank 2 — the key ends on the same trailing path segment (the layer index),
                which bridges the extra `language_model` level.
    """
    exact = f"{prefix}.{leaf_suffix}" if prefix else leaf_suffix
    if exact in modules:
        return exact, modules[exact]

    head = prefix.split(".") if prefix else []
    candidates: list[tuple[int, str]] = []
    for key in modules:
        if not key.endswith("." + leaf_suffix):
            continue
        segments = key[: -(len(leaf_suffix) + 1)].split(".")
        if head and segments[-len(head):] == head:
            candidates.append((1, key))
        elif head and segments and segments[-1] == head[-1]:
            candidates.append((2, key))

    best_rank = min((rank for rank, _ in candidates), default=None)
    winners = [key for rank, key in candidates if rank == best_rank]
    if len(winners) == 1:
        return winners[0], modules[winners[0]]
    return None, None


def _mount_rotation(
    configs: list[Any], prefix: str, leaf_suffix: str
) -> nn.Module | None:
    """Build the rotation for one insertion point, or `None` if not configured."""
    available: dict[str, Any] = {}
    for config in configs:
        modules = _rotation_modules(config)
        if not modules:
            continue
        available.update(modules)
        key, spec = _resolve_key(modules, prefix, leaf_suffix)
        if spec is not None:
            logger.info("glm5next_rotation: mounting rotation (config key %r)", key)
            return build_rotation(spec)

    if available:
        # The config asks for online rotation but nothing lines up with this
        # insertion point, which silently disables the inverse rotation.
        logger.warning(
            "glm5next_rotation: no rotation key matches prefix %r (leaf %r); "
            "available keys %s. Rotation NOT installed here.",
            prefix, leaf_suffix, sorted(available),
        )
    else:
        logger.debug(
            "glm5next_rotation: config has no online_rotation; nothing to mount "
            "for prefix %r.",
            prefix,
        )
    return None


# ---------------------------------------------------------------------------
# pre-hook
# ---------------------------------------------------------------------------

def _norm_input_pre_hook(rotation: nn.Module) -> Callable:
    """Rotate every tensor the final RMSNorm consumes.

    Upstream feeds these norms either `hidden_states` alone (trunk, sum already
    done) or `hidden_states` plus a `residual` the kernel adds internally (MTP).
    Both are linear inputs of the same normalization, so rotating each tensor
    separately equals rotating their sum — and this keeps dtype promotion and
    return values identical to upstream.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict):
        if not args:
            return None
        new_args = tuple(
            rotation(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        residual = kwargs.get("residual")
        if isinstance(residual, torch.Tensor):
            kwargs = {**kwargs, "residual": rotation(residual)}
        return new_args, kwargs

    return hook


def _install_pre_hook(norm: Any, rotation: nn.Module) -> None:
    if getattr(norm, _HOOK_FLAG, False):
        return
    norm.register_forward_pre_hook(_norm_input_pre_hook(rotation), with_kwargs=True)
    setattr(norm, _HOOK_FLAG, True)


def _final_norm(owner: Any) -> Any:
    """Return the final RMSNorm of `owner`, or `None` when it has none.

    Pipeline-parallel stages that do not own the final norm hold a
    `PPMissingLayer` there, and the MTP `shared_head` may hold one too.
    """
    norm = getattr(owner, "norm", None)
    return norm if isinstance(norm, RMSNorm) else None


def _candidate_configs(vllm_config: Any) -> list[Any]:
    """Collect the HF configs that may carry `online_rotation`.

    The MTP draft reads the rotation from its own (draft) config; the trunk
    reads it from the target model's config.  Both `hf_config` and
    `hf_text_config` are probed because a multimodal config may put the
    rotation under `text_config` only.
    """
    configs: list[Any] = []
    draft_config = getattr(
        getattr(vllm_config, "speculative_config", None), "draft_model_config", None
    )
    model_config = getattr(vllm_config, "model_config", None)
    for config in (
        getattr(draft_config, "hf_config", None),
        getattr(draft_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
    ):
        if config is not None and not any(config is seen for seen in configs):
            configs.append(config)
    return configs


# ---------------------------------------------------------------------------
# __init__ replacements
# ---------------------------------------------------------------------------

def _original(patched: Callable, full_target: str) -> Callable:
    """Read back the method `patch()` displaced when it installed `patched`."""
    originals = getattr(patched, PATCH_MARKER, None)
    if not isinstance(originals, dict) or full_target not in originals:
        raise RuntimeError(
            f"glm5next_rotation: patch() did not record {full_target!r}"
        )
    return originals[full_target]


def _positional_prefix(args: tuple, kwargs: dict) -> str:
    """Recover `prefix` from either calling convention.

    `Glm5NextModel.__init__` is keyword-only (`vllm_config=`, `prefix=`), while
    `Glm5NextMultiTokenPredictorLayer.__init__` takes both positionally.
    """
    if "prefix" in kwargs:
        return kwargs["prefix"]
    return args[1] if len(args) > 1 else ""


def _positional_vllm_config(args: tuple, kwargs: dict) -> Any:
    return kwargs["vllm_config"] if "vllm_config" in kwargs else args[0]


def _make_trunk_init(full_target: str) -> Callable:
    """Wrap `Glm5NextModel.__init__` and hook the trunk's final norm."""

    def _glm5next_model_init(self: nn.Module, *args, **kwargs):
        _original(_glm5next_model_init, full_target)(self, *args, **kwargs)

        norm = _final_norm(self)
        if norm is None:
            logger.debug(
                "glm5next_rotation: trunk has no final RMSNorm on this stage; "
                "rotation not installed."
            )
            return

        prefix = _positional_prefix(args, kwargs)
        rotation = _mount_rotation(
            _candidate_configs(_positional_vllm_config(args, kwargs)),
            prefix,
            _TRUNK_SUFFIX,
        )
        if rotation is None:
            return

        setattr(self, _ROTATION_LEAF, rotation)
        _install_pre_hook(norm, rotation)

    return _glm5next_model_init


def _make_mtp_init(full_target: str) -> Callable:
    """Wrap `Glm5NextMultiTokenPredictorLayer.__init__` and hook the MTP norm.

    The rotation sits on `shared_head`, matching the config and checkpoint key
    `...<prefix>.shared_head.prenorm_hadamard_rotation`, so it is mounted there
    rather than on the MTP layer.
    """

    def _glm5next_mtp_init(self: nn.Module, *args, **kwargs):
        _original(_glm5next_mtp_init, full_target)(self, *args, **kwargs)

        shared_head = getattr(self, "shared_head", None)
        if shared_head is None:
            logger.warning(
                "glm5next_rotation: MTP layer has no `shared_head`; "
                "rotation not installed."
            )
            return

        norm = _final_norm(shared_head)
        if norm is None:
            logger.debug(
                "glm5next_rotation: MTP shared_head has no RMSNorm on this stage; "
                "rotation not installed."
            )
            return

        prefix = _positional_prefix(args, kwargs)
        rotation = _mount_rotation(
            _candidate_configs(_positional_vllm_config(args, kwargs)),
            prefix,
            _MTP_SUFFIX,
        )
        if rotation is None:
            return

        setattr(shared_head, _ROTATION_LEAF, rotation)
        _install_pre_hook(norm, rotation)

    return _glm5next_mtp_init


# ---------------------------------------------------------------------------
# install (fires at import time, driven by patch/enhancement/__init__.py)
# ---------------------------------------------------------------------------

# Verified against the container image: `vllm/models/glm5next/nvidia/` is the
# active namespace, `Glm5NextModel` lives in `model.py` and
# `Glm5NextMultiTokenPredictorLayer` in `mtp.py`.
_TARGETS = [
    ("vllm.models.glm5next.nvidia.model", "Glm5NextModel.__init__",
     _make_trunk_init, "trunk"),
    ("vllm.models.glm5next.nvidia.mtp", "Glm5NextMultiTokenPredictorLayer.__init__",
     _make_mtp_init, "mtp"),
]


def _install(mod_path: str, attribute: str, factory: Callable, label: str) -> None:
    """Install one patch, reporting a missing target instead of aborting import."""
    full_target = f"{mod_path}.{attribute}"
    try:
        patch(mod_path, attribute)(factory(full_target))
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.warning(
            "glm5next_rotation: %s target %s not patched (%s: %s)",
            label, full_target, type(exc).__name__, exc,
        )
        return
    logger.info("glm5next_rotation: %s patched at %s", label, full_target)


for _mod_path, _attribute, _factory, _label in _TARGETS:
    _install(_mod_path, _attribute, _factory, _label)
