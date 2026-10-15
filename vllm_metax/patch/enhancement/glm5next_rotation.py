# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Inject QuaRot-style online inverse Hadamard rotation into GLM-5.3-Flash.
#     Two insertion points mirror the internal spec and the existing
#     deepseek_v2 / deepseek_mtp integration:
#
#       1. Glm5NextModel (main trunk): after the decoder-block loop, before
#          the final RMSNorm.  GLM's upstream forward ends with
#          `self.norm(hidden_states)` — single-argument, no residual — so we
#          wrap that one norm call with the prenorm rotation.
#
#       2. Glm5NextMultiTokenPredictorLayer (MTP draft): the layer bypasses
#          SharedHead.forward entirely and invokes `self.shared_head.norm(
#          hidden_states, residual=residual)` directly.  We patch that norm
#          call to apply the rotation first.
#
# Affected versions: vLLM 0.27.2rc1.dev22, reviewed 2026-10-15.
#
# Config keys (mounted under config.online_rotation.modules):
#     "model.language_model.prenorm_hadamard_rotation"
#     "model.language_model.layers.45.shared_head.prenorm_hadamard_rotation"
#
# Remove at: Upstream GLM supports online rotation natively.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _maybe_prefix(prefix: str, leaf: str) -> str:
    from vllm.model_executor.models.utils import maybe_prefix
    return maybe_prefix(prefix, leaf)


def _build_rotation_if_present(config, prefix: str) -> nn.Module | None:
    from vllm_metax.model_executor.layers.online_rotation import (
        build_rotation,
        get_online_rotation_module_config,
    )
    name = _maybe_prefix(prefix, "prenorm_hadamard_rotation")
    rot_cfg = get_online_rotation_module_config(config, name)
    if rot_cfg is None:
        return None
    logger.info("GLM5Next rotation enabled at %s", name)
    return build_rotation(rot_cfg)


def _get_original(patched_func: Any, full_target_name: str) -> Any:
    """Retrieve the original method that patch() captured before replacing."""
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


# ===========================================================================
# Patch 1 — Glm5NextModel  (main trunk final-norm prenorm rotation)
# ===========================================================================
# Two possible class locations depending on the vLLM layout.  We try both;
# whichever path successfully imports wins.  The other is silently skipped.

_GLM5NEXT_MODEL_CANDIDATES = [
    ("vllm.models.glm5next.nvidia.model", "Glm5NextModel"),
    ("vllm.models.glm5next.common.mtp", "Glm5NextModel"),
]


def _try_patch_glm5next_model():
    """Attempt to install Glm5NextModel patches at every known location."""
    from vllm_metax.patch.utils import patch

    for mod_path, cls_name in _GLM5NEXT_MODEL_CANDIDATES:
        try:
            decorator_init = patch(mod_path, f"{cls_name}.__init__")
            decorator_fwd = patch(mod_path, f"{cls_name}.forward")
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug(
                "glm5next_rotation: skipping %s.__init__/.forward (%s: %s)",
                mod_path, type(exc).__name__, exc,
            )
            continue

        full_init = f"{mod_path}.{cls_name}.__init__"
        full_fwd = f"{mod_path}.{cls_name}.forward"

        # Guard: if both nvidia.* and common.* resolve to the SAME class,
        # patching __init__ first makes forward's original already carry a
        # marker — that triggers RuntimeError above and the path is skipped.
        _already_patched_check = None  # placeholder

        @decorator_init
        def _patched_init(self, vllm_config, prefix: str = "", *args, **kwargs):
            # Call the genuine upstream __init__.
            original_init = _get_original(_patched_init, full_init)
            original_init(self, vllm_config, prefix, *args, **kwargs)
            # Graft rotation attributes.
            config = vllm_config.model_config.hf_config
            self.prenorm_hadamard_rotation = _build_rotation_if_present(
                config, prefix
            )
            self.use_hadamard_rotation = self.prenorm_hadamard_rotation is not None

        @decorator_fwd
        def _patched_forward(self, *args, **kwargs):
            original_forward = _get_original(_patched_forward, full_fwd)
            if not getattr(self, "use_hadamard_rotation", False):
                return original_forward(self, *args, **kwargs)
            # Wrap self.norm so its single-argument call applies rotation first.
            original_norm = self.norm
            rotation = self.prenorm_hadamard_rotation

            def _norm_with_rot(x, *na, **nk):
                return original_norm(rotation(x), *na, **nk)

            try:
                self.norm = _norm_with_rot
                return original_forward(self, *args, **kwargs)
            finally:
                self.norm = original_norm

        logger.info(
            "glm5next_rotation: patched %s.__init__ and .forward", cls_name
        )


# ===========================================================================
# Patch 2 — Glm5NextMultiTokenPredictorLayer  (MTP shared_head prenorm rotation)
# ===========================================================================
# The draft layer may live in nvidia/mtp, nvidia/model, or common/mtp.

_MTP_LAYER_CANDIDATES = [
    ("vllm.models.glm5next.nvidia.mtp", "Glm5NextMultiTokenPredictorLayer"),
    ("vllm.models.glm5next.nvidia.model", "Glm5NextMultiTokenPredictorLayer"),
    ("vllm.models.glm5next.common.mtp", "Glm5NextMultiTokenPredictorLayer"),
]


def _try_patch_mtp_layer():
    from vllm_metax.patch.utils import patch

    for mod_path, cls_name in _MTP_LAYER_CANDIDATES:
        try:
            decorator_init = patch(mod_path, f"{cls_name}.__init__")
            decorator_fwd = patch(mod_path, f"{cls_name}.forward")
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug(
                "glm5next_rotation: skipping %s.__init__/.forward (%s: %s)",
                mod_path, type(exc).__name__, exc,
            )
            continue

        full_init = f"{mod_path}.{cls_name}.__init__"
        full_fwd = f"{mod_path}.{cls_name}.forward"

        @decorator_init
        def _patched_init(self, vllm_config, prefix, *args, **kwargs):
            original_init = _get_original(_patched_init, full_init)
            original_init(self, vllm_config, prefix, *args, **kwargs)
            # GLM's MTP shared_head lives at f"{prefix}.shared_head".
            draft_config = (
                vllm_config.speculative_config.draft_model_config.hf_text_config
            )
            self._mtp_prenorm_rotation = _build_rotation_if_present(
                draft_config, f"{prefix}.shared_head"
            )

        @decorator_fwd
        def _patched_forward(self, *args, **kwargs):
            original_forward = _get_original(_patched_forward, full_fwd)
            rotation = getattr(self, "_mtp_prenorm_rotation", None)
            if rotation is None:
                return original_forward(self, *args, **kwargs)
            # GLM bypasses SharedHead.forward and calls shared_head.norm directly.
            original_sh_norm = self.shared_head.norm

            def _sh_norm_with_rot(x, *na, **nk):
                return original_sh_norm(rotation(x), *na, **nk)

            try:
                self.shared_head.norm = _sh_norm_with_rot
                return original_forward(self, *args, **kwargs)
            finally:
                self.shared_head.norm = original_sh_norm

        logger.info(
            "glm5next_rotation: patched %s.__init__ and .forward", cls_name
        )


# ---------------------------------------------------------------------------
# Module-load hook — fire-and-forget.
# ---------------------------------------------------------------------------
# We deliberately let exceptions propagate up so a missing vLLM install is
# obvious; individual location-skips above are silent DEBUG-level only.

_try_patch_glm5next_model()
_try_patch_mtp_layer()
