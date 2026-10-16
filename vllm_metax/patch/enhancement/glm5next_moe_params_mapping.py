# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# MoE expert-params mapping fixes.
#
# Upstream `RoutedExperts.build_expert_params_mapping` builds tuple entries like
#
#     param_name = "experts.routed_experts.w2_"       # spurious submodule
#     weight_name = "experts.8.down_proj."            # matches ckpt path
#
# The `.routed_experts.` segment in `param_name` is an _assumed_ extra
# submodule level that does not exist in the actual nn.Module tree.
# RoutedExperts is always registered under the parent name `"experts"` (see
# Llama4MoE.self.experts, FusedMoEFactory), so every compressed-tensors MoE
# method registers flat parameters directly on that module — e.g.
# `layer.register_parameter("w2_weight_packed", ...)`.  Inside the top-level
# `model.named_parameters()` dictionary those become
# `<layer>.experts.w2_weight_packed`, never
# `<layer>.experts.routed_experts.w2_weight_packed`.
#
# Consequently the `name.replace(weight_name, param_name)` call that every
# model's load_weights performs resolves to a non-existent parameter key.
# Stripping the bogus `.routed_experts.` collapses param_name to
# `"experts.w2_"` — then the underscore tail combines with whatever suffix
# the checkpoint carries (`.weight`, `.weight_packed`, `.weight_scale`, ...)
# into the exact flat key the quant method registered.
#
#    ckpt: "...experts.8.down_proj.weight_packed"
#      .replace("experts.8.down_proj.", "experts.w2_")
#    → "...experts.w2_weight_packed"   ✓ matches params_dict
#
# It also makes `make_expert_params_mapping` tolerate `model=None`.  The
# legacy entry point's `model.named_parameters()` scan was only ever used to
# detect a `.base_layer.` LoRA prefix — callers without a real model can pass
# None safely; the LoRA prefix is rare in compressed-tensors checkpoints.
#
# Affected versions: vLLM 0.27.2rc1.dev22 on MetaX MACA.
# Remove at: Upstream flattens the MoE param namespace to match what every
#     quant method already registers.
# -----------------------------------------------------------------------------

from __future__ import annotations

from vllm.logger import init_logger

from vllm_metax.patch.utils import PATCH_MARKER, patch

logger = init_logger(__name__)


def _strip_routed_experts_segment(param_name: str) -> str:
    """Remove only the ``.routed_experts`` submodule from upstream param_name.

    RoutedExperts is always registered under the parent name ``experts``, so
    the ``.routed_experts`` segment is spurious — but a possible LoRA
    ``.base_layer.`` segment that follows it is legitimate and must be kept.

        "experts.routed_experts.w13_"                 → "experts.w13_"
        "experts.routed_experts.w2_"                  → "experts.w2_"
        "experts.routed_experts.base_layer.w2_"       → "experts.base_layer.w2_"
        "experts.w2_"                                 → "experts.w2_"   (no-op)
        "other.w2_"                                   → "other.w2_"     (no-op)
    """
    marker = "experts.routed_experts."
    if param_name.startswith(marker):
        return "experts." + param_name[len(marker):]
    return param_name


def _rewrite_tuple(
    param_name: str, weight_name: str, expert_id: int, shard_id: str
) -> tuple[str, str, int, str]:
    """Strip the spurious routed_experts nesting, leave everything else intact."""
    return (
        _strip_routed_experts_segment(param_name),
        weight_name,
        expert_id,
        shard_id,
    )


# ---------------------------------------------------------------------------
# patch 1 / 2 / 3: rewrite tuples at the three entry points that produce them
# ---------------------------------------------------------------------------

def _patched_make_expert_params_mapping(
    model,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int = 0,
    routed_experts_prefix: str = "routed_experts",
):
    """Return flat-nesting tuples and tolerate ``model=None``."""
    lora_prefix = ""
    if model is not None:
        try:
            has_base_layer = any(
                ".base_layer." in n for n, _ in model.named_parameters()
            )
            lora_prefix = "base_layer." if has_base_layer else ""
        except Exception as exc:
            logger.warning(
                "glm5next_moe_params_mapping: model.named_parameters() "
                "failed (%s); defaulting lora_prefix to ''",
                exc,
            )

    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )

    raw = RoutedExperts.build_expert_params_mapping(
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts,
        routed_experts_prefix,
        lora_prefix,
    )
    return [_rewrite_tuple(*t) for t in raw]


def _patched_get_expert_mapping(
    self,
    ckpt_gate_proj_name=None,
    ckpt_down_proj_name=None,
    ckpt_up_proj_name=None,
    include_fused: bool = False,
):
    """Instance wrapper — same flat rewrite applied to RoutedExperts.load_weights."""
    originals = getattr(_patched_get_expert_mapping, PATCH_MARKER, {})
    original = originals.get(
        "vllm.model_executor.layers.fused_moe.routed_experts."
        "RoutedExperts.get_expert_mapping"
    )
    if original is None:
        raise RuntimeError(
            "glm5next_moe_params_mapping: patch did not record original "
            "get_expert_mapping"
        )
    raw = original(
        self,
        ckpt_gate_proj_name=ckpt_gate_proj_name,
        ckpt_down_proj_name=ckpt_down_proj_name,
        ckpt_up_proj_name=ckpt_up_proj_name,
        include_fused=include_fused,
    )
    return [_rewrite_tuple(*t) for t in raw]


def _patched_fused_moe_make_expert_params_mapping(
    model,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int = 0,
    routed_experts_prefix: str = "routed_experts",
):
    """Thin module-level wrapper — delegates to the patched staticmethod."""
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )

    return RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts,
        routed_experts_prefix,
    )


patch(
    "vllm.model_executor.layers.fused_moe.routed_experts",
    "RoutedExperts.make_expert_params_mapping",
)(_patched_make_expert_params_mapping)

patch(
    "vllm.model_executor.layers.fused_moe.routed_experts",
    "RoutedExperts.get_expert_mapping",
)(_patched_get_expert_mapping)

patch(
    "vllm.model_executor.layers.fused_moe.layer",
    "fused_moe_make_expert_params_mapping",
)(_patched_fused_moe_make_expert_params_mapping)

logger.info("glm5next_moe_params_mapping: installed")
