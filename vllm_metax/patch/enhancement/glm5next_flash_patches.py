# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# GLM-5.3 Flash 成套适配包（MACA / vLLM 0.27.2rc1.dev22）
#
# 三个 patch 相互独立，但在 Flash 模型加载 + 前向计算时**成套**生效：
#
#   Patch 1  compressed_tensors MoE scheme 分组路由
#     → 让 RoutedExperts 绑定到 ckpt 声明的 FusedMoE 目标组（W4A8）
#       而不是被上游无条件别名到 Linear 组（W8A8）。
#       不做这步 → W8A8 quant_method 分配 2× 体积的 int8 权重 → 单卡 OOM，
#       且 ckpt 的 INT32 packed 张量根本装不进 int8 权重。
#
#   Patch 2  RoutedExperts params mapping 扁平命名修复
#     → 上游虚构了 ".routed_experts." 中间子模块层级：
#         param_name = "experts.routed_experts.w2_"
#       但 FusedMoEFactory 把 RoutedExperts 扁平注册在父模块 "experts" 下，
#       所以 W4A8 quant_method 注册的参数名是扁平的：
#         layer.register_parameter("w2_weight_packed", ...)
#       结果：name.replace(weight_name, param_name) 必然落到不存在的 key
#       → KeyError: '...experts.routed_experts.w2_weight_packed'
#       修复：strip 掉虚构段，让 param_name 变回 "experts.w2_"。
#
#   Patch 3  QuaRot 在线逆旋转
#     → 在 Glm5NextModel 和 Glm5NextMultiTokenPredictorLayer 的最终 RMSNorm
#       上挂 forward pre-hook，对输入做 Hadamard 逆旋转。
#       hook 方式优于替换 __call__：不动模块树、不动 state_dict key，
#       旋转模块挂载名严格对齐 config / checkpoint。
#
# 运行时依赖：Patch 1 必须先于 Patch 2 生效——只有 W4A8 method 注册了扁平
# 参数名，Patch 2 strip 后的 param_name 才能对上真实 key。Patch 3 与此无关，
# 在前向阶段才介入。
#
# 上游移除条件：
#   1. 上游 compressed-tensors config 优先选择 FusedMoE 目标组；
#   2. 上游把 RoutedExperts 的 param namespace 拍平到与 quant method 一致；
#   3. 上游 GLM 原生支持 online rotation。
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


# =========================================================================
# Patch 1 — compressed_tensors MoE scheme 分组路由
# =========================================================================
#
# Upstream `CompressedTensorsConfig._add_fused_moe_to_target_scheme_map`
# unconditionally aliases `RoutedExperts` to the `Linear` group:
#
#     self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["Linear"]
#
# GLM-5.3-Flash checkpoints describe the experts as a *separate* group with
# its own bit-widths, e.g.
#
#     group_0  targets=['Linear']    weights=8  input=8
#     group_1  targets=['FusedMoE']  weights=4  input=8
#
# Alias to `Linear` therefore resolves the experts to the 8-bit scheme
# (W8A8 int8), which is both wrong and expensive: the checkpoint stores 4-bit
# nibble-packed int32 tensors (`weight_packed`), so the 8-bit method allocates
# per-expert int8 weights at twice the size and the rank OOMs during load.
#
# Keep the upstream fallback intact but prefer `FusedMoE` whenever that
# target is present, so `RoutedExperts` inherits the 4-bit scheme.

@patch(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
    "CompressedTensorsConfig._add_fused_moe_to_target_scheme_map",
)
def _add_fused_moe_to_target_scheme_map(self) -> None:
    """Bind `RoutedExperts` to the best matching group's scheme.

    Order of preference: a dedicated `FusedMoE` group (its own bit-widths),
    then the upstream fallback to the `Linear` group.
    """
    if (
        "Linear" not in self.target_scheme_map
        or "RoutedExperts" in self.target_scheme_map
    ):
        return

    if "FusedMoE" in self.target_scheme_map:
        self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["FusedMoE"]
        logger.info(
            "glm5next_flash_patches: RoutedExperts → FusedMoE group (%s)",
            self.target_scheme_map["FusedMoE"],
        )
        return

    self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["Linear"]
    logger.info(
        "glm5next_flash_patches: RoutedExperts → Linear group (upstream fallback)"
    )


# =========================================================================
# Patch 2 — RoutedExperts params mapping 扁平命名修复
# =========================================================================
#
# Upstream `RoutedExperts.build_expert_params_mapping` builds tuple entries
# like
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


# ---- patch the three entry points that produce expert-params tuples ----


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
                "glm5next_flash_patches: model.named_parameters() "
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
            "glm5next_flash_patches: patch did not record original "
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


# =========================================================================
# Patch 3 — QuaRot 在线逆旋转
# =========================================================================
#
# Two insertion points:
#
#   1. main trunk — `Glm5NextModel`.  Its forward ends with the residual
#      stream already summed and the final norm takes a single argument:
#          hidden_states = self.norm(hidden_states)
#      → rotate the norm's input.
#
#   2. MTP draft — `Glm5NextMultiTokenPredictorLayer`.  The draft bypasses
#      `SharedHead.forward` and drives the fused norm directly:
#          hidden_states, _ = self.shared_head.norm(
#              hidden_states, residual=residual)
#      i.e. the residual add happens *inside* RMSNorm.  The rotation is
#      linear, so rotating `hidden_states` and `residual` separately is
#      equivalent to rotating their sum, while the fused kernel keeps its
#      own dtype handling and return values exactly as upstream.
#
# Mechanism: a forward *pre-hook* on the final RMSNorm.  Replacing the
# callable instead would trip `nn.Module.__setattr__`, which refuses to bind
# a non-Module onto a name already present in `self._modules`; the pre-hook
# leaves both the module tree and the state_dict keys untouched.
#
# The rotation module is mounted under the exact name the config and the
# checkpoint use, so `sign` / `hk` load as ordinary weights:
#     trunk : <prefix>.prenorm_hadamard_rotation
#     MTP   : <prefix>.shared_head.prenorm_hadamard_rotation


_ROTATION_LEAF = "prenorm_hadamard_rotation"
_TRUNK_SUFFIX = _ROTATION_LEAF
_MTP_SUFFIX = f"shared_head.{_ROTATION_LEAF}"
_HOOK_FLAG = "_glm5next_rotation_hooked"


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

    The runtime `prefix` and the exporter-facing key can disagree about the
    path in between, because a multimodal config anchors the language model
    one level deeper than vLLM's draft module does.  Concretely, for layer 45
    this patch may see prefix `model.layers.45` while the config carries
    `model.language_model.layers.45.shared_head.prenorm_hadamard_rotation`.

    So the exact lookup is backed by two progressively weaker anchors, and
    only a unique best match is accepted:
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
            logger.info("glm5next_flash_patches: mounting rotation (config key %r)", key)
            return build_rotation(spec)

    if available:
        logger.warning(
            "glm5next_flash_patches: no rotation key matches prefix %r (leaf %r); "
            "available keys %s. Rotation NOT installed here.",
            prefix, leaf_suffix, sorted(available),
        )
    else:
        logger.debug(
            "glm5next_flash_patches: config has no online_rotation; nothing to mount "
            "for prefix %r.",
            prefix,
        )
    return None


# ---- pre-hook ----


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


# ---- helpers to locate the final RMSNorm ----


def _final_norm(owner: Any) -> Any:
    """Return the final RMSNorm of `owner`, or `None` when it has none."""
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


# ---- __init__ wrappers ----


def _original(patched: Callable, full_target: str) -> Callable:
    """Read back the method `patch()` displaced when it installed `patched`."""
    originals = getattr(patched, PATCH_MARKER, None)
    if not isinstance(originals, dict) or full_target not in originals:
        raise RuntimeError(
            f"glm5next_flash_patches: patch() did not record {full_target!r}"
        )
    return originals[full_target]


def _positional_prefix(args: tuple, kwargs: dict) -> str:
    """Recover `prefix` from either calling convention."""
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
                "glm5next_flash_patches: trunk has no final RMSNorm on this stage; "
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
                "glm5next_flash_patches: MTP layer has no `shared_head`; "
                "rotation not installed."
            )
            return

        norm = _final_norm(shared_head)
        if norm is None:
            logger.debug(
                "glm5next_flash_patches: MTP shared_head has no RMSNorm on this stage; "
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


# ---- install (fires at import time) ----


_TARGETS = [
    ("vllm.models.glm5next.nvidia.model", "Glm5NextModel.__init__",
     _make_trunk_init, "trunk"),
    ("vllm.models.glm5next.nvidia.mtp", "Glm5NextMultiTokenPredictorLayer.__init__",
     _make_mtp_init, "mtp"),
]


def _install(mod_path: str, attribute: str, factory: Callable, label: str) -> None:
    """Install one rotation patch, reporting a missing target instead of aborting import."""
    full_target = f"{mod_path}.{attribute}"
    try:
        patch(mod_path, attribute)(factory(full_target))
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.warning(
            "glm5next_flash_patches: %s target %s not patched (%s: %s)",
            label, full_target, type(exc).__name__, exc,
        )
        return
    logger.info("glm5next_flash_patches: %s rotation patched at %s", label, full_target)


for _mod_path, _attribute, _factory, _label in _TARGETS:
    _install(_mod_path, _attribute, _factory, _label)


logger.info("glm5next_flash_patches: all 3 patches installed")
