# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# GLM-5.3 Flash 适配包（MACA / vLLM 0.27.2rc1.dev22）
#
# 本文件只保留两块真正必要的补丁：
#
#   Patch 1  compact-tensors MoE scheme 分组路由
#     ckpt 里 MoE 专家是独立的一组量化参数：
#         group_0  targets=['Linear']    weights=8  input=8
#         group_1  targets=['FusedMoE']  weights=4  input=8
#     但上游 CompressedTensorsConfig 会把 RoutedExperts 无条件别名到
#     Linear 组（W8A8）。结果 MoE 走 W8A8：
#       - 分配 2 倍体积的 int8 权重 -> 单卡 OOM；
#       - 注册的参数叫 w13_weight / w2_weight，
#         而 ckpt 里是 *_weight_packed -> KeyError。
#     修法：在配置对象构造完成时就把 RoutedExperts 指向 FusedMoE 组，
#     这样后面无论哪个版本的 get_moe_method 都不会再把它改回 Linear。
#
#   Patch 2  QuaRot 在线逆旋转
#     在 trunk / MTP 的最终 RMSNorm 上挂 forward pre-hook，
#     对输入做 Hadamard 逆旋转。
#
# 另带一条**只读**诊断日志：RoutedExperts 构造完成后打印实际选中的
# quant_method 类名和是否存在 *_weight_packed 参数。它不改任何行为，
# 只用于确认 Patch 1 是否生效。
#
# 说明：不再 patch 专家参数名映射。上游 build_expert_params_mapping 生成的
#     experts.routed_experts.w2_ 前缀与 MoERunner 的真实子模块层级一致
#     （MoERunner.__init__ 里 self.routed_experts = routed_experts），
#     剥掉这一段反而会制造不存在的 key。
#
# 上游移除条件：
#   1. 上游 compressed-tensors 优先选择 FusedMoE 目标组；
#   2. 上游 GLM 原生支持 online rotation。
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


def _bits(scheme: Any) -> str:
    if not isinstance(scheme, Mapping):
        return "?"
    weight_quant = scheme.get("weights")
    act_quant = scheme.get("input_activations")
    return (
        f"W{getattr(weight_quant, 'num_bits', '?')}"
        f"A{getattr(act_quant, 'num_bits', '?')}"
    )


def _force_routed_experts_scheme(quant_config: Any) -> None:
    """把 RoutedExperts 绑定到 FusedMoE 组，必要时回退到 Linear 组。

    只改这一层的 target -> scheme 映射，不改量化参数本身。
    """
    scheme_map = getattr(quant_config, "target_scheme_map", None)
    if not isinstance(scheme_map, dict):
        logger.warning(
            "glm5next_flash_patches: CompressedTensorsConfig has no "
            "target_scheme_map; MoE scheme routing NOT applied."
        )
        return

    if "FusedMoE" in scheme_map:
        scheme_map["RoutedExperts"] = scheme_map["FusedMoE"]
        logger.info(
            "glm5next_flash_patches: RoutedExperts -> FusedMoE group (%s); "
            "map=%s",
            _bits(scheme_map["FusedMoE"]),
            {key: _bits(value) for key, value in scheme_map.items()},
        )
        return

    # 没有 FusedMoE 组时保持上游语义：仅当还没有显式 RoutedExperts 时才别名到 Linear。
    if "RoutedExperts" in scheme_map:
        source = "existing RoutedExperts entry"
    elif "Linear" in scheme_map:
        scheme_map["RoutedExperts"] = scheme_map["Linear"]
        source = "Linear group"
    else:
        source = "none"
    logger.warning(
        "glm5next_flash_patches: no FusedMoE group in target_scheme_map; "
        "MoE scheme source=%s; map=%s",
        source,
        {key: _bits(value) for key, value in scheme_map.items()},
    )


def _compressed_tensors_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """CompressedTensorsConfig.__init__ 包装：构造完立刻修正 target_scheme_map。"""
    _original(_compressed_tensors_init, _CT_INIT_TARGET)(self, *args, **kwargs)
    _force_routed_experts_scheme(self)


_CT_INIT_TARGET = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors.CompressedTensorsConfig.__init__"
)


def _install_scheme_routing() -> None:
    try:
        patch(
            "vllm.model_executor.layers.quantization."
            "compressed_tensors.compressed_tensors",
            "CompressedTensorsConfig.__init__",
        )(_compressed_tensors_init)
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.warning(
            "glm5next_flash_patches: scheme routing target %s not patched "
            "(%s: %s)",
            _CT_INIT_TARGET,
            type(exc).__name__,
            exc,
        )
        return
    logger.info("glm5next_flash_patches: scheme routing patched at %s", _CT_INIT_TARGET)


_install_scheme_routing()


# =========================================================================
# 诊断 —— RoutedExperts 实际使用的 quant_method
# =========================================================================

_REPORTED_METHODS: set[str] = set()


def _routed_experts_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """RoutedExperts.__init__ 包装：只打印选中的量化方法，不改行为。"""
    _original(_routed_experts_init, _ROUTED_EXPERTS_INIT_TARGET)(
        self, *args, **kwargs
    )

    quant_method = getattr(self, "quant_method", None)
    method_name = type(quant_method).__name__ if quant_method is not None else "None"
    if method_name in _REPORTED_METHODS:
        return
    _REPORTED_METHODS.add(method_name)

    names = [name for name, _ in self.named_parameters()]
    packed = [name for name in names if name.endswith("_weight_packed")]
    logger.info(
        "glm5next_flash_patches: MoE quant_method=%s (layer=%s), params=%d, "
        "packed=%d%s",
        method_name,
        getattr(self, "layer_name", "?"),
        len(names),
        len(packed),
        f" (e.g. {packed[:2]})" if packed else "",
    )
    if not packed:
        logger.error(
            "glm5next_flash_patches: no *_weight_packed parameter registered; "
            "the checkpoint (INT4 packed) will NOT load. All MoE params: %s",
            names[:12],
        )


_ROUTED_EXPERTS_INIT_TARGET = (
    "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts.__init__"
)


def _install_moe_diagnostic() -> None:
    """安装诊断补丁；目标模块缺失时告警而不是中断插件加载。"""
    try:
        patch(
            "vllm.model_executor.layers.fused_moe.routed_experts",
            "RoutedExperts.__init__",
        )(_routed_experts_init)
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.warning(
            "glm5next_flash_patches: MoE diagnostic target %s not patched "
            "(%s: %s)",
            _ROUTED_EXPERTS_INIT_TARGET,
            type(exc).__name__,
            exc,
        )
        return
    logger.info(
        "glm5next_flash_patches: MoE diagnostic patched at %s",
        _ROUTED_EXPERTS_INIT_TARGET,
    )


_install_moe_diagnostic()


# =========================================================================
# Patch 2 — QuaRot 在线逆旋转
# =========================================================================
#
# 两个插入点：
#
#   1. 主干 —— `Glm5NextModel`。forward 末尾 residual 已经加完，最终 norm
#      只接收一个参数：
#          hidden_states = self.norm(hidden_states)
#      → 旋转 norm 的输入。
#
#   2. MTP draft —— `Glm5NextMultiTokenPredictorLayer`。draft 不走
#      `SharedHead.forward`，而是直接驱动融合 norm：
#          hidden_states, _ = self.shared_head.norm(
#              hidden_states, residual=residual)
#      即 residual 加法发生在 RMSNorm 内部。旋转是线性的，分别旋转
#      `hidden_states` 和 `residual` 等价于旋转它们的和，同时保持融合
#      kernel 的 dtype 处理和返回值与上游完全一致。
#
# 机制：在最终 RMSNorm 上挂 forward *pre-hook*。直接替换 callable 会触发
#     `nn.Module.__setattr__` 的拒绝（同名已在 `self._modules` 中），
#     而 pre-hook 不动模块树、不动 state_dict key。
#
# 旋转模块按 config / checkpoint 里的真实名字挂载：
#     trunk : <prefix>.prenorm_hadamard_rotation
#     MTP   : <prefix>.shared_head.prenorm_hadamard_rotation


_ROTATION_LEAF = "prenorm_hadamard_rotation"
_TRUNK_SUFFIX = _ROTATION_LEAF
_MTP_SUFFIX = f"shared_head.{_ROTATION_LEAF}"
_HOOK_FLAG = "_glm5next_rotation_hooked"


def _rotation_modules(config: Any) -> Mapping[str, Any]:
    """返回 `config.online_rotation.modules`，没有则返回空映射。"""
    rotation_config = getattr(config, "online_rotation", None)
    if not isinstance(rotation_config, Mapping):
        return {}
    modules = rotation_config.get("modules")
    return modules if isinstance(modules, Mapping) else {}


def _resolve_key(
    modules: Mapping[str, Any], prefix: str, leaf_suffix: str
) -> tuple[str | None, Any]:
    """返回挂在 `prefix` 下的旋转配置 `(config_key, spec)`。

    运行时 `prefix` 与导出配置的 key 可能同时存在**顺序**和**层级深度**的
    差异。GLM-5.3-Flash 上观测到：trunk 实例化时的 prefix 是
    `language_model.model`，而 config 里写的是
    `model.language_model.prenorm_hadamard_rotation` —— 两个锚点顺序相反。
    MTP 的 key 还会多出 `shared_head` 这一层。

    因此精确查找后面依次挂更弱的锚点，且只接受唯一最优匹配：
       rank 1 —— key 的尾部路径段等于运行时 prefix；
       rank 2 —— key 的锚点段与运行时相同，只是顺序不同；
       rank 3 —— 运行时锚点是 key 路径段的子集（key 多出 `shared_head` 等层）；
       rank 4 —— key 与运行时以相同的末段（层号）结尾。
    """
    exact = f"{prefix}.{leaf_suffix}" if prefix else leaf_suffix
    if exact in modules:
        return exact, modules[exact]

    head = prefix.split(".") if prefix else []
    head_set = set(head)
    candidates: list[tuple[int, str]] = []
    for key in modules:
        if not key.endswith("." + leaf_suffix):
            continue
        segments = key[: -(len(leaf_suffix) + 1)].split(".")
        if not head or not segments:
            continue
        if segments[-len(head):] == head:
            candidates.append((1, key))
        elif len(segments) == len(head) and set(segments) == head_set:
            candidates.append((2, key))
        elif head_set.issubset(set(segments)):
            candidates.append((3, key))
        elif segments[-1] == head[-1]:
            candidates.append((4, key))

    best_rank = min((rank for rank, _ in candidates), default=None)
    winners = [key for rank, key in candidates if rank == best_rank]
    if len(winners) == 1:
        return winners[0], modules[winners[0]]
    return None, None


def _mount_rotation(
    configs: list[Any], prefix: str, leaf_suffix: str
) -> nn.Module | None:
    """为单个插入点构造旋转模块；未配置则返回 `None`。"""
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
    """旋转最终 RMSNorm 消费的每个张量。

    上游给这些 norm 的输入要么是 `hidden_states`（主干，residual 已加完），
    要么是 `hidden_states` + `residual`（MTP，kernel 内部相加）。
    两者都是同一次归一化的线性输入，所以分别旋转与旋转它们的和等价，
    同时完全保持上游的 dtype 提升与返回值。
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
    """返回 `owner` 的最终 RMSNorm，没有则返回 `None`。"""
    norm = getattr(owner, "norm", None)
    return norm if isinstance(norm, RMSNorm) else None


def _candidate_configs(vllm_config: Any) -> list[Any]:
    """收集可能携带 `online_rotation` 的 HF config。

    MTP draft 从自己的 draft config 读旋转；主干从 target model 的 config 读。
    这里同时探测 `hf_config` 与 `hf_text_config`，因为多模态 config 可能
    把旋转只放在 `text_config` 下。
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
    """读回 `patch()` 安装 `patched` 时替换掉的原始实现。"""
    originals = getattr(patched, PATCH_MARKER, None)
    if not isinstance(originals, dict) or full_target not in originals:
        raise RuntimeError(
            f"glm5next_flash_patches: patch() did not record {full_target!r}"
        )
    return originals[full_target]


def _positional_prefix(args: tuple, kwargs: dict) -> str:
    """兼容位置参数 / 关键字参数两种调用方式，取出 `prefix`。"""
    if "prefix" in kwargs:
        return kwargs["prefix"]
    return args[1] if len(args) > 1 else ""


def _positional_vllm_config(args: tuple, kwargs: dict) -> Any:
    return kwargs["vllm_config"] if "vllm_config" in kwargs else args[0]


def _make_trunk_init(full_target: str) -> Callable:
    """包装 `Glm5NextModel.__init__`，给主干最终 norm 挂 hook。"""

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
    """包装 `Glm5NextMultiTokenPredictorLayer.__init__`，给 MTP norm 挂 hook。

    旋转挂在 `shared_head` 上，与 config / checkpoint 的 key
    `...<prefix>.shared_head.prenorm_hadamard_rotation` 对齐。
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
    """安装单个旋转补丁；目标缺失时打印告警而不是中断 import。"""
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


logger.info("glm5next_flash_patches: scheme routing + rotation patches installed")
