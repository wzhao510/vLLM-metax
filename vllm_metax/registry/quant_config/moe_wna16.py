# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    is_layer_skipped_quant,
    MoeWNA16Method as vllm_MoeWNA16Method,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config


# Remove configs of marlin
@register_quantization_config("moe_wna16")
class MacaMoeWNA16Config(MoeWNA16Config):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_marlin = False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm_metax.registry.quant_config.auto_awq import MacaAutoAWQConfig
            from vllm_metax.registry.quant_config.auto_gptq import MacaAutoGPTQConfig

            if self.linear_quant_method == "gptq":
                return MacaAutoGPTQConfig.from_config(
                    self.full_config
                ).get_quant_method(layer, prefix)
            elif self.linear_quant_method in ("awq", "awq_marlin"):
                return MacaAutoAWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, RoutedExperts):
            return MoeWNA16Method(self, layer.moe_config)
        return None


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class MoeWNA16Method(vllm_MoeWNA16Method):
    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
