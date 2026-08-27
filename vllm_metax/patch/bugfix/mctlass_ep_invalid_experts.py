# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
# -----------------------------------------------------------------------------
# Note: MCTlass fused MoE does not skip the -1 expert IDs generated for experts
#       owned by other expert-parallel ranks and can access invalid weights.
#
# Affected versions: v0.26.0
#
# Remove at: after TritonExperts requests ignore_invalid_experts for MCTlass EP,
#            or after MCTlass safely handles invalid expert IDs.
# -----------------------------------------------------------------------------

from collections.abc import Callable
from typing import Any

import torch
import vllm_metax.envs as mx_envs
from vllm_metax.patch.utils import patch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


@patch("vllm_metax.model_executor.layers.fused_moe.fused_moe")
def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict[str, Any],
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    *,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int8_w8a8: bool = False,
    use_int4_w4a8: bool = False,
    block_shape: list[int] | None = None,
    ignore_invalid_experts: bool = False,
    block_size_m_override: Callable | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Prepare expert assignments for the aligned and low-latency Triton paths."""
    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
    # activates only a small fraction of total experts
    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
    # path of the fused_moe_kernel kernel
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * 4 <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
        # -----------------------------------------------------------------
        # Metax Modification:
        # mctlass moe not support `sorted_token_ids is None`
        # for all W4A8 WAA16 W8A8 and BF16
        and not mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    if block_size_m_override is not None:
        block_size_m_override()

    # /-------------------- MetaX Modification --------------------\
    ignore_invalid_experts = ignore_invalid_experts or (
        expert_map is not None and mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE
    )
    # \-------------------- MetaX Modification --------------------/

    return moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )
