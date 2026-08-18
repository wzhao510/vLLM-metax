# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX DeepSeek V4 compressor cache-store launcher.

The main compressed-attention cache is BF16. The Lightning Indexer cache is
INT8 with a block-segregated FP32 scale tail. FP8 and FP4 are rejected before
kernel selection.
"""

from typing import Any

import torch

from vllm.triton_utils import triton

from .bf16 import _fused_kv_compress_norm_rope_insert_sparse_attn_bf16
from .int8 import _fused_kv_compress_norm_rope_insert_indexer_attn_int8
from .bf16 import _fused_kv_compress_norm_rope_insert_sparse_attn_bf16
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
    _fused_kv_compress_norm_rope_insert_indexer_attn,
)


def compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    use_fp8_indexer: bool,
    use_fp8_kvcache: bool,
) -> None:
    """Shared triton launcher for the fused compress+norm+RoPE+insert path.

    Picks one of the three kernels in this module based on ``head_dim`` and
    ``use_fp4_cache``. Identical launch signature for all three.
    """
    if head_dim == 512:
        if use_fp8_kvcache:
            kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            scale_kargs = {"FP8_MAX":448.0}
        else:
            kernel = _fused_kv_compress_norm_rope_insert_sparse_attn_bf16
            scale_kargs = {"INT8_MAX":127.0}
        num_warps = 4
    else: # head_dim == 128
        if use_fp8_indexer:
            kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
            scale_kargs = {"FP8_MAX":448.0}
        else:
            kernel = _fused_kv_compress_norm_rope_insert_indexer_attn_int8
            scale_kargs = {"INT8_MAX":127.0}
        num_warps = 1

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **scale_kargs,
        **pdl_kwargs,
    )
