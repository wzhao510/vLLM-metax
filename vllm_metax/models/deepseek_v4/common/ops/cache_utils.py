# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from typing import Any

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_k_cache_kernel(
    out_ptr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    k_cache_ptr,
    k_cache_stride0: tl.constexpr,
    k_cache_stride1: tl.constexpr,
    seq_lens_ptr,
    block_table_ptr,
    offset: tl.constexpr,
    gather_lens_ptr,
    # constexpr
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_worker_id = tl.program_id(1)
    dim_block_id = tl.program_id(2)

    num_token_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len

    start_pos = seq_len - gather_len

    dim_offsets = dim_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_size

    for i in range(token_worker_id, gather_len, num_token_workers):
        pos = start_pos + i

        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        # A 0.27 packed-cache layer view can have a physical block stride much
        # larger than cache_block_size * head_size. Always address through the
        # concrete tensor strides; only the last dimension is contiguous.
        k_ptr = (
            k_cache_ptr
            + physical_block_idx.to(tl.int64) * k_cache_stride0
            + pos_in_block * k_cache_stride1
            + dim_offsets
        )

        out_ptr_cur = (
            out_ptr
            + batch_idx * out_stride0
            + (offset + i) * out_stride1
            + dim_offsets
        )

        vals = tl.load(k_ptr, mask=dim_mask, other=0.0)
        tl.store(out_ptr_cur, vals, mask=dim_mask)


def gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_size]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs] or None
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    if k_cache.ndim != 3 or k_cache.stride(2) != 1:
        raise ValueError(
            "DeepSeek V4 BF16 gather expects a 3D cache with a contiguous "
            f"head dimension; got shape={tuple(k_cache.shape)}, "
            f"stride={k_cache.stride()}."
        )
    if k_cache.dtype != torch.bfloat16:
        raise ValueError(
            f"DeepSeek V4 MetaX gather requires BF16 K cache; got {k_cache.dtype}."
        )
    if k_cache.shape[1] != block_size:
        raise ValueError(
            f"K-cache block dimension {k_cache.shape[1]} does not match "
            f"metadata block_size={block_size}."
        )
    if out.ndim != 3 or out.shape[2] != k_cache.shape[2] or out.stride(2) != 1:
        raise ValueError(
            "DeepSeek V4 gather output must be [num_reqs, tokens, head_size] "
            "with a contiguous head dimension."
        )

    num_reqs = seq_lens.shape[0]
    head_size = k_cache.shape[2]


    if gather_lens is not None:
        assert gather_lens.is_cuda
        assert gather_lens.shape == seq_lens.shape

    NUM_TOKEN_WORKERS = 128
    BLOCK_D = triton.next_power_of_2(head_size)


    _gather_k_cache_kernel[(num_reqs, NUM_TOKEN_WORKERS, 1)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        head_size=head_size,
        BLOCK_D=BLOCK_D,
    )
