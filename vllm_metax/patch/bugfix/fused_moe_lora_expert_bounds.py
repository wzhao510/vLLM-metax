
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# MetaX Modification:
# Bounds check for expert_id in fused MoE-LoRA small-batch kernel.
#
# Affected versions: vLLM 0.30.0
# Remove at: Upstream vLLM implements equivalent expert bounds checking.
# -----------------------------------------------------------------------------

import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _get_ptr,
    _pick_small_batch_chunk,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm_metax.patch.utils import patch


@patch(
    "vllm.lora.ops.triton_ops.fused_moe_lora_op",
    "_fused_moe_lora_small_batch_kernel",
)
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _fused_moe_lora_small_batch_kernel(
    # ---- pointers ----
    x_ptr,
    A_ptrs,
    B_ptrs,
    out_ptr,
    topk_weights_ptr,
    expert_ids_ptr,
    token_lora_mapping_ptr,
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    top_k_num,
    max_loras,
    work_total,
    pair_slices,
    # ---- strides ----
    stride_xm,
    stride_xk,
    stride_A_lora,
    stride_A_expert,
    stride_A_r,
    stride_A_k,
    stride_B_lora,
    stride_B_expert,
    stride_B_n,
    stride_B_r,
    stride_om,
    stride_on,
    # ---- runtime scalars ----
    slice_n_offset,
    n_tiles_per_program,
    n_chunks_per_pair_slice,
    # ---- constexpr ----
    token_mapping_factor: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    actual_rank: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    # /-------------------- MetaX Modification --------------------\
    NUM_EXPERTS: tl.constexpr,
    # \-----------------------------------------------------------/
    EVEN_K: tl.constexpr,
):
    """Persistent fused MoE-LoRA small-batch kernel."""

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    offs_r = tl.arange(0, BLOCK_R)
    rank_mask = offs_r < actual_rank

    safe_offs_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)

    for work_id in range(pid, work_total, num_programs):
        n_chunk_idx = work_id % n_chunks_per_pair_slice
        pair_slice_idx = work_id // n_chunks_per_pair_slice

        pair_idx = pair_slice_idx // NUM_SLICES
        slice_id = pair_slice_idx % NUM_SLICES

        token_idx = pair_idx // top_k_num
        lora_id = tl.load(token_lora_mapping_ptr + token_idx)

        valid = (lora_id >= 0) & (lora_id < max_loras)

        enabled = tl.load(
            adapter_enabled_ptr + tl.where(valid, lora_id, 0)
        )
        valid = valid & (enabled != 0)

        expert_id = tl.load(expert_ids_ptr + pair_idx)

        # /-------------------- MetaX Modification --------------------\
        # Prevent out-of-bounds A/B weight access for invalid expert IDs.
        valid = valid & (expert_id >= 0) & (expert_id < NUM_EXPERTS)
        # \-----------------------------------------------------------/

        if valid:
            cur_A_ptr = tl.load(A_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            cur_B_ptr = tl.load(B_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )

            A_base = (
                cur_A_ptr
                + lora_id * stride_A_lora
                + expert_id * stride_A_expert
            )
            B_base = (
                cur_B_ptr
                + lora_id * stride_B_lora
                + expert_id * stride_B_expert
            )

            x_row = pair_idx // token_mapping_factor
            x_row_ptr = x_ptr + x_row * stride_xm

            # ---- SHRINK GEMV ----
            rank_vec = tl.zeros((BLOCK_R,), dtype=tl.float32)

            if EVEN_K:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k

                    x_tile = tl.load(
                        x_row_ptr + cur_k * stride_xk
                    ).to(tl.float32)

                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)

                    rank_vec += tl.sum(
                        a_tile * x_tile[None, :], axis=1
                    )
            else:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k
                    k_mask = cur_k < K

                    x_tile = tl.load(
                        x_row_ptr + cur_k * stride_xk,
                        mask=k_mask,
                        other=0.0,
                    ).to(tl.float32)

                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    rank_vec += tl.sum(
                        a_tile * x_tile[None, :], axis=1
                    )

            # ---- EXPAND ----
            n_tile_start = n_chunk_idx * n_tiles_per_program

            out_row_ptr = (
                out_ptr
                + slice_id * slice_n_offset
                + pair_idx * stride_om
            )

            if MUL_ROUTED_WEIGHT:
                moe_w = tl.load(
                    topk_weights_ptr + pair_idx
                ).to(tl.float32)

            for nt in range(n_tiles_per_program):
                n_lo = (n_tile_start + nt) * BLOCK_N

                if n_lo < N:
                    offs_n = n_lo + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < N

                    b_tile = tl.load(
                        B_base
                        + offs_n[:, None] * stride_B_n
                        + safe_offs_r[None, :] * stride_B_r,
                        mask=n_mask[:, None] & rank_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    out_tile = tl.sum(
                        b_tile * rank_vec[None, :], axis=1
                    )

                    if MUL_ROUTED_WEIGHT:
                        out_tile = out_tile * moe_w

                    out_ptrs = out_row_ptr + offs_n * stride_on

                    if ADD_INPUTS:
                        prev = tl.load(
                            out_ptrs,
                            mask=n_mask,
                            other=0.0,
                        ).to(tl.float32)

                        tl.store(
                            out_ptrs,
                            (prev + out_tile).to(
                                out_ptr.dtype.element_ty
                            ),
                            mask=n_mask,
                        )
                    else:
                        tl.store(
                            out_ptrs,
                            out_tile.to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )


@patch(
    "vllm.lora.ops.triton_ops.fused_moe_lora_op",
    "_run_fused_moe_lora_small_batch",
)
def _run_fused_moe_lora_small_batch(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids_flat: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool,
    add_inputs: bool = True,
) -> None:
    """Small-batch GEMV-style wrapper with expert bounds checking."""

    num_slices = len(lora_a_stacked)
    device = qcurr_hidden_states.device

    A0 = lora_a_stacked[0]
    B0 = lora_b_stacked[0]

    max_loras_w = A0.shape[0]

    # /-------------------- MetaX Modification --------------------\
    num_experts = A0.shape[1]
    # \-----------------------------------------------------------/

    rank = A0.shape[2]
    K = A0.shape[3]
    N_per_slice = B0.shape[2]

    assert rank <= 64, (
        f"fused_moe_lora_small_batch supports rank<=64; got rank={rank}"
    )

    BLOCK_R = max(triton.next_power_of_2(rank), 16)

    num_tokens = topk_weights.shape[0]
    M_grid = num_tokens * top_k_num

    if M_grid == 0 or num_slices == 0:
        return

    token_mapping_factor = (
        1 if mul_routed_weight else top_k_num
    )

    A_ptrs = _get_ptr(lora_a_stacked, device)
    B_ptrs = _get_ptr(lora_b_stacked, device)

    assert output.dim() == 3, (
        f"output must be 3-D, got {output.shape}"
    )

    assert (
        output.stride(0)
        == output.shape[1] * output.stride(1)
    ), (
        "fused_moe_lora_small_batch requires "
        "output.stride(0) == top_k*stride(1); "
        f"got shape={output.shape} strides={output.stride()}"
    )

    out_view = output.view(-1, output.shape[-1])

    BLOCK_N = 128
    BLOCK_K = 128
    nw = 4
    ns = 3

    N_tiles = triton.cdiv(N_per_slice, BLOCK_N)
    pair_slices = M_grid * num_slices

    sm_count = current_platform.num_compute_units(
        device.index
    )

    n_tiles_per_program = _pick_small_batch_chunk(
        pair_slices,
        N_tiles,
        sm_count,
    )

    n_chunks = triton.cdiv(
        N_tiles,
        n_tiles_per_program,
    )

    work_total = pair_slices * n_chunks

    grid_size = min(
        work_total,
        max(1, 2 * sm_count),
    )
    grid = (grid_size,)

    _fused_moe_lora_small_batch_kernel[grid](
        qcurr_hidden_states,
        A_ptrs,
        B_ptrs,
        out_view,
        topk_weights,
        expert_ids_flat,
        token_lora_mapping,
        adapter_enabled,
        N_per_slice,
        K,
        top_k_num,
        max_loras_w,
        work_total,
        pair_slices,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        A0.stride(0),
        A0.stride(1),
        A0.stride(2),
        A0.stride(3),
        B0.stride(0),
        B0.stride(1),
        B0.stride(2),
        B0.stride(3),
        out_view.stride(0),
        out_view.stride(1),
        N_per_slice,
        n_tiles_per_program,
        n_chunks,
        token_mapping_factor=token_mapping_factor,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=add_inputs,
        BLOCK_R=BLOCK_R,
        actual_rank=rank,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SLICES=num_slices,

        # /-------------------- MetaX Modification --------------------\
        NUM_EXPERTS=num_experts,
        # \-----------------------------------------------------------/

        num_warps=nw,
        num_stages=ns,
    )
