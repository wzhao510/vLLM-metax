# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Purpose: run the V2 GPU n-gram scan kernel correctly on MACA.
# Added version: v0.32.0.
#
# Note: In upstream v0.32 the n-gram speculator scans candidate matches with a
#       length-MAX_N suffix preloaded into registers, and extracts each suffix
#       token via the Triton dynamic-indexing idiom
#           suf_val = tl.sum(tl.where(suf_iota == suf_idx, suffix, 0))
#       On MACA Triton 3.6 that idiom evaluates to 0 regardless of the suffix,
#       so `tok == suf_val` never holds, the scan writes an all-zero score for
#       every request, the finalizer falls back to the last-sampled fill value,
#       and n-gram acceptance collapses to ~2 % while output text stays correct.
#
#       Load each suffix token straight from the token buffer at
#       `seq_len - n_iter + j` with an in-range mask (other=-1), preserving the
#       upstream semantics exactly and avoiding the broken reduction. The score
#       packing (longest n, rightmost pos) and the finalize kernel are unchanged.
#
#       The replacement is installed on the provider module
#       (ngram.speculator); `NgramGPUSpeculator.propose` resolves the kernel
#       through the module global at call time, so no caller alias update is
#       needed beyond the shared patch utility.
#
# Affected versions: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be);
#     reproduced on MACA with mcoplib 0.4.16 / MACA 3.9.0.31 / triton 3.6.0,
#     2026-11-10. Standalone scan-kernel repro: scratch all-zero, drafts fill.
#
# Remove at: MACA Triton evaluates tl.sum(tl.where(vec == scalar, vec, 0))
#     correctly (verified by the standalone repro), or upstream drops the
#     suffix-preload idiom in favor of direct loads.
# -----------------------------------------------------------------------------

from vllm.triton_utils import tl, triton
from vllm_metax.patch.utils import patch


@patch(
    "vllm.v1.worker.gpu.spec_decode.ngram.speculator",
    "_ngram_scan_kernel",
)
@triton.jit
def _ngram_scan_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int64  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    L,  # int64 scalar (= max_model_len)
    MIN_N: tl.constexpr,
    MAX_N: tl.constexpr,
    MAX_N_PO2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1

    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req_state_idx).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    eligible_row = (num_sampled > 0) & (seq_len >= MIN_N)

    scratch_off = b * scratch_stride + blk

    # Ineligible rows, and blocks fully past the last candidate match
    # position, write 0 and exit.
    if not (eligible_row & (blk * BLOCK_L <= seq_len - MIN_N - 1)):
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    row_off = req_state_idx * token_ids_stride

    pos_iota = tl.arange(0, BLOCK_L).to(tl.int64)
    pos = blk * BLOCK_L + pos_iota  # ascending

    best_score = tl.zeros([BLOCK_L], dtype=tl.int64)

    for n_iter in tl.static_range(MIN_N, MAX_N + 1):
        max_pos_n = seq_len - n_iter - 1
        match = (pos >= 0) & (pos <= max_pos_n)
        for j in tl.static_range(0, n_iter):
            tok = tl.load(
                token_ids_ptr + row_off + (pos + j),
                mask=match,
                other=0,
            ).to(tl.int32)
            # NOTE(metax): load the suffix token directly from the token buffer
            # at seq_len - n_iter + j. The upstream idiom
            # `tl.sum(tl.where(suf_iota == suf_idx, suffix, 0))` returns 0 on
            # MACA Triton, which makes `tok == suf_val` always false and drops
            # every n-gram match to zero.
            suf_pos = seq_len - n_iter + j
            suf_in_range = (suf_pos >= 0) & (suf_pos < seq_len)
            suf_val = tl.load(
                token_ids_ptr + row_off + suf_pos,
                mask=suf_in_range,
                other=-1,
            ).to(tl.int32)
            match = match & (tok == suf_val)

        # Pack (n, pos) so a single max yields longest-n, rightmost-pos.
        cand = n_iter * Lp1 + pos + 1
        best_score = tl.where(match, cand, best_score)

    block_best = tl.max(best_score, axis=0)
    tl.store(scratch_ptr + scratch_off, block_best)
