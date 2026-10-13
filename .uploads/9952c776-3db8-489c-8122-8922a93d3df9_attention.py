# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import (
    VllmConfig,
)

from vllm_metax.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache that stores kpool-compressed entries.

    Setting ``tokens_per_state = index_kpool`` on the KV cache spec makes vLLM's
    indexer metadata builder emit pool-granular ``slot_mapping`` /
    ``seq_lens`` / ``cu_seq_lens`` / ``page_table`` for free, and shrinks the
    cache allocation store one state per ``index_kpool`` tokens. The pool
    *content* (softmax-weighted sum vs keep-every-Nth) is computed by the
    kpool compress kernel inside the indexer op — the cache only provides the
    addressing, which is identical for both schemes.

    The indexer shares one block with the co-located MLA (a single
    ``MLAAttentionSpec`` / block_table), so ``block_size`` is the model-wide
    ``cache_config.block_size``. DeepGEMM's paged-MQA kernel
    (``csrc/apis/attention.hpp``) requires ``block_kv`` to be exactly 32 or
    64, so the storage block is virtually split into pool pages of the
    largest such size that tiles it (``storage_kernel_block_size``); this
    needs ``block_size`` to be a multiple of ``index_kpool * 32`` (512 for
    ``index_kpool = 16``). A smaller block (e.g. the default 64) silently
    collapses ``storage_block_size`` (64 // 16 = 4) and only fails later at
    the opaque C++ assert; ``get_kv_cache_spec`` guards this up front
    instead.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "Glm5NextIndexerCache expects index_kpool > 1"
        # Keep chunked-prefill boundaries aligned to complete pools.
        assert cache_config.block_size % index_kpool == 0, (
            "Glm5NextIndexerCache: cache_config.block_size "
            f"({cache_config.block_size}) must be a multiple of index_kpool "
            f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
        )
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        # ``tokens_per_state`` is the KV-spec representation of kpool
        # compression in the current cache-layout API.
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, tokens_per_state=self._index_kpool)

        # DeepGEMM paged-MQA takes block_kv in {32, 64}; the storage block
        # (= block_size // index_kpool) is virtually split into pool pages of
        # the largest such size that tiles it, so it must be a multiple of 32.
        storage_block_size = spec.block_size // self._index_kpool
        assert (
            spec.block_size % self._index_kpool == 0 and storage_block_size % 32 == 0
        ), (
            "Glm5NextIndexerCache: kpool indexer requires cache block_size to "
            f"be a multiple of index_kpool * 32 ({self._index_kpool * 32}) so "
            "that DeepGEMM paged-MQA pool pages (32 or 64 entries) tile the "
            f"storage block, got block_size={spec.block_size} -> "
            f"storage_block_size={storage_block_size}."
        )
        max_page_size = max(PAGED_MQA_PAGE_SIZES)
        min_page_size = min(PAGED_MQA_PAGE_SIZES)
        if storage_block_size <= max_page_size:
            page_size = storage_block_size
        elif storage_block_size % max_page_size == 0:
            page_size = max_page_size
        else:
            page_size = min_page_size
        return replace(
            spec,
            storage_block_size=page_size * self._index_kpool,
        )


class Glm5NextTailCache(DeepseekV32IndexerCache):
    """Paged circular buffer for the kpool indexer's in-progress (tail) pool.

    Holds the trailing incomplete pool's raw K + gate score: one block of
    ``index_kpool`` slots per request, overwritten in place by ``pos % kpool``
    as decode/spec-decode advances. Prefill seeds it (instead of discarding the
    tail raw K+gate); the connector transfers it across PD; decode reads it to
    compress the boundary pool correctly. ``KpoolTailSpec`` /
    ``KpoolTailManager`` provide the no-prune, 1-block/req allocation that lets
    the in-progress pool survive across steps and across transfer.

    Stores raw bf16 K (``head_dim``) as the "K" half of each block and the
    bf16 gate score (``head_dim``) as the "V" half -- not the fp8-compressed
    entry, which lives in ``Glm5NextIndexerCache``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "Glm5NextTailCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # The two head slots form [K, gate score] in the generic
        # [block, head, state, content] cache view.
        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self._index_kpool,
        )

    def get_attn_backend(self):
        from vllm_metax.v1.attention.backends.mla.indexer import MacaKpoolTailBackend as KpoolTailBackend

        return KpoolTailBackend