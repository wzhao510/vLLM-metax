# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: support glm5next, need patch Glm5NextIndexerCache and Glm5NextTailCache
#
# Affected versions: v0.26.0
#
# Remove at: after refactored
# -----------------------------------------------------------------------------

from vllm_metax.patch.utils import patch

import torch
from torch import nn

from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from vllm.models.glm5next.nvidia.attention import Indexer

from vllm_metax.models.glm5next.attention import Glm5NextIndexerCache, Glm5NextTailCache

logger = init_logger(__name__)

@patch(
    "vllm.models.glm5next.nvidia.attention",
    "Indexer.__init__",
)
def __init__(
    self,
    vllm_config: VllmConfig,
    config: Glm5NextConfig,
    hidden_size: int,
    q_lora_rank: int,
    quant_config: QuantizationConfig | None,
    cache_config: CacheConfig | None,
    topk_indices_buffer: torch.Tensor | None,
    prefix: str = "",
):
    super(Indexer, self).__init__()
    self.vllm_config = vllm_config
    self.config = config
    self.quant_config = quant_config
    # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
    # Indexer is only constructed for v32 configs, where these sparse-indexer
    # fields are guaranteed populated; narrow away the `int | None` declared
    # on Glm5NextConfig for the optional-indexer case.
    assert config.index_topk is not None
    assert config.index_n_heads is not None
    assert config.index_head_dim is not None
    assert config.index_kpool is not None
    self.topk_tokens = config.index_topk
    self.n_head = config.index_n_heads  # 64
    self.head_dim = config.index_head_dim  # 128
    self.rope_dim = config.qk_rope_head_dim  # 64
    self.index_kpool = config.index_kpool
    self.q_lora_rank = q_lora_rank  # 1536

    # kpool
    self.index_kpool_compress_ape = nn.Parameter(
        torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
    )
    # Keep the checkpoint name ``index_kpool_compress_gate`` without a
    # ``.weight`` suffix. F.linear consumes its [head_dim, hidden_size] shape.
    self.index_kpool_compress_gate = nn.Parameter(
        torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
    )

    # no tensor parallel, just replicated
    self.wq_b = ReplicatedLinear(
        self.q_lora_rank,
        self.head_dim * self.n_head,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_b",
    )
    # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
    # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
    self.wk_weights_proj = MergedColumnParallelLinear(
        hidden_size,
        [self.head_dim, self.n_head],
        bias=False,
        quant_config=None,
        disable_tp=True,
        prefix=f"{prefix}.wk_weights_proj",
    )
    self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
    self.softmax_scale = self.head_dim**-0.5

    # Hadamard-128 rotation of the indexer query is fused with the FP8
    # quant (see forward: fwht128_quant_fp8) -- no precomputed matrix.

    self.scale_fmt = "ue8m0"
    self.quant_block_size = 128  # TODO: get from config
    self.topk_indices_buffer = topk_indices_buffer
    self._wp_fp32: torch.Tensor | None = None

    # NOTE: (zyongye) we use fp8 naive cache,
    #       where we store value in fp8 and scale in fp32
    #       per self.quant_block_size element
    self.k_cache = Glm5NextIndexerCache(
        head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
        dtype=torch.uint8,
        prefix=f"{prefix}.k_cache",
        cache_config=cache_config,
        index_kpool=self.index_kpool,
    )
    # Paged tail cache (in-progress pool's raw K + gate score). Written by
    # prefill (seeds the boundary pool) and decode (per-step stash); read by
    # the decode kernel to compress the boundary pool. Transferred across PD
    # so the decode side sees the prefill tail. See KpoolTailSpec/Manager.
    self.tail_cache = Glm5NextTailCache(
        head_dim=self.head_dim,
        dtype=torch.bfloat16,
        prefix=f"{prefix}.tail_cache",
        cache_config=cache_config,
        index_kpool=self.index_kpool,
    )
    self.max_model_len = vllm_config.model_config.max_model_len
    self.prefix = prefix
    from vllm_metax.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

    self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
    
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
    self.indexer_op = SparseAttnIndexerKpool(
        self.k_cache,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.head_dim,
        self.max_model_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
        tail_cache=self.tail_cache,
    )
