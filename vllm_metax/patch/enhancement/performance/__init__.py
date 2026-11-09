# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX grouped-topk dispatch and speculative-decode batch ordering at plugin
#     initialization. This initializer applies the patches by importing the child
#     modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All performance patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

from . import gumbel_sample  # noqa: F401
from . import grouped_topk_router  # noqa: F401
from . import int8_moe_eplb  # noqa: F401
from . import speculative_decode_perf  # noqa: F401

# NOTE(metax): MiMo-V2 DiffKV is handled natively in v0.32 by
# vllm_metax.v1.attention.registry, which points FLASH_ATTN_DIFFKV at the
# MetaX FlashAttentionDiffKVBackend. No per-model init patch is needed; do not
# reintroduce the v0.30 mimo_v2_fa_diffkv registration workaround here.

# Opt-in: unified source-packed MoE (W4A8 & W8A8)
from vllm_metax import envs as mx_envs

if mx_envs.VLLM_METAX_SOURCE_PACKED_MOE:
    from . import packed_moe_adapter  # noqa: F401

if mx_envs.VLLM_METAX_MOONCAKE_EARLY_READY:
    from . import mooncake_early_ready_loader  # noqa: F401
