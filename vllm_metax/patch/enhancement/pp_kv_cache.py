# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Note: PP projection can leave global KV groups with no local layers. Preserve
# those groups and their IDs, but allocate no tensor views for them. Otherwise
# empty UniformTypeKVCacheSpecs may raise StopIteration during allocation.
#
# Affected versions: vLLM 0.30.0, tested get_kv_cache_config_from_groups snapshot
# 2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2.
# Remove at: upstream skips empty local groups during KV tensor allocation.
#
# This replacement mirrors the tested upstream function, including the attention
# free, HiSparse and GLM5 layout paths. Its only allocation change is the explicit
# empty-group guard. It does not fix Mooncake physical/logical block-size mapping.

from collections import defaultdict
from typing import cast

from vllm.config import VllmConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    _get_kv_cache_bytes_per_block,
    compute_layout_strides,
    get_hisparse_host_pool_bytes,
    get_hisparse_kv_cache_config,
    may_override_num_blocks,
    validate_kv_cache_layout,
)
from vllm_metax.patch.utils import patch


@patch("vllm.v1.core.kv_cache_utils", "get_kv_cache_config_from_groups")
def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    if vllm_config.attention_config.hisparse_config is not None:
        host_budget = get_hisparse_host_pool_bytes(vllm_config)
        return get_hisparse_kv_cache_config(
            vllm_config, kv_cache_groups, available_memory, host_budget
        )

    glm5_layout_fn = getattr(kv_cache_utils, "_glm5_next_tensor_layout", None)
    if glm5_layout_fn is not None and (
        glm5_layout := glm5_layout_fn(kv_cache_groups)
    ) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs

        kv_cache_tensors: list[KVCacheTensor] = []

        def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )

        for index, mla_name in enumerate(mla_names):
            offset = index * mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)

        idx_base = len(mla_names) * mla_page * num_blocks
        for index, idx_name in enumerate(idx_names):
            offset = idx_base + index * idx_page * num_blocks
            add_tensor(idx_name, attn_specs[idx_name], offset)
            if tail_names:
                tail_name = tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, kv_cache_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups, layout)
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    size = bytes_per_block * num_blocks

    # Groups alias from byte 0. Spec regions are laid out differently:
    #
    # block-outer (the same packing repeats for every block):
    # group 0: | blk 0 [ A | B  | pad ] | blk 1 [ A | B  | pad ] | ...
    # group 1: | blk 0 [  C  |    D   ] | blk 1 [  C  |    D   ] | ...
    #          |<--- bytes_per_block -->|
    #
    # layer-outer (only supported for uniform page sizes or single-group models):
    # group 0: | A [ blk 0 | blk 1 | ... ] | B [ blk 0 | blk 1 | ... ] |
    # group 1: | C [ blk 0 | blk 1 | ... ] | D [ blk 0 | blk 1 | ... ] |

    kv_cache_tensors = []
    for group in kv_cache_groups:
        # Keep PP-global group IDs, but allocate no views for non-local groups.
        if not group.layer_names:
            continue
        group_spec = group.kv_cache_spec
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec,
                num_blocks,
                len(layer_names),
                layout,
                fixed_strides=(None, interleaved_block_stride, None, None, None),
            )
            offset = (
                byte_offset
                * max(layer_stride, spec.page_size_bytes)
                // spec.page_size_bytes
            )
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=offset,
                )
            )
            byte_offset += len(layer_names) * spec.page_size_bytes

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )
