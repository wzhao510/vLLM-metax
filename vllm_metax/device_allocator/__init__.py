# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm.device_allocator import MemAllocator


def get_mem_allocator_instance() -> MemAllocator:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()
