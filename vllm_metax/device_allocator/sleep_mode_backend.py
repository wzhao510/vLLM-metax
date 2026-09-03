# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable sleep-mode backends (RFC #34303).

vLLM's sleep/wake-up today is hard-wired to ``CuMemAllocator``: the GPU worker
calls ``allocator.sleep(...)`` / ``allocator.wake_up(...)`` directly. RFC #34303
proposes additional mechanisms for freeing and restoring GPU state - CUDA
process checkpoint, CRIU, durable snapshot/restore - that share the *dispatch*
(``/sleep`` endpoint -> engine -> executor -> worker) but differ in *mechanism*
and in which resources they preserve (NCCL communicators, compiled kernels,
CUDA graphs, survival across process restart).

This module introduces a thin backend abstraction so those mechanisms can be
selected by name without changing the public API. The default ``cumem`` backend
wraps today's ``CuMemAllocator`` path 1:1, so existing users see no behavior
change. The factory mirrors ``KVConnectorFactory`` and lets third-party
backends register through a ``vllm.general_plugins`` entry point at import time.
"""

from __future__ import annotations

from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)


class MacaMemBackend(SleepModeBackend):
    """Default backend.

    Wraps the platform sleep-mode allocator exactly as the GPU worker did
    before this abstraction existed, so behavior is identical to vLLM's current
    sleep/wake-up. ``get_mem_allocator_instance()`` resolves to
    ``CuMemAllocator`` on CUDA and ``XpuMemAllocator`` on XPU; suspend offloads
    per-allocation between GPU and host, with NCCL buffers left untouched (they
    are allocated outside the allocator pool).
    """

    def suspend(self, level: int = 1) -> None:
        from vllm_metax.device_allocator import get_mem_allocator_instance

        self._state = "SUSPENDED"
        allocator = get_mem_allocator_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    def resume(self, tags: list[str] | None = None) -> None:
        from vllm_metax.device_allocator import get_mem_allocator_instance

        self._state = "RESUMING"
        allocator = get_mem_allocator_instance()
        allocator.wake_up(tags)
        self._state = "RUNNING"

    @classmethod
    def preserves_communicators(cls) -> bool:
        # Communicator buffers (e.g. NCCL) live outside CuMemAllocator's pool, so
        # an allocator-level sleep leaves them intact (no reinit needed on resume).
        return True


SleepModeBackendFactory.register_backend(
    "macamem",
    "vllm_metax.device_allocator.sleep_mode_backend",
    "MacaMemBackend",
)
