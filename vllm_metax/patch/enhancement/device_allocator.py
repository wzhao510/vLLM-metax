# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Replace CUDA `CuMemAllocator` with the MetaX allocator to support
#       sleep mode on MACA.
#
# Affected versions: v0.21.0
# -----------------------------------------------

import gc
from vllm.device_allocator import MemAllocator
from vllm_metax.patch.utils import patch
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
)
from vllm.distributed.ec_transfer import (
    ensure_ec_transfer_shutdown,
)


@patch("vllm.device_allocator")
def get_mem_allocator_instance() -> MemAllocator:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()


@patch("vllm.v1.worker.gpu_worker", "Worker.shutdown")
def shutdown(self) -> None:
    gc.unfreeze()

    # has_kv_transfer_group can be None during interpreter shutdown.
    if ensure_kv_transfer_shutdown is not None:
        ensure_kv_transfer_shutdown()
    if ensure_ec_transfer_shutdown is not None:
        ensure_ec_transfer_shutdown()
    if self.profiler is not None:
        self.profiler.shutdown()

    if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
        weight_transfer_engine.shutdown()

    # Release GPU resources held by the model runner so that memory
    # can be reclaimed when running in-process
    if model_runner := getattr(self, "model_runner", None):
        model_runner.shutdown()

    # Release kept-alive cumem pools while the pluggable allocator wrappers
    # and callbacks are still alive, so MemPool teardown is not deferred to
    # interpreter finalization (pytorch/pytorch#145168).
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    if CuMemAllocator.instance is not None:
        CuMemAllocator.instance.release_pools()
