# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Provide a CUDA API compatibility wrapper backed by MetaX libraries.
#
# Affected versions: v0.21.0
# -----------------------------------------------

"""This file is a pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

# this line makes it possible to directly load `libcudart.so` using `ctypes`
from vllm_metax.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm_metax.patch.utils import patch

patch("vllm.distributed.device_communicators.cuda_wrapper", "cudaIpcMemHandle_t")(
    cudaIpcMemHandle_t
)

patch(
    "vllm.distributed.device_communicators.cuda_wrapper",
    "CudaRTLibrary",
)(CudaRTLibrary)
