# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Allow the dynamic per-token INT8 MoE method to run under EPLB. EPLB only
#       remaps expert ids before dispatch and rearranges per-expert weight
#       tensors; the INT8 kernels are untouched. Upstream gates it on a
#       per-method ``supports_eplb`` flag that only the FP8 and unquantized
#       methods set, so enabling EPLB otherwise raises ``NotImplementedError``.
#
# Affected versions: v0.30.0 (ced6857afa0e)
#
# Remove at: Remove after upstream sets ``supports_eplb`` for the compressed-tensors
#            INT8 MoE method, or after the MetaX method sets it itself.
# -----------------------------------------------------------------------------
"""EPLB support flag for the MetaX compressed-tensors W8A8 INT8 MoE method."""

from vllm_metax.patch.utils import patch


@patch(
    "vllm_metax.registry.quant_config.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
    "CompressedTensorsW8A8Int8MoEMethod.supports_eplb",
)
@property
def supports_eplb(self) -> bool:
    # /-------------------- MetaX Modification --------------------\
    return True
    # \-------------------- MetaX Modification --------------------/
