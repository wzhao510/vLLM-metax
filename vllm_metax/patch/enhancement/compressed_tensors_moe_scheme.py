# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Route the MoE expert scheme to the dedicated `FusedMoE` group when the
#     checkpoint declares one.  Injected at runtime so upstream vLLM sources
#     stay untouched.
#
#     vLLM's `CompressedTensorsConfig._add_fused_moe_to_target_scheme_map`
#     unconditionally aliases `RoutedExperts` to the `Linear` group:
#
#         self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["Linear"]
#
#     GLM-5.3-Flash checkpoints describe the experts as a *separate* group with
#     its own bit-widths, e.g.
#
#         group_0  targets=['Linear']    weights=8  input=8
#         group_1  targets=['FusedMoE']  weights=4  input=8
#
#     Alias to `Linear` therefore resolves the experts to the 8-bit scheme
#     (`_is_dynamic_token_w8a8` -> CompressedTensorsW8A8Int8MoEMethod).  That is
#     both wrong and expensive: the checkpoint stores 4-bit nibble-packed int32
#     tensors (`weight_packed`), so the 8-bit method allocates per-expert int8
#     weights at twice the size and the rank OOMs during load.
#
#     This patch keeps the upstream fallback intact but prefers `FusedMoE`
#     whenever that target is present, so `RoutedExperts` inherits the 4-bit
#     scheme (`_is_dynamic_token_w4a8_int` -> CompressedTensorsW4A8Int4MoEMethod).
#
# Affected versions: vLLM 0.27.2rc1.dev22 on MetaX MACA, reviewed 2026-10-15.
# Remove at: upstream vLLM learns to prefer a dedicated `FusedMoE` target group.
# -----------------------------------------------------------------------------

from __future__ import annotations

from vllm_metax.patch.utils import patch


# ---------------------------------------------------------------------------
# target_scheme_map resolution
# ---------------------------------------------------------------------------

@patch(
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
    "CompressedTensorsConfig._add_fused_moe_to_target_scheme_map",
)
def _add_fused_moe_to_target_scheme_map(self) -> None:
    """Bind `RoutedExperts` to the best matching group's scheme.

    Order of preference: a dedicated `FusedMoE` group (its own bit-widths),
    then the upstream fallback to the `Linear` group.
    """
    if (
        "Linear" not in self.target_scheme_map
        or "RoutedExperts" in self.target_scheme_map
    ):
        return

    if "FusedMoE" in self.target_scheme_map:
        self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["FusedMoE"]
        return

    self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["Linear"]
