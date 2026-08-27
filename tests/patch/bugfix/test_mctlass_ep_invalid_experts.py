# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.

from typing import Any

import pytest
import torch
from vllm_metax.patch.bugfix import mctlass_ep_invalid_experts as patch_module


@pytest.mark.parametrize(
    ("enabled", "has_expert_map", "requested", "expected"),
    [
        (True, True, False, True),
        (False, True, False, False),
        (True, False, False, False),
        (False, False, True, True),
    ],
)
def test_ignore_invalid_experts_for_mctlass_ep(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    has_expert_map: bool,
    requested: bool,
    expected: bool,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_moe_align_block_size(*args: Any, **kwargs: Any) -> tuple[None, None, None]:
        captured_kwargs.update(kwargs)
        return None, None, None

    monkeypatch.setattr(
        patch_module,
        "moe_align_block_size",
        fake_moe_align_block_size,
    )
    monkeypatch.setattr(
        patch_module.mx_envs,
        "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE",
        enabled,
    )

    expert_map = torch.tensor([0]) if has_expert_map else None
    patch_module._prepare_expert_assignment(
        torch.tensor([[0]], dtype=torch.int32),
        {"BLOCK_SIZE_M": 16},
        1,
        1,
        1,
        expert_map,
        ignore_invalid_experts=requested,
    )

    assert captured_kwargs["ignore_invalid_experts"] is expected
