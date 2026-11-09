# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Register MetaX mixed-precision linear kernels for out-of-tree dispatch.
#
# Affected versions: v0.21.0
# -----------------------------------------------

from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel as vllm_ExllamaLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.platforms import PlatformEnum, current_platform

import torch

def _allocate_gptq_workspaces(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    weight_bits: int,
    group_size: int,
    desc_act: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    perm_space = torch.empty(0, device=x_2d.device)
    temp_space = torch.empty(0, device=x_2d.device)

    if weight_bits not in (4, 8) and group_size not in (64, 128):
        return perm_space, temp_space

    if desc_act:
        perm_space = torch.empty(
            x_2d.shape[0],
            x_2d.shape[1],
            dtype=torch.float16,
            device=x_2d.device,
        )

    if x_2d.dtype == torch.bfloat16:
        temp_space = torch.zeros(
            x_2d.shape[0],
            qweight.shape[1],
            dtype=torch.float32,
            device=x_2d.device,
        )

    return perm_space, temp_space


# This used for:
#  - warmup in process_weights_after_loading
#  - forward implement
def _apply_gptq_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None,
    g_idx: torch.Tensor,
    use_exllama: bool,
    weight_bits: int,
    group_size: int,
    desc_act: bool,
) -> torch.Tensor:
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (qweight.shape[-1],)

    # mcoplib already exposes the vLLM 0.30 GPTQ GEM signature:
    #   gptq_gemm(a, qweight, qzeros, scales,
    #             use_exllama, use_v2_format, bit)
    output = torch.ops._C.gptq_gemm(
        reshaped_x,
        qweight,
        qzeros,
        scales,
        use_exllama,
        False,  # use_v2_format
        weight_bits,
    )

    if bias is not None:
        output.add_(bias)
    return output.reshape(out_shape)


def _ensure_gptq_shuffle_compat() -> None:
    """Adapt vLLM 0.30's two-arg gptq_shuffle to the MetaX three-arg op.

    vLLM 0.30 removed GPTQ group/dynamic activation ordering (upstream PR
    #54809) and now calls ``gptq_shuffle(q_weight, bit)``. The MetaX extension
    still exposes the older ``gptq_shuffle(q_weight, q_perm, bit)`` schema, so
    install a thin shim before the vLLM Exllama weight repack path runs.
    """
    import vllm._custom_ops as vllm_ops

    current = getattr(vllm_ops, "gptq_shuffle", None)
    if getattr(current, "__vllm_metax_gptq_shuffle_compat__", False):
        return

    def compat(q_weight: torch.Tensor, bit: int) -> None:
        op = getattr(torch.ops._C, "gptq_shuffle", None)
        if op is None:
            current(q_weight, bit)
            return

        # Fast path: the backend already exposes the vLLM 0.30 two-arg schema.
        try:
            op(q_weight, bit)
            return
        except (TypeError, RuntimeError):
            pass

        # MetaX extension path: shim into the older three-arg schema.
        q_perm = torch.empty((0,), dtype=torch.int32, device=q_weight.device)
        op(q_weight, q_perm, bit)

    compat.__vllm_metax_gptq_shuffle_compat__ = True
    vllm_ops.gptq_shuffle = compat


class MacaExllamaLinearKernel(vllm_ExllamaLinearKernel):
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm and Maca",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        # ------------------------------------------------
        # On maca we support both float16 and bfloat16
        # ------------------------------------------------
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Exllama only supports float16 and bfloat16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _ensure_gptq_shuffle_compat()
        super().process_weights_after_loading(layer)

        c = self.config
        if c.group_size not in (64, 128):
            return

        w_q, w_s, w_zp = self._get_weight_params(layer)
        assert w_zp is not None, "Zero points are required by Exllama"
        # vLLM >= 0.30 removed GPTQ group/dynamic activation ordering, so the
        # layer no longer carries a g_idx weight. Keep the MetaX GEM kernel
        # happy by passing the same empty tensor sentinel it uses for
        # "no act reordering".
        g_idx = torch.empty((0,), dtype=torch.int32, device="meta")

        if w_s.dtype == torch.bfloat16:
            return

        # partition_weight_shape[0] is the unpacked input width, equivalent to
        # the old qweight.shape[0] * (32 // weight_bits) warmup shape.
        warmup_x = torch.randn(
            1,
            c.partition_weight_shape[0],
            dtype=w_s.dtype,
            device=w_q.device,
        )
        _ = _apply_gptq_impl(
            warmup_x,
            w_q,
            w_s,
            w_zp,
            None,
            g_idx,
            True,
            c.weight_type.size_bits,
            c.group_size,
            False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        w_q, w_s, w_zp = self._get_weight_params(layer)

        assert w_zp is not None, "Zero points are required by Exllama"
        g_idx = torch.empty((0,), dtype=torch.int32, device="meta")
        return _apply_gptq_impl(
            x,
            w_q,
            w_s,
            w_zp,
            bias,
            g_idx,
            True,
            c.weight_type.size_bits,
            c.group_size,
            False,
        )


register_linear_kernel(
    kernel_class=MacaExllamaLinearKernel, platform=PlatformEnum.OOT, kernel_type="mp"
)
