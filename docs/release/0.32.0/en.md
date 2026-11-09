# vLLM-MetaX v0.32.0 Release Notes

[中文](/release/0.32.0/zh.md){ .md-button }

Release date: TODO · Source branch: [releases/v0.32.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.32.0)

This release aligns with upstream vLLM v0.32.0 and carries the dev030 DeepEP/Distributed-PD, Qwen3.8-Flash-Next P/D, MiniMax-M3.1, GLM-5.3-Flash and DSV4.1 W4A8 features.

## 1. Main changes

- **DeepSeek-V4.1 / DSV4.1 W4A8**: unify V4.1 quant config and add W4A8 + DeepEP LL masked INT8/source-packed transport.
- **Qwen3.8-Flash-Next P/D**: add Nixl-based prefill/decode split and remote KV transfer, incl. `pd_proxy.py` and Qwen4Exp P/D validation.
- **GLM-5.3-Flash**: add GLM5.next model implementation, sparse MLA indexer, MTP and kpool paths.
- **MiniMax-M3.1 / Kimi-K3**: add M3.1 Flash and Kimi-K3 MXFP4/MTP support; disable CUDA-only low-latency GEMM on MACA.
- **Attention backends**: register MetaX `FLASH_ATTN_DIFFKV`; MiMo-V2 uses it through the v0.32 backend registry.
- **MRV2**: add post-capture warmup and async output toggles; fix GLM4.6/4.7-MTP memory faults.
- **Patch layout**: move runtime patches to `patch/enhancement/performance`; add EPLB, Mooncake early-ready and source-packed MoE switches.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.32.0 | 3.9.0.x | 2.10 | 0.4.16 | TODO(image URL) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.32.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| DeepSeek-V4 / DeepSeek-V4.1 | BF16/FP8/INT8/MXFP4 support; DSV4.1 W4A8 DeepEP LL transport |
| Qwen3.8-Flash-Next | Qwen4Exp model family; P/D and MTP support |
| GLM-5.3-Flash | New GLM5.next implementation and sparse MLA indexer |
| MiniMax-M3 / M3.1 | New M3.1 Flash, DSpark/MTP adaptations |
| Kimi-K3 | MXFP4 support with CUDA-only low-latency GEMM disabled |

## 4. Upgrade notes

1. When upgrading from v0.30, use the requirements paired with mcoplib 0.4.16 and PyTorch 2.10.
2. `patch/performance` moved to `patch/enhancement/performance`.
3. Qwen-P/D requires Nixl/NCCL; `VLLM_METAX_MASKED_INT8_MOE` defaults to 1.
4. MRV2 is the default CUDA-graph runner; set explicit toggles for older runner behavior.

## 5. Known issues

| Area | Notes |
| --- | --- |
| Test coverage | Some `tests/` cases still target old vLLM APIs and need baseline sync |
| Source-packed W4A8 | Performance path; requires matching mcoplib/DeepGem and numerical/throughput validation |
| DeepSeek-V4 TP | TP must satisfy checkpoint `o_groups` divisibility |
