# vLLM-MetaX v0.32.0 发布说明

[English](/release/0.32.0/en.md){ .md-button }

发布日期：TODO（待发布确认） · 源码分支：[releases/v0.32.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.32.0)

本版本对齐上游 vLLM v0.32.0，并同步 dev030 路线上的 DeepEP/Distributed-PD、Qwen3.8-Flash-Next P/D、MiniMax-M3.1、GLM-5.3-Flash 与 DSV4.1 W4A8 性能特性。

## 1. 主要更新

- **DeepSeek-V4.1 / DSV4.1 W4A8**：支持 V4.1 量化配置统一，引入 W4A8 + DeepEP LL 的 masked INT8/source-packed 传输路径。
- **Qwen3.8-Flash-Next P/D**：支持基于 Nixl 的 prefill/decode 分离和远程 KV 传输，新增 `pd_proxy.py` 与 `Qwen4Exp` P/D 校验。
- **GLM-5.3-Flash**：新增 GLM5.next 模型实现及 sparse MLA indexer 适配，支持 MTP 与 kpool 路径。
- **MiniMax-M3.1 / Kimi-K3**：对齐 MiniMax M3.1 与 Kimi-K3 MXFP4/MTP 的运行路径，并抑制 CUDA-only 低延迟 GEMM 入口在 MACA 下误用。
- **attention backend**：注册 MetaX `FLASH_ATTN_DIFFKV` 后端，`MiMo-V2` DiffKV 通过 registry 原生选择。
- **MRV2**：补齐 post-capture warmup 与 async output 调试开关，修复 GLM4.6/4.7 MTP 内存越界问题。
- **patch 整理**：迁移 runtime patch 到 `patch/enhancement/performance`；增加 EPLB、Mooncake early-ready、source-packed MoE 等性能开关。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.32.0 | 3.9.0.x | 2.10 | 0.4.16 | TODO(镜像 URL) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.32.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| DeepSeek-V4 / DeepSeek-V4.1 | BF16/FP8/INT8/MXFP4 支持，DSV4.1 W4A8 DeepEPLL 传输路径 |
| Qwen3.8-Flash-Next | 新增 Qwen4Exp 系列模型，P/D 与 MTP 支持 |
| GLM-5.3-Flash | 新增 GLM5.next 实现和 sparse MLA indexer |
| MiniMax-M3 / M3.1 | 新增 M3.1 Flash、DSpark/MTP 适配 |
| Kimi-K3 | 支持 MXFP4 并禁用 CUDA-only 低延迟 GEMM |

## 4. 升级说明

1. 从 v0.30 升级时，需要按 v0.32 requirements 对应 mcoplib 0.4.16 与 PyTorch 2.10；源码快照依赖与发布镜像组合不同。
2. `patch/performance` 已迁移到 `patch/enhancement/performance`，不应继续转载旧路径。
3. Qwen-P/D 需要 Nixl/NNCCL 配置；`VLLM_METAX_MASKED_INT8_MOE` 默认开启，需要禁用时显式设置 `=0`。
4. MRV2 为 CUDA graph 默认模式；如需回退旧 runner 行为，按模型实际支持显式固定。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| 测试未覆盖 | `tests/` 目录部分用例仍使用旧 vLLM API，需先同步 baseline |
| source-packed W4A8 | 属于性能 path，需根据 mcoplib/DeepGem 版本 enable 并验证数值/吞吐 |
| DeepSeek-V4 TP | 仍需满足 `o_groups` 整除约束；TP 需要确认 checkpoint layout |
