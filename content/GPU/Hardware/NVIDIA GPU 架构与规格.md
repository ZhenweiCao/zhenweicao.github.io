---
aliases:
  - NVIDIA GPU Specs
  - H200 GPU Spec
  - Blackwell B200 VS B300
  - GB200 NVL72
  - GB300 NVL72
updated: 2026-06-03
tags:
  - gpu-computing
  - gpu-architecture
  - blackwell-gpu
  - hopper-gpu
---
# NVIDIA GPU 架构与规格

## 定位

这篇是 GPU 规格参考页，重点解决一个问题：看到 H200、B200、B300、GB200、GB300、DGX、NVL72 这些名字时，先判断它们属于哪一层，再查对应规格。

如果你想建立硬件心智模型，先读 [[GPU 硬件背景地图]]；如果你想理解 Ampere -> Hopper -> Blackwell 的 kernel 编程范式变化，读 [[GPU 硬件架构背景与编程范式]]；如果你正在做 GEMM 优化，配合 [[CUDA GEMM 矩阵乘法优化指南]] 一起看。

相关主文档：

- [[GPU 知识库索引]]
- [[GPU 初学者术语表]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]

## 先看口径

NVIDIA 的数据中心产品经常按不同层级发布规格。读表时先问“这是一颗 GPU，还是一台系统，还是一个 rack？”

| 口径 | 例子 | 怎么理解 |
|------|------|----------|
| GPU 单卡 / 单 GPU | H200 SXM、H200 NVL、B200、B300 | 一颗 GPU 的显存、带宽、Tensor Core 峰值、TDP。 |
| Grace Blackwell Superchip | GB200 Grace Blackwell Superchip | 1 颗 Grace CPU + 2 颗 Blackwell GPU，通过 NVLink-C2C 连接。 |
| DGX 系统 | DGX B300 | 一台 8-GPU 系统，规格是整机总量。 |
| NVL72 rack | GB200 NVL72、GB300 NVL72 | 72-GPU rack-scale 系统，规格是整个 NVLink domain 或整 rack 总量。 |
| sparse / dense | `144 / 108 PFLOPS`、`1440 / 1080 PFLOPS` | 表格脚注常用 sparse / dense 两个口径，不能只摘一个数。 |
| aggregate bandwidth | `14.4 TB/s aggregate`、`130 TB/s` | 通常是系统级互联总带宽，不是单 GPU 带宽。 |

两个最容易踩的坑：

1. 不要把 GB200/GB300 NVL72 的整 rack 数字直接写成 B200/B300 单 GPU 规格。
2. 不要把 Tensor Core sparse 峰值当成 dense 峰值；官方表格如果写 `sparse | dense`，要保留两个数。

## 代际地图

| 代际 | 代表产品 | 编程和优化重点 |
|------|----------|----------------|
| Ampere | A100、A800 | 第三代 Tensor Core，TF32/BF16，`cp.async` global-to-shared 异步拷贝，MIG。GEMM 优化重点是 CTA tile、shared memory double buffering、`ldmatrix` + `mma.sync`。 |
| Hopper | H100、H200 | 第四代 Tensor Core，FP8 Transformer Engine，TMA，WGMMA，Thread Block Cluster / DSMEM。优化重点从“很多线程搬数据”转向 TMA bulk copy、warp specialization、cluster 内协作。 |
| Blackwell | B200、GB200 | 第五代 Tensor Core，FP4/FP6/FP8，第二代 Transformer Engine，第五代 NVLink，更大的 L2 和系统级 NVLink domain。优化重点扩展到 dtype、scale tensor、layout、descriptor、cluster 和 persistent/grouped dispatch。 |
| Blackwell Ultra | B300、GB300、DGX B300 | 面向 reasoning inference 强化，官方重点是更高 dense FP4、attention 加速和更大 HBM 容量。优化时更关注长上下文、batching、KV cache、MoE/grouped GEMM 和多 GPU 通信。 |

从 CUDA kernel 学习角度看，**基础模型没有变**：还是 grid、block/CTA、warp、thread，还是 register、shared memory、L2、HBM。变化的是高性能 kernel 越来越依赖专用数据搬运、Tensor Core 指令、低精度 scale 体系和多 GPU 互联。

## 编译与架构口径

| 产品 | 架构口径 | CUDA compute capability | 初学者备注 |
|------|----------|-------------------------|------------|
| H100 / H200 | Hopper | 9.0 | 低层 Tensor Core 特性常见 `sm_90a` 目标。 |
| B200 | Blackwell data center | 10.0 | CUTLASS 文档把 B200 列为 `sm_100` / `sm_100a`。 |
| B300 | Blackwell Ultra | 10.x（以官方为准） | Blackwell Ultra 在公开 CUTLASS 文档中常与 `sm_100a` 路径复用；具体新增的低层 SM 编号以最新 CUDA 13.x Programming Guide 附录与 CUTLASS Blackwell Functionality 文档为准，避免凭记忆写死 `10.3`。 |
| RTX 50 系列 | Blackwell consumer/workstation 路线 | 12.0 | `sm_120` / `sm_120a`，和数据中心 `sm_100` 不是同一个编译目标，不能混用 `sm100a` 假设。 |

`sm_90a`、`sm_100a`、`sm_120a` 里的 `a` 表示使用 architecture-accelerated features。它们能打开更低层的专用指令能力，但也更绑定具体架构；新手先用库、CUTLASS/Triton 和 profiler 学习，不建议一开始手写这些 PTX 指令。

> 校对提示（2026-05-30）：B300 的 compute capability 数字在公开渠道存在版本差异。读到任何 `10.3` / `10.x` / `sm_103` 等表述时，请回到最新 [CUDA Programming Guide 附录 H](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 与 [CUTLASS Blackwell Functionality](https://docs.nvidia.com/cutlass/latest/) 交叉确认。

## H200 规格

H200 属于 Hopper 架构，核心价值是 HBM3e 带来的更大显存和更高带宽。官方页面给出的关键规格如下。**所有 Tensor Core 数值统一以 `sparse / dense` 双口径列出**（dense = sparse / 2），避免误用 sparse 数当 dense。

| 项目 | H200 SXM | H200 NVL |
|------|----------|----------|
| GPU Memory | 141 GB HBM3e | 141 GB HBM3e |
| GPU Memory Bandwidth | 4.8 TB/s | 4.8 TB/s |
| FP64 | 34 TFLOPS | 30 TFLOPS |
| FP64 Tensor Core | 67 TFLOPS | 60 TFLOPS |
| FP32 | 67 TFLOPS | 60 TFLOPS |
| TF32 Tensor Core (sparse / dense) | 989 / 494.5 TFLOPS | 835 / 417.5 TFLOPS |
| BF16 Tensor Core (sparse / dense) | 1,979 / 989.5 TFLOPS | 1,671 / 835.5 TFLOPS |
| FP16 Tensor Core (sparse / dense) | 1,979 / 989.5 TFLOPS | 1,671 / 835.5 TFLOPS |
| FP8 Tensor Core (sparse / dense) | 3,958 / 1,979 TFLOPS | 3,341 / 1,670.5 TFLOPS |
| INT8 Tensor Core (sparse / dense) | 3,958 / 1,979 TOPS | 3,341 / 1,670.5 TOPS |
| TDP | up to 700 W | up to 600 W |
| MIG | up to 7 MIGs @ 18 GB each | up to 7 MIGs @ 16.5 GB each |
| Form Factor | SXM | PCIe dual-slot air-cooled |
| Interconnect | NVLink 4 双向聚合 900 GB/s + PCIe Gen5 双向 128 GB/s | 2- or 4-way NVLink bridge，双向 900 GB/s per GPU + PCIe Gen5 双向 128 GB/s |

口径说明：

- 上面 sparse 数字直接来自 H200 datasheet 官方表（含 2:4 结构化稀疏）；dense = sparse / 2 是 NVIDIA 自身脚注里给的换算。
- NVLink / PCIe 写的 `900 GB/s`、`128 GB/s` 均为**双向聚合**带宽口径；单向减半（NVLink 4 单向 450 GB/s，PCIe Gen5 ×16 单向 64 GB/s）。

H200 的典型价值不是 FP4 reasoning，而是在 Hopper 软件生态下提供更大的 HBM 容量和带宽，适合大模型推理、HPC 和仍依赖 Hopper kernel 路线的部署。

## GB200 / GB300 / DGX B300 规格

Blackwell 和 Blackwell Ultra 的公开规格常以系统为单位发布，所以这里保留官方系统口径，不擅自折算成单 GPU 规格。

### GB200 NVL72

GB200 NVL72 是 rack-scale 系统：36 颗 Grace CPU + 72 颗 Blackwell GPU，组成一个 72-GPU NVLink domain。下表所有数字为**整 rack 聚合口径**，不是单 GPU 规格。

| 项目 | GB200 NVL72 (rack-aggregate) | GB200 Grace Blackwell Superchip (1 CPU + 2 GPU) |
|------|------------------------------|------------------------------------------------|
| 配置 | 36 Grace CPU + 72 Blackwell GPU | 1 Grace CPU + 2 Blackwell GPU |
| NVFP4 Tensor Core (sparse / dense) | 1,440 / 720 PFLOPS | 40 / 20 PFLOPS |
| FP8 / FP6 Tensor Core (sparse / dense) | 720 / 360 PFLOPS | 20 / 10 PFLOPS |
| INT8 Tensor Core (sparse / dense) | 720 / 360 POPS | 20 / 10 POPS |
| FP16 / BF16 Tensor Core (sparse / dense) | 360 / 180 PFLOPS | 10 / 5 PFLOPS |
| TF32 Tensor Core (sparse / dense) | 180 / 90 PFLOPS | 5 / 2.5 PFLOPS |
| FP32 | 5,760 TFLOPS | 160 TFLOPS |
| FP64 / FP64 Tensor Core | 2,880 TFLOPS | 80 TFLOPS |
| GPU Memory / Bandwidth | 13.4 TB HBM3e / 576 TB/s（rack 总聚合带宽） | 372 GB HBM3e / 16 TB/s |
| NVLink Bandwidth (system aggregate) | 130 TB/s（72 GPU 双向聚合） | 3.6 TB/s（单 superchip 双向聚合） |
| CPU Core Count | 2,592 Arm Neoverse V2 cores | 72 Arm Neoverse V2 cores |
| CPU Memory / Bandwidth | 17 TB LPDDR5X / 14 TB/s | up to 480 GB LPDDR5X / up to 512 GB/s |

口径说明：

- 上表 sparse 数字来自官方页面；dense = sparse / 2（NVIDIA 官方换算）。
- `576 TB/s` 是 NVL72 整 rack HBM3e **聚合**带宽：72 GPU × 8 TB/s ≈ 576 TB/s。
- `130 TB/s` 是 NVL72 NVLink Switch 系统**整 rack 聚合**双向带宽，不是单 GPU 端口带宽（单 GPU NVLink 5 双向 1.8 TB/s）。

### GB300 NVL72

GB300 NVL72 属于 Blackwell Ultra，官方定位更偏 reasoning inference。它同样是 72-GPU rack-scale 系统，但 HBM 容量和 FP4 dense 能力更强。下表所有数字为**整 rack 聚合口径**。

| 项目 | GB300 NVL72 (rack-aggregate) |
|------|------------------------------|
| 配置 | 72 NVIDIA Blackwell Ultra GPU + 36 NVIDIA Grace CPU |
| NVLink Bandwidth (system aggregate) | 130 TB/s（72 GPU 双向聚合） |
| Fast Memory | 37 TB |
| GPU Memory / Bandwidth | 20 TB HBM3e / up to 576 TB/s（rack 总聚合带宽） |
| CPU Memory / Bandwidth | 17 TB LPDDR5X / 14 TB/s |
| CPU Core Count | 2,592 Arm Neoverse V2 cores |
| FP4 Tensor Core (sparse / dense) | 1,440 / 1,080 PFLOPS |
| FP8 / FP6 Tensor Core (sparse / dense) | 720 / 360 PFLOPS |
| INT8 Tensor Core (sparse / dense) | 720 / 360 POPS |
| FP16 / BF16 Tensor Core (sparse / dense) | 360 / 180 PFLOPS |
| TF32 Tensor Core (sparse / dense) | 180 / 90 PFLOPS |
| FP32 | 5,760 TFLOPS |
| FP64 / FP64 Tensor Core | 2,880 TFLOPS |

口径校对说明（2026-05-30 修订）：

- 原表 `INT8 24 POPS` / `FP32 6 PFLOPS` / `FP64 100 TFLOPS` 是从其他来源传入的错值，与 GB200 NVL72 同形系统的量级差几十倍，已按官方 GB300 NVL72 datasheet 与 GB200 NVL72 对照修正：INT8 与 FP8 在同一稀疏体系下约为 720 / 360 POPS（rack），FP32 ≈ 5.76 PFLOPS（rack），FP64 Tensor Core ≈ 2,880 TFLOPS（rack）。
- GB300 相对 GB200 的官方宣传点是 **FP4 dense 1.5×**（720→1,080 PFLOPS）、**attention 2×**、**HBM3e 容量 1.5×**；其他精度算力多数与 GB200 NVL72 接近。
- 不同公开来源对 `FP4 sparse` 数值偶有 `1,400 / 1,440` 微差，以最新 [GB300 NVL72 产品页](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) 为准。

官方页面同时强调：GB300 NVL72 相比 Blackwell GPU 提供 1.5x dense FP4 Tensor Core FLOPS 和 2x attention performance，并且 Blackwell Ultra GPU 的 HBM3e 容量提升到前代的 1.5x。

### DGX B300

DGX B300 是 8-GPU 系统，适合和 GB300 NVL72 区分：DGX 是一台系统，NVL72 是 rack-scale 72-GPU 系统。

| 项目 | DGX B300 |
|------|----------|
| GPUs | 8x NVIDIA Blackwell Ultra SXM |
| CPU | Intel Xeon 6776P processors |
| Total GPU Memory | 2.1 TB |
| FP4 Tensor Core | 144 / 108 PFLOPS |
| FP8 Tensor Core | 72 PFLOPS |
| NVLink Switch System | 2x |
| NVLink Bandwidth | 14.4 TB/s aggregate |
| Networking | 8x OSFP ports serving 8x single-port NVIDIA ConnectX-8 VPI |
| Power Consumption | about 14 kW |
| Rack Units | 10U |

DGX B300 的官方脚注同样说明 FP4 是 sparse / dense 口径，FP8 是 sparse 口径，dense 是所列 sparse 数值的一半。

## Tensor Core 代际：硬件原生形状 vs 算力峰值

下表是其他文档引用的**唯一来源**——其他笔记（GEMM 指南、范式文档、课程层）若涉及 Tensor Core 形状或代际算力，都应 wikilink 回到这里，不要在本地重复维护。

![[GPU/Drawings/Tensor Core 代际形状演进.svg]]

可编辑源图：[[GPU/Drawings/Tensor Core 代际形状演进.excalidraw]]

### 硬件原生 MMA 形状（按 PTX ISA / 架构文档）

| 代际 | 代表 GPU | 硬件原生 mma 形状（PTX 视角） | WMMA API 暴露的 tile 形状 |
|------|----------|------------------------------|---------------------------|
| Volta | V100 | `m8n8k4`（FP16） | `m16n16k16`、`m32n8k16`、`m8n32k16` |
| Turing | T4 | `m8n8k16` (INT8)、`m8n8k4`(FP16) | 同 Volta + INT8/INT4 子精度 |
| Ampere | A100 | `m16n8k16` (FP16/BF16)、`m16n8k8` (TF32)、`m16n8k32` (INT8) | 同上 + 第三代 |
| Hopper | H100 / H200 | `wgmma.mma_async` `m64n{8..256, step 8}k16`(FP16/BF16)、`k32`(FP8/INT8)、`k8`(TF32)；同时保留 `mma.sync` ampere 形状 | WMMA 接口可用；新代码推荐走 wgmma |
| Blackwell | B200 / B300 | `tcgen05.mma`：CTA-group 级异步指令族，围绕 Tensor Memory (TMEM) + matrix descriptor + block scaling 组织（FP4/FP6/FP8/FP16/BF16/TF32） | 高级编程从 WMMA 转向 CUTLASS 4.x / cuBLASLt / Triton 模板 |

要点：

- **V100 经常被简写为 `16×16×16`，那是 WMMA API tile 形状**；硬件原生 mma 是 `m8n8k4`。
- Ampere 起原生形状变为 `m16n8k*`，FP16/BF16 k=16、TF32 k=8、INT8 k=32。
- Hopper wgmma 的 N 维不是单一 256，而是 8 的倍数 8~256；A 操作数有 SS（A/B 都在 SMEM）和 RS（A 在寄存器、B 在 SMEM）两种变体。
- Blackwell `tcgen05.mma` 引入 Tensor Memory (TMEM) 与 block-scaled 数据格式（MXFP/NVFP4），缩放因子是 **E8M0 (8-bit exponent-only) 或 FP8(E4M3)**，不是 FP16。
- Blackwell 规格表里 FP4、FP8/FP6、FP16/BF16、TF32、INT8 等算力口径与 A/B 输入、scale、metadata、accumulator/output 的对应关系，见 [[Blackwell 架构新特性与 Kernel 编程#Blackwell 算力口径和输入输出格式]]；本页只维护峰值数字，不重复维护格式表。

### 代际峰值算力（每卡 dense / sparse，统一口径）

| GPU | 精度 | sparse（含 2:4） | dense（实测可达上限） |
|-----|------|------------------|-----------------------|
| V100 SXM2 | FP16 | — | 125 TFLOPS |
| A100 SXM4 80GB | FP16 / BF16 | 312 TFLOPS | 156 TFLOPS |
| A100 SXM4 80GB | TF32 | 156 TFLOPS | 78 TFLOPS |
| A100 SXM4 80GB | INT8 | 624 TOPS | 312 TOPS |
| H100 SXM5 | FP16 / BF16 | 1,979 TFLOPS | 989 TFLOPS |
| H100 SXM5 | TF32 | 989 TFLOPS | 494.5 TFLOPS |
| H100 SXM5 | FP8 | 3,958 TFLOPS | 1,979 TFLOPS |
| H100 SXM5 | INT8 | 3,958 TOPS | 1,979 TOPS |
| H200 SXM | 同 H100 SXM5 计算口径 | 详见上文 H200 表 | 详见上文 H200 表 |
| B200 (1 GPU) | FP4 (NVFP4) | ~20 PFLOPS | ~10 PFLOPS |
| B200 (1 GPU) | FP8 / FP6 | ~10 PFLOPS | ~5 PFLOPS |
| B200 (1 GPU) | FP16 / BF16 | ~5 PFLOPS | ~2.5 PFLOPS |
| B200 (1 GPU) | TF32 | ~2.5 PFLOPS | ~1.25 PFLOPS |
| B300 (1 GPU) | FP4 (NVFP4) | ~20 PFLOPS | ~15 PFLOPS（dense 较 B200 +50%） |
| B300 (1 GPU) | FP8 / FP6 | ~10 PFLOPS | ~5 PFLOPS |

口径校对说明：

- A100 / H100 数字以官方 datasheet 为源；H100 表中 989 TFLOPS 是 FP16 **dense**，1,979 TFLOPS 是 FP16 **sparse**；TF32 dense 是 494.5 TFLOPS（之前文档把 989 TFLOPS 同时写 TF32 和 FP16 是混了精度）。
- B200 / B300 单 GPU 数字由 GB200 NVL72 / GB300 NVL72 **rack-aggregate 反推**（÷72）；不同公开来源略有 ±5% 差异，以 [NVIDIA Blackwell Architecture Technical Overview PDF](https://resources.nvidia.com/en-us-blackwell-architecture) 与产品页为准。
- 任何文档引用 Tensor Core 数字时，**必须同时标注 dense 还是 sparse、单 GPU 还是 rack**。NCU 默认报 dense flops，sparse 路径需单独走 `__nv_sparse_meta` 体系。

## 怎么用规格指导 kernel 判断

规格表不是拿来背的，而是拿来判断瓶颈方向。

| 你看到的规格 | 对 kernel 意味着什么 | 相关文档 |
|--------------|----------------------|----------|
| HBM 容量 | 决定模型权重、KV cache、batch、长上下文能不能放下。 | [[GPU 硬件背景地图]] |
| HBM 带宽 | decode、embedding、reduction、小 batch attention 常常被它限制。 | [[Nsight Compute NCU 分析方法与优化思路]] |
| L2 容量 | 影响权重 tile、KV cache、跨 SM 复用和 memory-bound kernel。 | [[GPU 硬件架构背景与编程范式]] |
| Shared memory 上限 | 影响 CTA tile 大小、stage 数、bank conflict 和 occupancy。 | [[CUDA Shared Memory 与 Bank Conflict]] |
| Tensor Core 峰值 | 只说明理论上限，实际能否接近取决于 dtype、layout、tile、pipeline、occupancy。 | [[CUDA GEMM 矩阵乘法优化指南]] |
| NVLink / NVSwitch | 影响 tensor parallel、expert parallel、KV/activation 交换、all-reduce/all-to-all。 | [[GPU 硬件背景地图]] |
| MIG / TDP | 影响部署隔离、功耗墙、频率和长期稳定吞吐。 | [[CUDA 线程配置与占用率]] |

一个实用读法：

```text
先看模型/算子的瓶颈
  -> memory-bound：优先看 HBM、L2、访存合并、复用
  -> compute-bound：优先看 Tensor Core dtype、tile、pipeline
  -> communication-bound：优先看 NVLink/NVSwitch、NCCL、并行策略
  -> occupancy/scheduling-bound：优先看寄存器、SMEM、block size、cluster
```

## 旧截图

历史截图可以辅助核对，但不要把截图当成最终事实来源。规格以本文下方官方链接和本地 PDF 为准。

- [[Attachment/Image/GPU/Blackwell B200 VS B300/Blackwell B200 VS B300-2026-01-06-00-10-58.png]]
- [[Attachment/Image/GPU/Blackwell B200 VS B300/Blackwell B200 VS B300-2026-01-06-00-11-25.png]]
- [[Attachment/Image/GPU/H200 GPU Spec/H200 GPU Spec-2026-01-06-10-24-41.png]]

## 更新规则

1. 产品规格优先使用 NVIDIA 官方产品页和官方 datasheet。
2. 编译目标、compute capability、低层 kernel 能力优先看 CUDA 文档和 CUTLASS 文档。
3. Blackwell / Blackwell Ultra 的公开资料更新很快，摘录时必须写清 `GPU 单卡 / Superchip / DGX / NVL72` 口径。
4. 如果是从系统总量估算单 GPU 数值，必须明确写“估算”，不要混入官方规格表。

## 参考资料

- [NVIDIA H200 GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [NVIDIA DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [[GPU/References/hpc-datasheet-sc24-h200-datasheet-3002446.pdf|hpc-datasheet-sc24-h200-datasheet-3002446.pdf]]
- [[GPU/References/NVIDIA Blackwell Architecture Technical Overview.pdf|NVIDIA Blackwell Architecture Technical Overview.pdf]]

## 阅读顺序

1. 先读 [[GPU 硬件背景地图]]，建立 GPU 硬件层次。
2. 再用本篇确认产品规格口径。
3. 做 CUDA kernel 时读 [[CUDA 线程配置与占用率]]、[[CUDA Shared Memory 与 Bank Conflict]]。
4. 做 GEMM、attention、Tensor Core 相关优化时读 [[CUDA GEMM 矩阵乘法优化指南]] 和 [[GPU 硬件架构背景与编程范式]]。
