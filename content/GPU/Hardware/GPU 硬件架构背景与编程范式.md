---
aliases:
  - GPU 硬件架构背景
  - Tensor Core 与指令集
  - GPU 编程范式演进
updated: 2026-05-30
tags:
  - gpu-computing
  - gpu-architecture
  - architecture-note
  - tensor-core
---
# GPU 硬件架构背景与编程范式

> 目标：把 GPU 硬件单元、Tensor Core、PTX 指令族，以及 CUDA/CUTLASS/Triton 编程范式的变化串起来。初学者先读 [[GPU 硬件背景地图]] 和 [[GPU 初学者术语表]]，再用本文理解 Ampere/Hopper/Blackwell 的编程范式演进。查具体产品规格时看 [[NVIDIA GPU 架构与规格]]，做算子优化时看 [[CUDA GEMM 矩阵乘法优化指南]]。

## 本文怎么读

这篇是深入层，不是第一篇入门文档：

- 如果你还分不清 SM、warp、CTA、shared memory，先读 [[GPU 硬件背景地图]]。
- 如果你只是要查 H200、GB200、GB300、DGX B300 的规格，看 [[NVIDIA GPU 架构与规格]]。
- 如果你要理解为什么 Hopper 之后 kernel 写法变成 TMA/WGMMA/producer-consumer，看本文。
- 如果你要写或分析 GEMM kernel，看 [[CUDA GEMM 矩阵乘法优化指南]]。

## 一张总图

硬件层次的第一张地图见 [[GPU 硬件背景地图]]。下面这张图进一步把 kernel launch、grid、CTA、SM、warp/thread 和 L2/HBM 放到一起。它不是芯片真实 floorplan，而是帮助建立“kernel launch -> grid -> CTA -> SM -> warp/thread -> L2/HBM”的读图顺序。

![[GPU/Drawings/GPU 硬件层次总览.svg]]

编程范式的演进可以按这条链路理解：SIMT 标量/向量 CUDA Core -> shared memory tiling -> warp-level Tensor Core / WMMA -> PTX MMA / `mma.sync` + `ldmatrix` -> Ampere `cp.async` pipeline -> Hopper TMA + WGMMA + `mbarrier` -> Blackwell `tcgen05.mma` + Tensor Memory + FP4/FP6 -> cuBLASLt、CUTLASS/CuTe、Triton 等库/DSL 优先。

这条线不是说旧方法失效，而是说性能敏感 kernel 的抽象层级越来越高：早期关注“每个线程算什么”，后来关注“warp/warpgroup 如何共同发出矩阵指令”，再往后关注“数据搬运引擎、同步对象、scale 元数据和调度策略如何配合”。复杂的代际对比见下一节的 SVG 图。

## Ampere 到 Blackwell 的演进主线

![[GPU/Drawings/Ampere 到 Blackwell 架构与编程范式演进.svg]]

从 Ampere 到 Hopper 再到 Blackwell，CUDA 的基本模型没有变：还是 grid、block/CTA、warp、thread，还是 global memory、L2、shared memory、register。但高性能 kernel 的“控制重心”在变化：

```text
Ampere:   单 CTA 内，线程/warp 显式组织 copy + MMA 的流水线
Hopper:   TMA + WGMMA 进入主路径，producer/consumer warp specialization 变重要
Blackwell: tcgen05 + 低精度 scale + descriptor/layout，把 GEMM 变成更强的体系化调度问题
```

### 代际对比

| 架构 | 硬件重点 | 编程范式变化 | 优化思路变化 |
|------|----------|--------------|--------------|
| Ampere，SM80/SM86 | 第三代 Tensor Core；TF32、BF16、FP64 Tensor Core；`cp.async` global-to-shared 异步拷贝；split arrive/wait barrier；更大的 L2 和 L2 persistence 控制。 | 生产 GEMM kernel 从“load -> sync -> compute”走向 `cp.async` multi-stage pipeline；常见组合是 `ldmatrix` + `mma.sync` + shared-memory double buffering。 | 重点调 CTA tile、warp tile、SMEM layout、`num_stages`、寄存器和 occupancy；把 HBM 访问合并并尽量提前搬到 shared memory。 |
| Hopper，SM90 | 第四代 Tensor Core；FP8 Transformer Engine；TMA；WGMMA；`mbarrier`；Thread Block Cluster / DSMEM。 | 数据搬运从很多线程的 load 指令升级为 TMA bulk tensor copy；计算从 warp MMA 推进到 warpgroup MMA；producer warp 和 consumer warpgroup 分工成为主流。 | 重点调 TMA descriptor、pipeline phase、warp specialization、cluster size、DSMEM 访问对齐；用 cluster/TMA multicast 减少跨 CTA 重复搬运。 |
| Blackwell，SM100/SM120 | 第五代 Tensor Core 路线；SM100 GEMM 引入 `tcgen05.mma`；FP4/FP6/FP8 narrow precision；block-scaled GEMM；更大 L2、第五代 NVLink。 | MMA 不再只是 A/B 数据和 accumulator，还要处理 scale tensor、descriptor、CTA group、layout/alignment；对 SM100，CUTLASS/CuTe 的 `tcgen05` 抽象比手写 PTX 更适合作为入口。 | 重点从“让 Tensor Core 吃满”扩展为“选对 dtype、scale 粒度、layout、tile shape、dispatch policy”；MoE/grouped GEMM 更依赖 persistent/cooperative 调度。 |

### Ampere：异步拷贝把 GEMM 写法推向软件流水线

Ampere 的关键不是“CUDA 模型改了”，而是它把 global memory 到 shared memory 的 copy 做成了硬件加速的异步路径。`cp.async` 允许 kernel 显式重叠数据搬运和计算，并且可以减少 copy 过程占用的寄存器。

这带来的优化心智模型是：

```text
stage k:     Tensor Core 消费 shared memory 中的 tile k
stage k + 1: cp.async 预取下一块 A/B tile
barrier:     arrive/wait 管理生产者和消费者的交接
```

所以 Ampere GEMM 的核心问题变成：

- `BM/BN/BK` 多大，才能在 SMEM、register、occupancy 之间平衡。
- shared memory layout 是否适配 `ldmatrix`，是否避免 bank conflict。
- `num_stages` 太少会隐藏不了 HBM 延迟，太多会增加寄存器和 SMEM 压力。
- L2 persistence 对反复访问的权重、KV 或中间 tile 是否有帮助。

### Hopper：TMA/WGMMA 把“搬数据”和“算 MMA”拆成角色

Hopper 在 Ampere 异步 copy 之上增加 TMA。TMA 可以把 1D 到 5D tensor 在 global/shared/cluster shared memory 之间做 bulk asynchronous copy，一个线程就能发起较大的 tensor copy，地址计算和搬运从普通 SM 指令中解放出来。

同时，Hopper 的 WGMMA 让一个 warpgroup 参与异步 MMA。于是 kernel 的组织方式从“所有 warp 都差不多”变成更明确的角色分工：

```text
producer warp:
  发起 TMA，把全局内存 tile 搬到 shared memory
  通过 mbarrier 标记数据就绪

consumer warpgroup:
  等待 mbarrier
  发起 WGMMA 消费 shared memory tile
  累加到寄存器 accumulator
```

Thread Block Cluster 进一步把协作边界从一个 CTA 扩到多个 CTA：cluster 内 CTA 可以访问彼此的 shared memory，也就是 DSMEM。它适合处理单 CTA SMEM 放不下、但又不希望完全回退到 global memory 的中间场景。

**Cluster 关键限制**（容易踩坑，详见 [[CUDA CTA 与 Thread Block Cluster 入门]]）：

- **Cluster size**：portable 上限 = 8 CTA；通过 `cudaFuncSetAttribute(cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` 可 opt-in 到 16 CTA。Hopper 与 Blackwell 都遵守这一上限——**Blackwell 没有把 cluster 再扩大**。
- **物理位置**：cluster 内所有 CTA 必须落在**同一个 GPC**（GPU Processing Cluster）的不同 SM 上。
- **DSMEM 路径**：cluster 内 CTA 间的 SMEM 访问走**片内 SM-to-SM fabric**（cluster local network），**不走 NVLink**——NVLink 是 GPU 之间的互联，与 cluster 内通信无关。
- **DSMEM 容量**：每个 CTA 自己的 SMEM 仍是 228KB 上限（Hopper / Blackwell），cluster 可见 SMEM = N × per-CTA SMEM（N = cluster size）。"cluster 可达 SMEM = 912KB" 是 4-CTA cluster 的**集合容量**，单个 CTA 自己的 SMEM 不会变大。

Hopper 开始，优化时要额外问：

- TMA tile 是否足够大，是否避免碎片化小 copy。
- producer/consumer warp 分工是否让 Tensor Core 和 TMA 都持续有活干。
- `mbarrier` 的 phase 和 wait 是否正确，是否引入不必要 stall。
- cluster size 是否真的提高复用，还是降低了活跃 CTA 数。
- DSMEM 访问是否对齐、coalesced，是否可以先搬到 local SMEM 再访问。

### Blackwell：低精度、scale 和调度策略进入 GEMM 主路径

Blackwell 继续保留 Hopper 的 CUDA 编程模型、cluster 和 TMA 思路，但在 SM100 GEMM 上引入 `tcgen05.mma` 指令族。根据 CUTLASS Blackwell 文档，`tcgen05.mma` 支持 legacy 类型，也支持新的 4/6/8-bit floating point 数据类型以及带 scale factor 的 block-scaled GEMM。

这意味着 Blackwell 上的高性能 GEMM 不只是换一个更快的 MMA 指令。kernel 必须同时处理：

- packed narrow precision 数据，例如 FP4(E2M1)、FP6(E3M2 / E2M3)、FP8(E4M3 / E5M2)。
- **scale tensor**：MXFP / NVFP4 块缩放因子格式由 OCP Microscaling spec 定义为 **E8M0（8-bit exponent-only）或 FP8(E4M3)**——**不是 FP16**。块大小：MXFP4 = 32 元素，NVFP4 = 16 元素。

  ![[GPU/Drawings/Block-scaled 数据布局.svg]]

  可编辑源图：[[GPU/Drawings/Block-scaled 数据布局.excalidraw]]

- A/B layout、alignment、T/N 组合和 tensor copy 的额外对齐要求。
- `cta_group::1` / `cta_group::2` 这类参与范围。
- epilogue、dequant、activation、MoE grouped GEMM 的调度问题。

对 LLM 推理尤其重要的是，Blackwell 让“低精度算力”和“参数带宽”同时变成优化对象。FP4/NVFP4 不是把权重压小就完了：scale 的组织、加载、广播、参与 MMA 的方式会直接影响 kernel 形态。

实战上，Blackwell 的优化入口更应该是：

```text
cuBLASLt / CUTLASS / TensorRT-LLM / Triton
  -> 看它选择的 dtype、layout、tile、cluster、dispatch policy
  -> 再用 Nsight 检查 Tensor Core、TMA、L2/HBM、shared memory stall
```

而不是一开始就手写 `tcgen05` PTX。需要按 kernel 编程级别展开 `tcgen05.mma`、TMEM、CLC、`cta_group::2`、NVFP4 block scaling、PDL/GDC 时，看 [[Blackwell 架构新特性与 Kernel 编程]]。

### 优化思路的总体变化

| 问题 | Ampere 思路 | Hopper 思路 | Blackwell 思路 |
|------|-------------|-------------|----------------|
| 隐藏 HBM 延迟 | `cp.async` + 多 stage | TMA + producer/consumer warp | TMA 继续使用，同时考虑 low-bit tensor copy alignment |
| 提高 Tensor Core 利用率 | `mma.sync` tile 层级设计 | WGMMA + warpgroup 调度 | `tcgen05` + CTA group + 合法 tile/layout |
| 提高片上复用 | CTA 内 shared memory tiling | cluster / DSMEM / TMA multicast | cluster + 更复杂的 scale/layout 复用 |
| 处理小矩阵 / MoE | split-K、grouped GEMM、persistent kernel | persistent + TMA warp-specialized | grouped GEMM + cooperative/persistent dispatch policy |
| 控制精度与带宽 | FP16/BF16/TF32/INT8 | FP8 Transformer Engine | FP4/FP6/FP8 + block scale / microscaling |

## GPU 硬件心智模型

### 从外到内看 GPU

新手容易同时遇到 `GPU`、`GPC`、`SM`、`CTA`、`warp`、`thread`、`shared memory`、`L2` 这些词。可以先按三条线拆开：

| 线索 | 从大到小 | 读法 |
|------|----------|------|
| 执行层次 | Grid -> CTA/block -> warp -> thread | kernel launch 后，grid 被拆成很多 CTA，CTA 再拆成 warp 和 thread。 |
| 硬件层次 | GPU -> GPC/SM group -> SM -> CUDA Core / Tensor Core | CTA 被调度到 SM；warp scheduler 在 SM 内发射指令。 |
| 存储层次 | HBM/global -> L2 -> SM 内 L1/shared memory -> register | 越靠近 thread 越快、越小；越靠近 HBM 越大、越慢。 |

这里的 `L1/shared memory` 不是说 shared memory 在 L1 的下一层，而是说二者都属于 SM 内的片上数据通路。按编程模型看，shared memory 是显式管理的 scratchpad；L1 是硬件自动管理的缓存。

一个 CTA/block 通常驻留在一个 SM 上执行。CTA 的优势是 **block 内线程可以共享 shared memory 并同步**；Hopper 之后的 Thread Block Cluster 进一步允许多个 CTA 形成 cluster，通过 Distributed Shared Memory 访问彼此的 shared memory，但 CTA 本身仍不跨多个 SM 执行。

### SM 是基本执行单元

GPU 由多个 SM（Streaming Multiprocessor）组成。一个 CUDA block 会被调度到某个 SM 上执行，同一个 block 内线程共享该 SM 的寄存器，以及 carve out 给 shared memory / L1 的片上资源，并通过 `__syncthreads()`、barrier 或更高级同步对象协作。

![[GPU/Drawings/SM 内部执行与存储路径.svg]]

可以把一个现代 NVIDIA SM 粗略看成：

- Warp scheduler：选择 ready warp 发射指令，隐藏访存和依赖延迟。
- CUDA Core：执行 FP32/INT32 等标量或向量化的 SIMT 指令。
- Tensor Core：执行矩阵乘加 MMA 指令，主要服务 GEMM、attention、convolution、MoE expert GEMM。
- LD/ST 单元：负责 global/shared/local 等内存访问。
- Register file：线程私有状态，速度快但容量有限，寄存器压力会影响 occupancy。
- Shared Memory / L1：SM 内片上资源。Shared Memory 是显式复用数据的 scratchpad，L1 是自动缓存路径；容量 carveout 会影响两者的平衡，bank conflict 主要是 SMEM 访问问题。
- L2 + HBM：跨 SM 共享缓存和全局显存，是 decode、KV cache、embedding、all-to-all 等场景的关键瓶颈。

### 性能瓶颈通常落在三类资源

| 资源 | 典型瓶颈 | 典型场景 |
|------|----------|----------|
| Tensor Core / CUDA Core | 算力未吃满、tile 太小、指令发射不足 | 大 batch GEMM、prefill attention、MLP |
| HBM / L2 | 带宽不足、重读、cache 命中差 | decode KV cache、embedding、低 batch attention |
| 片上资源 | shared memory bank conflict、寄存器溢出、同步开销 | 手写 GEMM、FlashAttention、Triton kernel |

对大模型推理来说，prefill 常常更接近 compute-bound，decode 常常更接近 bandwidth/KV-bound。不能只拿 Tensor Core 峰值 TFLOPS 推导端到端吞吐。

## Tensor Core 是什么

Tensor Core 是矩阵乘加专用硬件。它处理的是形如：

```text
D = A * B + C
```

的 MMA（Matrix Multiply-Accumulate）操作。它不是“自动让所有矩阵运算变快”的魔法开关，而需要满足：

- 数据类型受硬件代际限制，例如 FP16、BF16、TF32、FP8、INT8、FP4、FP6 等。
- 矩阵 tile shape 受指令限制，例如 `m16n8k16`、`m64nNk16` 等。
- 数据布局和对齐必须符合指令或库的要求。
- 多个线程需要协作持有 fragment 或 descriptor；越新的架构越强调 warpgroup、TMA、Tensor Memory 等协作机制。
- 累加精度、scale 因子和输出类型会影响数值稳定性。

从使用层级看，可以这样理解：

| 层级 | 代表 | 适合谁 | 特点 |
|------|------|--------|------|
| 框架/库 | PyTorch、cuBLAS、cuBLASLt、cuDNN、TensorRT-LLM | 业务和模型工程 | 最稳，自动选 kernel，但调参空间有限 |
| 模板库/DSL | CUTLASS/CuTe、Triton | kernel 优化学习和生产定制 | 能表达 tile、pipeline、layout、schedule |
| CUDA C++ API | `nvcuda::wmma` | 学习 Tensor Core 基本模型 | warp 级 fragment，抽象较高但灵活性有限 |
| PTX/SASS | `mma.sync`、`wgmma.mma_async`、`tcgen05.mma` | 极致优化/库作者 | 控制力强，版本和架构绑定明显 |

> **Tensor Core 硬件原生 mma 形状（如 V100 `m8n8k4`、Ampere `m16n8k16` 等）和各代际 dense/sparse 峰值算力，统一以 [[NVIDIA GPU 架构与规格]] §"Tensor Core 代际：硬件原生形状 vs 算力峰值" 为唯一来源**——本文不再重复维护这两张表。常见误区：V100 经常被简写为 `16×16×16`，那是 WMMA API tile 形状，硬件原生 mma 是 `m8n8k4`。

## 指令与能力演进

| 架构 | 重点硬件/能力 | 代表指令或 API | 编程范式变化 |
|------|---------------|----------------|--------------|
| Volta / Turing | Tensor Core 进入 CUDA 编程模型；Turing 增强 INT8/INT4/二值推理能力。 | `nvcuda::wmma`、`wmma.mma` | 从 SIMT 标量 FMA 转向 warp 协作的矩阵 fragment。 |
| Ampere | 第三代 Tensor Core；TF32、BF16、FP64 Tensor Core；异步 global-to-shared copy。 | `mma.sync`、`ldmatrix`、`cp.async`、split barrier | GEMM kernel 开始强调 `ldmatrix` + `mma.sync` + shared-memory double buffering。 |
| Hopper | 第四代 Tensor Core；FP8 Transformer Engine；TMA；Thread Block Cluster / DSMEM；WGMMA。 | `wgmma.mma_async`、`cp.async.bulk.tensor`、`mbarrier` | 从 warp 级 MMA 推进到 warpgroup 级异步 MMA，生产者/消费者 warp specialization 变重要。 |
| Blackwell | 第五代 Tensor Core；FP4/FP6 和 microscaling；Tensor Memory；更强的 cluster/NVLink 系统能力。 | `tcgen05.mma`、`tcgen05.*`、TMEM descriptor | 低精度 GEMM 变成“数据 + scale + descriptor + Tensor Memory”的协同问题，手写 PTX 更不适合作为新手起点。 |

### 计算指令族

- `wmma.*`：CUDA C++ 暴露的 warp matrix API。一个 warp 的所有线程协作加载 fragment、执行 `mma_sync`、写回结果。
- `mma.sync`：PTX 级 warp MMA。常配合 `ldmatrix` 从 shared memory 加载矩阵片段，再发 MMA 指令。
- `wgmma.mma_async`：Hopper 开始的 warpgroup 异步 MMA。一个 warpgroup 是 4 个 warp / 128 线程，适合更大 tile 和异步流水线。要点：
  - **形状**：`m` 固定为 64；`n ∈ {8, 16, 24, ..., 256}`（8 的倍数）；`k` 由精度决定：FP16/BF16 = 16，FP8/INT8 = 32，TF32 = 8。**只写 `m64n256k16` 是片面的**——常用 tile 还有 `m64n128k16`、`m64n64k16` 等。
  - **操作数来源**：有两种变体：**SS**（A 和 B 都来自 shared memory，通过 matrix descriptor 描述地址 + swizzle 模式）和 **RS**（A 来自寄存器，B 来自 shared memory）。**不存在"A/B 必须全部来自 SMEM"的限制**——RS 变体允许把 A fragment 留在寄存器中复用。
  - **scale 参数**：标量参数顺序为 `scale-d, scale-a, scale-b, trans-a(仅 SS), trans-b(仅 SS)`，控制累加缩放与转置；具体见 PTX ISA `wgmma.mma_async` 章节。

  ![[GPU/Drawings/WGMMA SS 与 RS 操作数路径.svg]]

  可编辑源图：[[GPU/Drawings/WGMMA SS 与 RS 操作数路径.excalidraw]]

- `tcgen05.mma`：Blackwell 第五代 Tensor Core 的 PTX 指令族。官方 PTX 文档将其建模为异步指令，围绕 **Tensor Memory (TMEM, 每 SM 256KB)**、matrix descriptor、CTA group (`cta_group::1` 单 CTA / `cta_group::2` 双 CTA 协作) 和 block scaling 组织。配套的 `tcgen05.ld` / `tcgen05.st` 在 TMEM 和寄存器之间搬数据。具体形状/语义随 CUDA Toolkit 12.8/13.x 仍在演进，引用时建议标明 PTX ISA 版本。

### 数据搬运与同步指令族

- `ldmatrix`：把 shared memory 中的矩阵片段按 Tensor Core 需要的方式加载到寄存器 fragment。
- `cp.async`（Ampere，warp 级）：将 global memory 异步拷贝到 shared memory，**每个线程发起 4/8/16 字节级别的小拷贝**，由 warp 内若干线程协作覆盖一段 tile；与计算流水线配合需要 `cp.async.commit_group` + `cp.async.wait_group`。
- `cp.async.bulk` / `cp.async.bulk.tensor`（Hopper TMA，CTA 级）：在 `cp.async` 之上的批量异步搬运。**只需一个线程发起**即可让 TMA 引擎把 1D~5D tensor 在 global / shared / cluster shared memory 之间整块搬运；地址计算和访存协调全部由 TMA 硬件完成。`cp.async.bulk.tensor.{1d,2d,3d,4d,5d}` 是按 tensor 维数分的指令变种，配合 `mbarrier` 完成事件通知；cluster multicast 变体可在 cluster 内多个 CTA 间一次广播 tile。
- `mbarrier`（Hopper+）：异步事务同步对象，支持 split arrive/wait 语义。典型用法：`mbarrier.init.shared` 初始化 → producer 完成搬运后 `mbarrier.arrive.expect_tx`（带事务字节数）→ consumer 用 `mbarrier.try_wait`/`wait` 阻塞等待。WGMMA 和 TMA 事务都依赖 mbarrier 做信号同步。
- Thread Block Cluster / DSMEM：多个 CTA 组成 cluster，可以访问彼此的 shared memory。详细约束见上文 Hopper 段与 [[CUDA CTA 与 Thread Block Cluster 入门]]。

## 编程范式的变化

### 1. 线程级 SIMT

最早的 CUDA 直觉是：一个线程负责一个元素，所有线程以 SIMT 方式执行同一段代码。它适合 vector add、简单 elementwise、基础 reduction，但对 GEMM 这类高复用算子远远不够。

学习重点：

- global memory coalescing
- warp divergence
- block/grid 配置
- occupancy 和寄存器限制

对应文档：[[CUDA 编程基础]]、[[CUDA 线程配置与占用率]]。

### 2. CTA 级 shared memory tiling

GEMM 和 convolution 的第一层优化是把 A/B tile 搬到 shared memory，让一个 tile 被 block 内多个线程复用。这个阶段的核心不是 Tensor Core，而是 IO-aware：少读 HBM，多用片上 SRAM。

学习重点：

- tile size：BM / BN / BK
- shared memory bank conflict
- vectorized load / store
- register tiling

对应文档：[[CUDA Shared Memory 与 Bank Conflict]]、[[CUDA Kernel 示例：矩阵乘法]]。

### 3. Warp 级 Tensor Core

WMMA 把“一个线程算一个元素”的心智模型改成“一个 warp 共同算一个矩阵 fragment”。每个 lane 持有一部分 fragment，fragment 的内部布局是架构相关的，不能随便假设。

学习重点：

- `wmma::fragment`
- `load_matrix_sync`
- `mma_sync`
- `store_matrix_sync`
- fragment 不要跨不同架构编译单元传递

这个阶段适合理解 Tensor Core，但生产 GEMM 通常不会直接手写 WMMA。

### 4. PTX MMA 与流水线 GEMM

更高性能的 GEMM 通常会越过 WMMA，使用 `mma.sync` 和 `ldmatrix`。此时 kernel 的重点变成：

- shared memory layout 要适配 `ldmatrix`
- 多级 tiling：CTA tile、warp tile、MMA tile
- double buffering 或 multi-stage pipeline
- 寄存器、shared memory、occupancy 之间取平衡

这也是 CUTLASS/CuTe 最值得学习的地方：它把复杂的 tile/layout/pipeline 变成可组合模板。

### 5. Ampere 异步拷贝流水线

Ampere 的 `cp.async` 让 global-to-shared copy 可以和 Tensor Core 计算重叠。kernel 不再是“搬完一块、同步、算一块”，而是维护多 stage pipeline：

```text
stage 0: compute tile k
stage 1: async copy tile k+1
stage 2: wait / commit / swap buffer
```

这解释了为什么现代 GEMM 里常见 `num_stages`、software pipeline、producer-consumer barrier 等概念。

### 6. Hopper TMA + WGMMA + warp specialization

Hopper 的关键变化是 TMA 和 WGMMA：

- TMA 把多维 tensor 搬运交给专门引擎，一个线程可以发起较大的 tensor copy。
- WGMMA 让一个 warpgroup 发起更大粒度的异步 MMA。
- `mbarrier` 负责让 producer warp 和 consumer warp 正确交接数据。
- Thread Block Cluster / DSMEM 让 CTA 之间共享片上数据成为可能。

![[GPU/Drawings/Thread Block Cluster 与 DSMEM.svg]]

编程范式变成：

```text
producer warp: 发起 TMA，把全局内存 tile 搬到 shared memory
consumer warps: 等待 mbarrier，然后用 WGMMA 消费 tile
pipeline: 多个 stage 重叠 TMA、MMA、epilogue
```

FlashAttention-3、CUTLASS 3.x/CuTe 的 Hopper kernel 都围绕这套心智模型展开。

这里要特别小心一个表述：**不是 CTA 自己跨 SM 使用 shared memory**。更准确地说，CTA 仍然驻留在一个 SM 上；cluster 把多个 CTA 协同调度到一组 SM，并提供 cluster 级同步和 DSMEM 地址空间，让一个 CTA 能访问同一 cluster 内其他 CTA 的 shared memory。

### 7. Blackwell Tensor Memory + tcgen05 + block scaling

Blackwell 进一步把第五代 Tensor Core、Tensor Memory 和低精度 scale 体系绑定在一起。需要注意两点：

1. 官方 PTX 中的主线叫 `tcgen05.mma` / `tcgen05.*`，很多口语资料会把它概括成“新一代 MMA/UMMA”，但写文档时最好以官方指令名为准。
2. FP4/FP6 不是只把数据压到更小，它通常伴随 block scale / microscaling，kernel 必须同时处理 packed data、scale tensor、descriptor、accumulator 存放位置和 epilogue。

对新手来说，Blackwell 低层指令不适合直接手写；更好的路线是先读 CUTLASS Blackwell GEMM 文档，理解 `tcgen05.mma` 支持的数据类型、block-scaled GEMM 和调度策略，再回到 PTX 文档查细节。

## 对 LLM 推理的影响

### Prefill 与大 GEMM

Prefill 处理 prompt，序列长度较大，QKV projection、MLP、attention score 计算更容易形成大 GEMM 或大 tile attention。这个阶段更有机会吃满 Tensor Core，因此：

- FP16/BF16/FP8/FP4 格式是否被 kernel 支持很关键。
- tile shape 和 batch/sequence shape 会影响 cuBLASLt、FlashAttention、Triton 的 kernel 选择。
- Tensor Core 峰值只是上限，真正要看 kernel timeline 和 tensor pipe utilization。

### Decode 与 KV cache

Decode 每步只生成少量 token，经常是小 batch、小矩阵和大量 KV cache 读取。此时瓶颈可能从 Tensor Core 转向 HBM/L2/NVLink：

- GQA/MQA/MLA 的价值在于减少 KV cache 存储和读取。
- KV cache 量化可以降带宽，但会引入 dequant 或低精度 attention kernel 的复杂度。
- continuous batching、paged attention、prefix cache、PD 分离是系统级优化，不能只靠一个 GEMM kernel 解决。

### 低精度与 scale 体系

FP8、FP4、FP6 对推理的意义不只是“显存更小”：

- 权重和激活是否都低精度，决定 GEMM 输入格式。
- scale 是 per-tensor、per-channel、per-block 还是 per-token，会决定 scale tensor 的形状和访问模式。
- Blackwell 的 FP4/FP6 和 microscaling 把 scale 元数据进一步纳入 Tensor Core 路径。
- 对 MoE 来说，grouped GEMM + FP4/FP8 + expert dispatch 会同时考验 Tensor Core、HBM 和通信。

相关文档：[[FP4 精度]]、[[NVFP4 量化与反量化]]、[[MoE EP 与非 EP 计算过程对比]]。

## 推荐学习顺序

1. 先看 [[CUDA 编程基础]]：理解 kernel、thread/block/grid、warp。
2. 再看 [[CUDA 线程配置与占用率]]：知道 occupancy 只是资源约束，不是最终目标。
3. 继续看 [[CUDA Shared Memory 与 Bank Conflict]]：理解片上 SRAM 复用和 bank conflict。
4. 跟着 [[CUDA GEMM 矩阵乘法优化指南]]：从 naive GEMM 走到 Tensor Core。
5. 回到本文：把 Tensor Core、PTX 指令、TMA/WGMMA/tcgen05 的代际关系串起来。
6. 进入 LLM 侧：[[Attention]]、[[MHA 与 GQA 实现]]、[[LLM 推理优化]]、[[LLM 量化与低精度总览]]。
7. 最后读 CUTLASS/CuTe 和 Triton 代码：用实际 kernel 验证这些概念。

## 后续建议新增的文档

| 文档 | 内容建议 |
|------|----------|
| `Tensor Core 指令速查` | 用一页表格列 `wmma`、`mma.sync`、`wgmma.mma_async`、`tcgen05.mma` 的层级、线程协作粒度、输入来源、同步方式。 |
| `Hopper TMA WGMMA 编程模型` | 专门解释 producer/consumer warp specialization、mbarrier、TMA descriptor、FlashAttention-3 的执行图。 |
| `Blackwell tcgen05 与 Tensor Memory` | 专门整理 TMEM、CTA group、block scaled GEMM、FP4/FP6/NVFP4 与 CUTLASS Blackwell kernel。 |
| `Nsight Compute GPU 指标速查` | 把 tensor pipe、SM active、warp stall、HBM bandwidth、L2 hit、shared bank conflict 和 LLM 推理瓶颈对应起来。 |
| `CUTLASS CuTe GEMM 阅读笔记` | 从一个最小 GEMM template 开始，解释 tile、layout、copy atom、mma atom、pipeline。 |

## 可靠性口径

- 硬件规格优先看 [[NVIDIA GPU 架构与规格]] 和官方产品页，不把系统级、单卡、sparse/dense 口径混在一起。
- 指令语义以 NVIDIA CUDA Programming Guide、PTX ISA 和架构 tuning guide 为准。
- Blackwell 相关资料变化较快，写具体吞吐、shape、指令格式时要标注 CUDA/PTX/CUTLASS 版本。
- 本文日期为 2026-05-19；若 CUDA Toolkit 或 CUTLASS 版本升级，需重新核对 `tcgen05.*`、SM100/SM120/SM12x 的细节。

## 参考资料

- [NVIDIA CUDA Programming Guide - Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [NVIDIA CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-in/data-center/tensor-cores/)
- [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
