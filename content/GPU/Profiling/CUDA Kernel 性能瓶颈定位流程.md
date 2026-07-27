---
aliases:
  - 单个 CUDA Kernel 瓶颈定位
  - Kernel 性能瓶颈定位流程
  - CUDA Kernel Profiling Workflow
updated: 2026-05-30
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
---
# CUDA Kernel 性能瓶颈定位流程

这篇是单个 CUDA kernel 的性能诊断流程。它依赖 [[Nsight Compute NCU 分析方法与优化思路]] 的指标解释，也可以配合 [[NCU_ANALYSIS]] 的 naive/tiled matmul 案例练习。

相关笔记：

- [[Nsight Compute NCU 分析方法与优化思路]]
- [[复杂 Python 进程选择性 NCU Profiling 操作手册]]
- [[NCU_ANALYSIS]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件背景地图]]

## 先建立一个心智模型

分析单个 kernel 时，不要问“哪个指标最大”，而是问：

```text
这个 kernel 想让 GPU 做什么？
GPU 实际忙在什么地方？
忙的地方是不是我期望的地方？
如果不是，代码里哪一行造成了这个现象？
```

NCU 的指标不是结论，而是证据。一个可靠判断通常需要串起这条链：

```text
时间占比
-> launch 形态
-> Speed Of Light 大方向
-> occupancy / waves 是否喂饱
-> memory / compute 具体资源
-> scheduler / warp stall 原因
-> source / SASS 热点行
-> 修改一个点并复测
```

![[GPU/Drawings/NCU Kernel 分析闭环.svg]]

## 一页诊断流程

### 0. 先确认它值得优化

单个 kernel 很慢，不一定是端到端瓶颈。先用 `nsys`、框架 profiler 或业务 trace 确认：

- 这个 kernel 在端到端耗时里占比高。
- 它不是偶发慢，而是在稳定输入下反复出现。
- 它不是被 CPU 调度、launch gap、NCCL 等待、跨进程同步拖慢。

如果问题是“很多小 kernel 之间有空洞”，优先看 [[Nsight Compute NCU 分析方法与优化思路]] 里提到的端到端时间线，而不是先钻单个 kernel。

### 1. 固定实验条件

记录这些信息，否则优化结果容易不可复现：

| 信息 | 为什么重要 |
|------|------------|
| GPU 型号、SM 数、CUDA/driver/NCU 版本 | 峰值、指标口径、cache/SMEM 容量都依赖架构。 |
| shape、dtype、batch、seq length | kernel 瓶颈常随规模变化。 |
| baseline kernel time | 后面每次只和同一条件比。 |
| warmup 和重复次数 | 排除首次 JIT、lazy init、冷 cache。 |
| 正确性误差 | 性能优化不能破坏结果。 |

建议命令：

```bash
ncu --set full \
  --kernel-name regex:".*target_kernel.*" \
  --launch-skip 10 \
  --launch-count 1 \
  -f -o ncu_target \
  ./app args...
```

日常迭代可以只采关键 section：

```bash
ncu \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --kernel-name regex:".*target_kernel.*" \
  --launch-count 1 \
  -f -o ncu_target \
  ./app args...
```

### 2. 看 LaunchStats：kernel 是否被“喂饱”

先看这些字段：

| 指标 | 主要问题 |
|------|----------|
| Grid Size | 总 block 数够不够覆盖所有 SM。 |
| Block Size | 每个 block 有多少线程/warp。 |
| Registers Per Thread | 是否导致每 SM 只能驻留很少 block。 |
| Shared Memory Per Block | 是否限制 resident block 数。 |
| Waves Per SM | 总共有多少波 block，尾波是否很瘦。 |

核心公式：

```text
warps_per_block = ceil(threads_per_block / 32)

resident_blocks_per_sm
  = min(寄存器限制, shared memory 限制, warp 限制, block 槽位限制, barrier/cluster 限制)

waves_per_sm = grid_blocks / (SM 数 * resident_blocks_per_sm)
```

读法：

- `Waves Per SM` 很小：总 block 数太少，SM 没活干。
- `Waves Per SM` 的小数部分很小，例如 `13.05`：尾波很瘦，最后阶段很多 SM 空闲。
- `Block Limit Registers = 2` 不一定坏。如果每个 block 有 32 个 warp，两个 block 已经是 64 warps/SM，occupancy 仍可能是 100%。
- block 很大不一定好。1024 threads/block 会减少调度粒度，tail effect 和资源占用都可能更明显。

### 3. 看 Speed Of Light：先判断大方向

Speed Of Light，简称 SOL，是某类硬件资源达到峰值能力的百分比。第一眼用它判断大方向：

| 现象 | 初步判断 | 下一步 |
|------|----------|--------|
| Compute/SM throughput 高，Memory/DRAM 不高 | 更像计算或指令管线受限 | 看 ComputeWorkloadAnalysis、tensor pipe、SASS 指令。 |
| DRAM throughput 高，SM throughput 上不去 | 更像 HBM 带宽受限 | 看 global load/store 字节、coalescing、L2 hit、Roofline。 |
| Memory throughput 高，DRAM throughput 低 | 不一定是 HBM，可能是 L1TEX/shared/L2/访存管线 | 展开 memory breakdown，看 L1TEX、shared、sector/request、bank conflict。 |
| Compute 和 Memory 都低 | 更像没喂饱、延迟隐藏不足、同步/分支/依赖 | 看 occupancy、eligible warps、stall reason、waves。 |
| Compute 和 Memory 都高 | 可能是 balanced，也可能双重压力 | 看 duration 是否已经接近目标，再找最明显的 source hotspot。 |

最重要的一句：**Memory Throughput 高不等于 HBM 打满；判断 HBM 要看 DRAM Throughput 和真实 DRAM bytes。**

### 4. 看 Occupancy：warp 够不够隐藏延迟

Occupancy 是：

```text
active warps / max resident warps
```

它回答的是“SM 上有多少 warp 可以轮换执行”，不是“性能百分比”。

| 现象 | 可能含义 | 优化方向 |
|------|----------|----------|
| Theoretical Occupancy 低 | 资源限制了能驻留的 warp | 降寄存器、降 SMEM、调 block size、拆 kernel。 |
| Theoretical 高但 Achieved 低 | 理论能放，但运行中 warp 不活跃 | 看 tail effect、同步、分支、负载不均。 |
| Occupancy 已经很高但性能差 | 问题不在 warp 数量 | 看 issue active、stall、memory/compute pipeline。 |

不要盲目追 100%。GEMM/Tensor Core kernel 经常用更多寄存器和 shared memory 换数据复用，occupancy 低一些也可能更快。

### 5. 看 MemoryWorkloadAnalysis：到底是哪级内存卡住

先分清访问路径：

| 代码访问 | 常见路径 | 应该看什么 |
|----------|----------|------------|
| global/local load-store | L1TEX -> L2 -> DRAM | L1/L2 hit、sectors/request、DRAM throughput、long scoreboard。 |
| shared memory | SM 内 shared data bank | bank conflict、shared load/store、barrier stall。 |
| texture/surface | TEX -> L1TEX -> L2 -> DRAM | TEX request、sector、hit rate。 |

常见诊断：

| 现象 | 判断 | 优化方向 |
|------|------|----------|
| DRAM throughput 高，L2 hit 低 | HBM 带宽或容量局部性问题 | 减少读写字节、tiling、复用 shared/register、融合算子、低精度。 |
| sectors/request 高 | 访问不合并或很碎 | 调整数据布局、让 warp 连续访问、向量化 load/store。 |
| L1TEX 高但 DRAM 低 | 片上访存路径或请求数量压力 | 减少 shared/global load 指令、提高每次 load 的计算复用、检查 bank conflict。 |
| shared bank conflict 高 | 多个 lane 访问同一 bank | padding、swizzle、改变 tile layout、检查 `ldmatrix` 友好布局。 |
| local memory load/store 高 | 寄存器溢出到 local memory | 减少 live range、减小 unroll/tile、拆函数/拆 kernel。 |

> L1TEX / SMEM / L2 / DRAM 的具体 throughput 峰值（用于"我达到了几成"的口径）以 [[NVIDIA GPU 架构与规格]] 为唯一来源；不同代际差异较大（A100 vs H100 vs B200），不要把任何一组数字写死在本文中。NCU 默认报 dense flops，sparse 路径需单独读取。

### 6. 看 ComputeWorkloadAnalysis：计算单元是否用对

如果这是 GEMM、attention、conv 这类计算密集 kernel，重点看：

| 问题 | 证据 | 优化方向 |
|------|------|----------|
| 没走 Tensor Core | SASS 里没有 `MMA`/`WGMMA`/`HMMA`，tensor pipe 低 | 调 dtype、layout、对齐、tile，优先用 cuBLAS/CUTLASS/Triton 正确路径。 |
| Tensor Core 利用低 | tensor pipe 低，memory 或 stall 高 | pipeline global->shared->register，扩大 tile，提高数据复用。 |
| 非矩阵开销太高 | Source 里 epilogue、mask、rescale、branch 热 | 融合 epilogue、减少中间写回、简化索引和边界处理。 |
| FP/INT/SFU 管线拥塞 | 对应 pipe throttle 或指令占比高 | 减少 div/mod/sqrt/exp，预计算，使用近似或查表。 |

对初学者来说，计算密集 kernel 的第一条规则是：**先确认代码真的走到了你以为的硬件路径。** 例如 FP16/BF16 GEMM 如果没有走 Tensor Core，后面调 block size 通常不是主矛盾。

### 7. 看 Scheduler / Warp State：为什么发不出指令

SchedulerStats 先看 `Issue Active` 和 eligible warps。只有当 scheduler 发射不饱时，stall reason 才是重点。

| Stall 现象 | 常见原因 | 优化方向 |
|------------|----------|----------|
| Long Scoreboard 高 | 等 global/local memory 数据 | 提前加载、提高 locality、增加独立计算、避免 spill、提高 occupancy。 |
| Barrier 高 | `__syncthreads()` 频繁或 block 内负载不均 | 减少 block-wide barrier，改 warp-level primitive，双缓冲。 |
| MIO / LSU throttle 高 | load/store/shared memory 管线压力 | 减少访存指令、向量化、提高 load 后复用、优化 shared layout。 |
| Math pipe throttle 高 | 某计算管线拥塞 | 换指令路径、减少特殊函数、提高指令混合。 |
| Branch resolving / divergence 高 | warp 内控制流分裂 | 数据分组、拆 kernel、减少 warp 内分支差异。 |
| Wait / dependency 高 | 指令依赖链长 | unroll、重排计算、把 load 和 compute 间隔拉开。 |

### 8. 回到 Source / SASS：找要改的代码行

最后一定要落到源码或 SASS：

- 哪些行贡献了最多 global load/store。
- 哪些行贡献了 shared load/store 或 bank conflict。
- 是否出现 local memory，说明 register spill。
- barrier 是否集中在 tile 循环内。
- SASS 是否出现预期的 `LDG`、`LDS`、`STG`、`MMA`、`WGMMA`、`LDMatrix`、`CP.ASYNC` 或 TMA。

如果无法把指标落到源码，至少要落到指令类型或模板参数，否则优化动作容易变成猜。

## 瓶颈到优化方向速查

| 诊断结论 | 典型证据 | 首选优化 |
|----------|----------|----------|
| Launch-bound / grid 太小 | kernel 很短、grid blocks 少、waves 小 | 合并小 kernel、增大 grid、persistent kernel、减少 launch 次数。 |
| HBM bandwidth-bound | DRAM throughput 高、Roofline 靠近带宽斜线 | 少搬字节、融合算子、低精度/量化、提高 L2/shared/register 复用。 |
| L2/locality 问题 | L2 throughput 高、L2 hit 低 | blocking、重排数据、改善访问顺序、让相邻 CTA 复用相邻数据。 |
| L1TEX/on-chip path 压力 | Memory 高、L1TEX 高、DRAM 低 | 减少请求数、向量化、减少 shared/global load、提高每次 load 的计算量。 |
| Shared memory bank conflict | bank conflict 指标高 | padding、swizzle、改 tile layout、检查 lane 到 bank 映射。 |
| Occupancy/resource-bound | Block Limit Registers/SMEM 低，active warp 不足 | 降寄存器 live range、调 block size、减 tile、拆 kernel、谨慎用 `launch_bounds`。 |
| Latency-bound | SOL 都不高，eligible warp 少，long scoreboard/wait 高 | 增加独立工作、预取、提高 occupancy、缩短依赖链。 |
| Compute/Tensor-bound | compute/tensor pipe 高，Roofline 接近平顶 | 优化 tile 和 pipeline，减少非 MMA 开销，接受合理低 occupancy。 |
| Instruction-bound | div/mod/SFU/地址计算热 | 简化索引、预计算、位运算替代、减少特殊函数。 |
| Branch/divergence-bound | branch stall 或 warp divergence 高 | 数据重排、按分支拆 kernel、让 warp 内处理同类元素。 |

## 不同 kernel 的常见第一怀疑点

| Kernel 类型 | 常见瓶颈 | 初学者优先看 |
|-------------|----------|--------------|
| Elementwise / activation | HBM 带宽、小 kernel launch | DRAM bytes、coalescing、向量化、fusion。 |
| Reduction / scan | shared memory、同步、warp 间规约 | bank conflict、barrier、warp shuffle、block 内负载。 |
| GEMM / MLP | Tensor Core、tile、global/shared/register pipeline | Tensor pipe、MMA/WGMMA、tile shape、occupancy 不要盲追。 |
| Softmax / attention | shared/register、exp/sum、mask、KV 读取 | SFU、barrier、DRAM/L2、online softmax、FlashAttention 分块。 |
| Norm | memory-bound + reduction | 向量化、一次读写、融合后续 scale/activation。 |
| Decode attention | KV cache 读取、L2/HBM、访存延迟 | DRAM/L2、long scoreboard、KV layout、GQA/MQA、KV 量化。 |
| MoE | 小 GEMM、tail wave、load imbalance | waves、tensor pipe、expert batching、grouped GEMM。 |

## 一个完整判断示例

来自 [[NCU_ANALYSIS]] 的 naive/tiled matmul：

```text
结果：Tiled 更快
-> SOL：Memory Throughput 和 Compute Throughput 都高
-> DRAM：极低，说明不是 HBM 带宽瓶颈
-> L1TEX：高，说明压力更多在片上访存路径 / shared data path
-> Occupancy：接近 100%，继续追 occupancy 没意义
-> Scheduler：看 issue active、barrier、shared memory 访问节奏
-> Source：检查 shared tile、同步、bank conflict、thread tiling
```

因此优化方向不是“继续提高 occupancy”，也不是“减少 DRAM 带宽”，而是：

1. 先用 bank conflict 指标验证 shared memory 是否冲突。
2. 用 1D/2D thread tiling 增加每次 shared load 服务的 FMA 数。
3. 尝试 double buffering / async copy，把 load tile 和 compute 重叠。
4. 与 cuBLAS/CUTLASS baseline 对齐，确认自写 kernel 差距来自 tile 设计还是指令路径。

## 记录模板

```markdown
## Kernel Profiling Case: <kernel name>

- 目标：为什么要优化它，端到端占比是多少
- 环境：GPU / CUDA / driver / NCU / clock
- 输入：shape / dtype / batch / seq / layout
- Baseline：kernel time / throughput / 正确性
- NCU 命令：
- Launch:
  - grid / block / registers / shared memory / waves
- SOL:
  - compute / memory / L1TEX / L2 / DRAM
- Occupancy:
  - theoretical / achieved / block limit
- Memory:
  - DRAM bytes / L2 hit / sectors/request / bank conflict
- Compute:
  - tensor pipe / fp pipe / instruction mix
- Scheduler:
  - issue active / eligible warps / top stall
- Source/SASS:
  - 热点行或热点指令
- 判断：
  - launch-bound / memory-bound / compute-bound / latency-bound / balanced
- 修改：
- 复测：
- 结论：
```

## 常见误判

- 只看 `Memory Throughput` 高就说 HBM 瓶颈。应该同时看 `DRAM Throughput`、L2、L1TEX 和 breakdown。
- 看到 `Occupancy` 低就一定想办法提高。GEMM 常用寄存器和 shared memory 换复用，低 occupancy 未必坏。
- 看到 `Block Limit Registers = 2` 就认为寄存器是严重问题。要同时看每 block warp 数和最终 occupancy。
- 看到 stall 排名就直接改代码。先确认 `Issue Active` 是否真的低，否则 stall 只是正常等待的统计。
- 把 NCU 单 kernel 结论外推到 LLM serving 端到端。调度、通信、cache 命中率和 batch 形态可能改变瓶颈。
