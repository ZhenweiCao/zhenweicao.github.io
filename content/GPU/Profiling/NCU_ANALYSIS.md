---
tags:
  - gpu-computing
  - performance-profiling
---
# NCU 完整分析报告：Naive vs Tiled MatMul

> **测试环境**  
> GPU: NVIDIA B300 SXM6 AC (sm_103, CC 10.3)  
> SM 数量: 148 | 每 SM 调度器: 4 | 最大 Warp/SM: 64  
> DRAM 频率: 4.00 GHz | SM 频率: 1.09 GHz  
> 矩阵规模: 2048×2048 float32 | TILE_SIZE: 32  

> **口径提醒**  
> 这篇是一次具体 NCU 报告的案例分析，不是 B300 或 NCU 指标的通用结论。`Memory Throughput`、`L1/TEX Throughput`、`DRAM Throughput` 都是相对峰值的 SOL 指标，不能直接理解成“传输了多少字节”。准确指标口径见 [[Nsight Compute NCU 分析方法与优化思路]]。

---

## 目录

1. [ncu 报告文件说明](#1-ncu-报告文件说明)
2. [如何读懂一份 ncu 报告](#2-如何读懂一份-ncu-报告)
3. [Section 1：GPU Speed Of Light Throughput](#3-section-1gpu-speed-of-light-throughput)
4. [Section 2：Launch Statistics（启动参数）](#4-section-2launch-statistics启动参数)
5. [Section 3：Occupancy（Warp 占用率）](#5-section-3occupancy-warp-占用率)
6. [Section 4：Memory 深度分析](#6-section-4memory-深度分析)
7. [Section 5：计算管线分析](#7-section-5计算管线分析)
8. [Workload Distribution（负载分布）](#8-workload-distribution负载分布)
9. [综合对比与瓶颈判断](#9-综合对比与瓶颈判断)
10. [优化路线图](#10-优化路线图)
11. [ncu 进阶使用技巧](#11-ncu-进阶使用技巧)
12. [参考与口径依据](#12-参考与口径依据)

---

## 1. ncu 报告文件说明

```
report_naive.ncu-rep   ← matmul_naive_kernel 的完整 ncu 报告
report_tiled.ncu-rep   ← matmul_tiled_kernel 的完整 ncu 报告
```

### 打开方式

```bash
# 方式一：命令行摘要（快速查看）
ncu --import report_naive.ncu-rep --print-summary per-kernel

# 方式二：命令行原始所有指标
ncu --import report_naive.ncu-rep --page raw

# 方式三：导出 CSV（便于 Excel/脚本处理）
ncu --import report_naive.ncu-rep --csv > naive_metrics.csv

# 方式四：GUI（最直观，需图形界面）
ncu-ui report_naive.ncu-rep
```

---

## 2. 如何读懂一份 ncu 报告

ncu 把分析结果组织为多个 **Section**，每个 Section 关注一个维度：

```
GPU Speed Of Light Throughput  ← 总体健康度（最先看这里）
GPU and Memory Workload        ← 各级存储的活跃周期比较
Launch Statistics              ← 你的 kernel 配置是否合理
Occupancy                      ← Warp 利用率
Memory Workload Analysis       ← 内存访问细节（需 --set full 或 --section）
Compute Workload Analysis      ← 计算指令组成
Warp State Statistics          ← Warp 在做什么 vs 在等什么
```

**看报告的顺序**：
1. 先看 **Speed of Light** → 总体瓶颈在计算还是内存？
2. 看 **Occupancy** → Warp 够不够多来隐藏延迟？
3. 看 **Memory** → 是哪级存储成为瓶颈？
4. 看 **Warp State** → 时间都花在哪里？

---

## 3. Section 1：GPU Speed Of Light Throughput

这是最重要的 Section，给出各资源相对峰值的利用率百分比。

### 3.1 完整数据对比

| 指标 | Naive | Tiled | 说明 |
|------|-------|-------|------|
| **Duration** | 4.32 ms | 2.97 ms | 墙钟时间（ncu 分析时比正常运行慢，因为需要多次 replay） |
| **Elapsed Cycles** | 4,730,576 | 3,257,072 | SM 时钟周期数 |
| **SM Active Cycles** | 4,601,715 | 3,201,459 | SM 实际工作周期数 |
| **Memory Throughput** | 77.37% | **91.68%** | 内存子系统综合 SOL，由多个 memory breakdown 指标中的高贡献项决定 |
| **L1/TEX Throughput** | 78.93% | **92.95%** | L1TEX 子系统吞吐率，可能包含 L1 cache、shared data path、texture/surface 等活动 |
| **L2 Cache Throughput** | 3.44% | 3.39% | L2 利用率（两者都极低） |
| **DRAM Throughput** | 0.10% | 0.15% | HBM 侧吞吐占峰值比例极低 |
| **Compute (SM) Throughput** | **77.33%** | 76.91% | `sm__throughput` 高层 SOL 指标，不等同于 FLOPS 利用率 |

### 3.2 如何解读 "SM Throughput"

NCU Speed Of Light 里的 `Compute (SM) Throughput` 通常对应：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

它是 **SM 子系统的高层 throughput metric**，含义是：这个 kernel 运行的 elapsed cycles 内，SM 相关硬件活动接近该架构持续峰值能力的百分比。它不是：

- 不是 FLOPS / 理论 FLOPS。
- 不是 FMA 数量 / 理论 FMA 数量。
- 不是 Tensor Core 利用率。
- 不是 active warp 数，也不是 occupancy。

更准确的心智模型是：

```text
sm__throughput
  ≈ max(SM breakdown 中各个子管线/子活动各自的 % of peak)
```

也就是说，某个 SM 子管线或 SM 内活动接近峰值，就可能把 `sm__throughput` 拉高。具体是哪一项贡献最大，要用 breakdown 看：

```bash
ncu --metrics breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed ...
```

所以本文里 Naive 77.33%、Tiled 76.91% 的正确读法是：**两者平均 SM 高层资源压力相近**。它不能直接推出“两者数学计算单元利用率相同”，也不能说明 FMA/Tensor Core 利用率相同。

访存慢会不会影响 `SM Throughput`？会，但通常是**间接影响**：

- 如果 warp 等 global/local memory，scheduler 没有足够 eligible warp 可发，SM 子管线空转时间变多，`sm__throughput` 往往会下降；这时常伴随 `Long Scoreboard`、`eligible warps` 少、`Issue Active` 低。
- 如果访存虽然慢，但有足够 warp 隐藏延迟，或者 load/store / shared memory / L1TEX 路径本身很忙，`sm__throughput` 不一定低；压力可能更多体现在 `Memory Throughput`、`L1/TEX Throughput`、`L2 Throughput` 或 `DRAM Throughput`。

一句话：**慢访存不直接等于低 SM Throughput；它通过让 warp 等待、减少可发射指令，或者让 SM 内访存相关管线忙起来，间接改变 SM Throughput。**

### 3.3 如何解读"Memory Throughput"

Memory Throughput 不是 L1/L2/DRAM 的字节带宽相加，也不严格等于这三者中某一个显示项。它是 NCU 的高层 throughput 指标：多个底层 memory constituent counter 会先各自换算成 `% of peak`，再由高贡献项决定高层 SOL。

本报告里：

- Naive: Memory Throughput 77.37%，L1/TEX Throughput 78.93%，DRAM Throughput 0.10%。
- Tiled: Memory Throughput 91.68%，L1/TEX Throughput 92.95%，DRAM Throughput 0.15%。

**关键洞察**：两者都不是 HBM/DRAM 带宽瓶颈。高 Memory Throughput 更可能来自 L1TEX / shared data path / memory pipeline 活动，而不是 DRAM。这个测试规模下 A、B、C 三个矩阵总工作集约 48 MiB，报告中的 L2 Cache 约 126 MiB，因此 L2 与片上访问局部性会显著降低 HBM 侧流量；不要把原因归结为单 SM 的 L1 能容纳矩阵数据。

### 3.4 Speed of Light 的"理想状态"

ncu 会在 GUI 中画一个 Roofline 图，横轴是计算强度（FLOP/Byte），纵轴是 GFLOPS。理想情况下：
- 计算密集型 kernel → Compute Throughput 接近 100%，Memory Throughput 低
- 带宽密集型 kernel → Memory Throughput 接近 100%，Compute Throughput 低
- 我们的两个 kernel：**内存和计算双高**，接近 balanced workload

---

## 4. Section 2：Launch Statistics（启动参数）

### 4.1 完整数据对比

| 指标 | Naive | Tiled | 说明 |
|------|-------|-------|------|
| Block Size | 1024 (32×32×1) | 1024 (32×32×1) | 每 Block 线程数 |
| Grid Size | 4096 (64×64×1) | 4096 (64×64×1) | Block 总数 |
| Registers Per Thread | **30** | **32** | 每线程寄存器数 |
| Registers Per Thread (allocated) | 32 | 32 | 实际分配值，通常会按硬件分配粒度向上取整 |
| Static Shared Memory Per Block | 0 bytes | **8.19 KB** | Tiled 的 As+Bs 各 32×32×4=4096 bytes |
| Driver Shared Memory Per Block | 1.02 KB | 1.02 KB | CUDA driver/runtime 保留的 shared memory 口径 |
| Total Shared Memory Per Block | 1.02 KB | **9.22 KB** | — |
| # SMs | 148 | 148 | — |
| Waves Per SM | 13.84 | 13.84 | 4096 blocks / 148 SMs ÷ 2 blocks/SM ≈ 13.84 波 |

### 4.2 Registers Per Thread 的影响

寄存器是**每 SM 最宝贵的资源**之一。B300 每 SM 有 65536 个寄存器。

- 每 Block 1024 线程 × 32 寄存器/线程 = **32768 寄存器/Block**
- 每 SM 65536 ÷ 32768 = **2 个 Block 可同时驻留**（Block Limit Registers = 2）

这是 Occupancy 的关键约束，见下节。

### 4.3 Static Shared Memory 的影响

Tiled kernel 每 Block 使用 **8.19 KB** Shared Memory（8192 bytes 数据 + 1024 bytes driver）：

- B300 每 SM Shared Memory: 233 KB（可配置）
- 当前配置: 32.77 KB（`Shared Memory Configuration Size`）
- 32.77 KB ÷ 9.22 KB/Block = **3 个 Block 可驻留**（Block Limit Shared Mem = 3）

> Naive 的 Shared Memory 限制是 8，Tiled 降到 3——但真正的瓶颈是寄存器（2），所以 Shared Memory 不是实际约束。

### 4.4 Waves Per SM

```
Waves Per SM = Grid Size / (# SMs × resident blocks per SM)
             = 4096 / (148 × 2)
             = 13.84 波
```

每一"波"是一批同时在 SM 上运行的 Block。这里的 `2 blocks/SM` 不是硬件绝对最大值，而是这个 kernel 受寄存器、warp 数、shared memory 等资源限制后，每个 SM 实际最多能同时驻留的 block 数。

所以一波最多同时跑：

```text
148 SM × 2 blocks/SM = 296 blocks
```

总共有 4096 个 block，因此需要：

```text
4096 / 296 = 13.84 波
```

13.84 波意味着前 13 波基本铺满所有 SM，最后还有 0.84 波。尾波比较厚，所以 tail effect 不明显；如果是 13.05 这种最后只剩很少 block 的情况，尾效应会更明显。

---

## 5. Section 3：Occupancy（Warp 占用率）

Occupancy 衡量 SM 上实际活跃 Warp 数与理论最大值的比例，是隐藏延迟的关键。

### 5.1 完整数据对比

| 指标 | Naive | Tiled | 说明 |
|------|-------|-------|------|
| Block Limit Registers | 2 | 2 | 寄存器限制最多 2 Block/SM |
| Block Limit Shared Mem | 8 | **3** | Smem 限制最多 3 Block/SM（Tiled 更严） |
| Block Limit Warps | 2 | 2 | Warp 数量限制最多 2 Block/SM |
| Block Limit Barriers | 32 | 32 | 同步屏障不是约束 |
| **Theoretical Active Warps/SM** | **64** | **64** | 2 blocks × 32 warps/block |
| **Theoretical Occupancy** | **100%** | **100%** | 达到理论上限 |
| **Achieved Occupancy** | **98.04%** | **98.16%** | 实测 Warp 活跃率（极好） |
| Achieved Active Warps/SM | 62.74 | 62.82 | 实测平均活跃 warp 数 |

### 5.2 为什么 Occupancy 是 100%？

这里的关键是：**限制驻留 block 数，不等于限制 occupancy 到很低**。因为这个 kernel 的 block 本身很大。

先算每个 block 有多少 warp：

```text
threads_per_block = 1024
warps_per_block = 1024 / 32 = 32 warps
```

再看寄存器限制：

```text
Block Limit Registers = 2 blocks/SM
```

意思是：由于每个线程要用约 32 个寄存器，一个 SM 的寄存器文件最多支持同时放下 2 个这样的 block。

这时一个 SM 上的理论 active warp 数是：

```text
2 blocks/SM × 32 warps/block = 64 warps/SM
```

而 B300 这份报告里的硬件上限是：

```text
max warps/SM = 64
```

所以：

```text
Theoretical Occupancy = active warps / max warps
                      = 64 / 64
                      = 100%
```

也就是说，寄存器确实把驻留 block 数限制到了 2，但每个 block 有 32 个 warp，两个 block 已经刚好把 SM 的 64 个 warp 槽位填满了。因此 occupancy 仍然是 100%。

Tiled 虽然 Shared Memory 限制更严（3 blocks），但寄存器仍然限制在 2，所以 Occupancy 相同。

### 5.3 Occupancy 不是越高越好

高 Occupancy（≥50%）通常足以隐藏内存延迟。我们的 kernel 已经 98%+，继续提高没有意义。  
应该关注的是：**在当前 Occupancy 下，warp 的时间都花在哪里？**

---

## 6. Section 4：Memory 深度分析

### 6.1 内存访问路径全景

```
线程寄存器
    ↓ (本地变量)
Shared Memory / L1TEX（SM 内同层片上资源，延迟低但口径随架构变化）
    ↕
L1/TEX Cache / Shared Data Path
    ↕
L2 Cache (126 MB 全芯片，~200 cycles)
    ↕
HBM (DRAM，~500-700 cycles)
```

### 6.2 Workload Distribution（各级存储的活跃周期）

单个 kernel 运行期间，各级存储各自"工作了多少时间"：

#### Naive Kernel
| 存储层级 | Average Active Cycles | Total Elapsed Cycles | 活跃率 |
|---------|----------------------|---------------------|--------|
| SM | 4,601,715 | 694,811,300 | 98.0% active |
| L1 | 4,601,715 | 694,811,300 | 98.0% active |
| L2 | 5,086,154 | 1,221,746,936 | 97.4% active |
| DRAM | **16,476** | 1,104,861,184 | **0.10% active** |

#### Tiled Kernel
| 存储层级 | Average Active Cycles | Total Elapsed Cycles | 活跃率 |
|---------|----------------------|---------------------|--------|
| SM | 3,201,459 | 480,419,762 | 98.1% active |
| L1 | 3,201,459 | 480,419,762 | 98.1% active |
| L2 | 2,410,808 | 841,482,324 | 96.9% active |
| DRAM | **16,461** | 760,696,320 | **0.15% active** |

**关键发现**：DRAM 几乎处于空闲状态（< 0.2%）。这不是 “L1 装下了矩阵”。单个 2048×2048 float32 矩阵约 16 MiB，A、B、C 三个矩阵总共约 48 MiB，远大于单 SM 的 L1/SMEM，但小于这份报告中的 L2 Cache（约 126 MiB）。因此更合理的解释是：这个规模下工作集能较好地被 L2 与片上路径承接，加上访问局部性较好，HBM 侧压力很低。

### 6.3 L1 Cache 详细指标

| 指标 | Naive | Tiled | 说明 |
|------|-------|-------|------|
| L1/TEX Throughput | 78.93% | **92.95%** | L1TEX / shared data path 繁忙度 |
| Data Bank Reads (% peak) | 19.93% | **31.44%** | L1TEX data bank 读压力；Tiled 中主要体现 shared memory 读 |
| Data Bank Writes (% peak) | 1.22% | **3.51%** | L1TEX data bank 写压力；Tiled 中包含 shared memory 写 |
| L2 Cache Throughput | 3.44% | 3.39% | L2 需求很低 |

**Naive 的 Data Bank Reads 低于 Tiled 的原因**：Tiled 显式把 tile 搬到 shared memory，后续内层计算会反复读 shared memory，因此 L1TEX data bank 活动更高。Naive 虽然也会经过 L1TEX 处理 global load，但复用主要依赖硬件 cache 路径，和 shared memory 显式复用的指标表现不同。

### 6.4 为什么 `l1tex__t_sector_hit_rate` Naive=95%，Tiled=0%？

`l1tex__t_*` 中的 `t` 指 L1TEX 的 tag stage。这个命中率主要描述需要 tag lookup 的 L1TEX cache 路径，不能当成 shared memory 访问是否命中的指标。

- **Naive**：A/B 大量访问走 global memory -> L1TEX cache 路径，可能看到较高 tag hit rate。
- **Tiled**：A/B tile 从 global memory 搬进 shared memory 后，后续复用主要走 shared memory data bank，不再依赖 L1 cache tag lookup，所以这个指标可能很低甚至为 0。

> 这不是 Tiled 的缺点。它说明主要复用路径从“硬件 L1 cache 自动命中”转成了“shared memory 显式复用”。Tiled 仍然需要 global load 把 tile 搬进来，只是后续重复使用不再主要依赖 L1 cache hit rate。

### 6.5 DRAM 带宽利用率极低的原因分析

```
理论 DRAM 访问量（无缓存）：
  Naive: 每次 FMA 读 2 个全局内存元素 = 2 × N³ × 4 bytes = 2 × 2048³ × 4 ≈ 68.7 GB
  Tiled: 每个输出 tile 都要沿 K 维加载 A/B tile
         约 2 × N³ / TILE_SIZE × 4 bytes
         = 2 × 2048³ / 32 × 4 ≈ 2.15 GB

唯一数据量下界：
  A + B + C ≈ 3 × 2048² × 4 bytes ≈ 48 MiB
  如果 L2/cache 复用很好，实际 HBM 流量会更接近这个量级，而不是理论无缓存访问量。

实测 DRAM 活跃率 0.10-0.15%：
  说明在这个输入规模和 profiling 条件下，L2/cache 复用很好，
  HBM 侧不是主要瓶颈。换更大矩阵、冷 cache 或不同 launch 顺序时不能直接外推。
```

---

## 7. Section 5：计算管线分析

### 7.1 SM 指令发射效率

| 指标 | Naive | Tiled | 说明 |
|------|-------|-------|------|
| SM Issue Active | **59.36%** | 41.74% | 每个时钟周期内调度器发射指令的比例 |
| SM Throughput | 77.33% | 76.91% | SM 高层 SOL，平均资源压力相近 |
| FMA Pipe Active (light) | **36.26%** | 16.64% | 普通 FMA 管线活跃率 |
| FMA Pipe Active (heavy) | **51.15%** | 7.53% | 架构相关的 FMA-heavy 管线活跃率；不要把它直接等同于 Tensor Core |

**Naive 的 Issue Active 更高（59% > 42%）**：

这反映了一个有趣的现象：Naive kernel 的内层循环更简单，指令 mix 更少，没有 tile 级 `__syncthreads()`。虽然累加变量本身存在依赖链，但编译器可能通过展开和多条 load/FMA 交错提供一定 ILP。Tiled kernel 增加了 load tile、shared memory 访问和 block 同步，调度器发射节奏可能更不连续，Issue Active 下降。

### 7.2 为什么 Tiled 更快但 FMA Active 更低？

```
Naive:
  FMA heavy pipe active = 51.15%
  Duration = 4.32 ms
  工作量 = 2048³ 次 FMA ≈ 8.59 × 10⁹ FMA
        = 2 × 2048³ FLOPs ≈ 17.2 × 10⁹ FLOPs

Tiled:
  FMA heavy pipe active = 7.53%
  Duration = 2.97 ms
  工作量相同 = 8.59 × 10⁹ FMA / 17.2 × 10⁹ FLOPs
```

矛盾！同样的数学工作量，Tiled 更快，但 FMA Active 反而更低？

**更谨慎的解释**：不要用单个 `fmaheavy` 指标直接推断“计算工作量”。`fma` / `fmaheavy` 是架构相关的执行管线分类，不等同于 Tensor Core。Tensor Core 应该看 tensor pipe、HMMA/WGMMA/MMA 指令或对应的 ComputeWorkloadAnalysis/SASS。

Tiled 的 `__syncthreads()`、shared memory 访问和 tile 加载会引入周期性的“加载 -> 同步 -> 计算 -> 同步”结构，FMA 管线在时间上更不连续，平均 active 百分比可能降低。要确认原因，需要进一步看：

- ComputeWorkloadAnalysis 的 instruction mix。
- SchedulerStats 的 issue active / issue stall。
- WarpStateStats 的 barrier、short/long scoreboard、MIO throttle。
- Source/SASS 中是否出现预期的 FMA 指令。

更稳的对照指标是 **SM Throughput（`sm__throughput`）**：两者都是 ~77%，说明平均 SM 高层资源压力相近。它不等同于 FLOPS 利用率，也不能单独证明 FMA/Tensor Core 利用率相同。Tiled 更快的直接证据是 elapsed cycles / duration 更少：它在更短时间内完成了同样矩阵乘数学结果。

---

## 8. Workload Distribution（负载分布）

### 8.1 各层级时钟周期对比

| 层级 | Naive Total Cycles | Tiled Total Cycles | 比值 |
|------|-------------------|-------------------|------|
| SM (Total) | 694,811,300 | 480,419,762 | Tiled 少 31% |
| L2 (Total) | 1,221,746,936 | 841,482,324 | Tiled 少 31% |
| DRAM (Total) | 1,104,861,184 | 760,696,320 | Tiled 少 31% |

所有层级的总 elapsed cycles 都减少了约 31%，与运行时间的缩短（4.32→2.97ms，减少 31%）一致。

注意：elapsed cycles 下降本身只是运行时间变短的另一种表达，不能单独证明“总工作量减少”。要证明 Tiled 减少了冗余访存，需要结合 global load/store 数、L1/L2 sector、shared memory 访问和源码结构一起看。这个案例更合理的结论是：Tiled 把一部分复用从 global/cache 路径转移到 shared memory 路径，降低了部分 L2/全局路径压力，但也引入了 shared memory 与同步开销。

### 8.2 DRAM vs L2 vs L1 活跃周期对比

```
Naive:
  DRAM 活跃: 16,476 cycles   (相对 L1 的 4,601,715: 仅 0.36%)
  L2 活跃:   5,086,154 cycles (相对 L1: 110.5%)
  L1 活跃:   4,601,715 cycles

Tiled:
  DRAM 活跃: 16,461 cycles   (相对 L1 的 3,201,459: 仅 0.51%)
  L2 活跃:   2,410,808 cycles (相对 L1: 75.3%)
  L1 活跃:   3,201,459 cycles
```

L2 活跃周期和 L1 活跃周期来自不同硬件单元，不能当成同一条流水线上的延迟直接相除。更稳妥的读法是：L2 需要汇聚多个 SM 的请求，活跃周期更长并不奇怪；L2 相对于 L1 的活跃比从 Naive 的 110.5% 降到 Tiled 的 75.3%，说明 Tiled 可能减少了对 L2/cache 后端的压力，但最好结合 L2 sectors、requests 和 hit rate 一起确认。

---

## 9. 综合对比与瓶颈判断

### 9.1 性能汇总表

| 指标 | Naive | Tiled | 对比解读 |
|------|-------|-------|----------|
| 运行时间（正常） | 2.71 ms | 1.62 ms | **1.67x 加速** |
| GFLOPS | 6339 | 10578 | **+67%** |
| Elapsed Cycles | 4,730,576 | 3,257,072 | 少 31% |
| SM Throughput | 77.33% | 76.91% | 相当 |
| Memory Throughput | 77.37% | **91.68%** | Tiled 的内存子系统 SOL 更高 |
| L1TEX Throughput | 78.93% | **92.95%** | Tiled 的片上访存路径压力更高 |
| DRAM Throughput | 0.10% | 0.15% | 均极低 |
| Achieved Occupancy | 98.04% | 98.16% | 均接近满载 |
| Issue Active | **59.36%** | 41.74% | Naive 指令发射更密集 |
| Registers/Thread | 30 | 32 | 相近 |
| Smem/Block | 0 | 8.19 KB | Tiled 显式使用 |

### 9.2 真正的瓶颈是什么？

**两个 kernel 都不是 HBM 带宽瓶颈，更像 on-chip memory pipeline 与 SM 计算之间的 balanced 状态**：

- Memory Throughput 和 Compute Throughput 都在 77% 左右或以上。
- DRAM Throughput 极低，不能称为 HBM-bound。
- L1TEX / shared data path 很高，说明压力更多落在片上访存路径、shared memory、load/store pipe、同步和指令发射节奏上。

所以这里不要简单写成 “Memory Bound”。更准确的说法是：**balanced，但 on-chip memory/L1TEX 路径压力明显**。

**Tiled 为什么没有获得更大加速？**

在一些老架构或更大工作集上，Tiled 可能获得更大加速，原因通常是 naive kernel 的全局访存复用差、L1/L2 容量或复用距离不足，必须更频繁访问 L2/DRAM。本案例中，矩阵总工作集约 48 MiB，小于报告中的 L2 Cache 约 126 MiB，Naive 的重复访问也能从缓存局部性受益，因此 Tiled 与 Naive 的差距没有“理论无缓存模型”那么夸张。

### 9.3 如何用 ncu 判断瓶颈

```
看 Speed of Light Section:
  Memory Throughput >> Compute Throughput  → 内存瓶颈
  Compute Throughput >> Memory Throughput  → 计算瓶颈
  两者接近                                 → balanced（好的状态或双重瓶颈）

细分内存瓶颈：
  DRAM Throughput 高（>50%）              → 带宽瓶颈，优化方向：减少 DRAM 访问
  L2 Throughput 高（>50%）               → L2 是瓶颈，优化方向：提高 L1 复用
  L1TEX Throughput 高，DRAM/L2 低        → 压力在片上访存路径，不是 HBM 瓶颈

判断计算瓶颈：
  SM Issue Active 低（<50%）             → 调度器空闲，可能有等待
  FMA Active 低                          → 对应 FMA 管线平均活跃不高，需要结合 instruction mix 判断
  Achieved Occupancy 低（<50%）          → Warp 不够多，延迟无法被隐藏
```

---

## 10. 优化路线图

基于以上分析，下一步优化应按此顺序进行：

### 10.1 当前状态评估

```
✅ Occupancy: 98%+（无需优化）
✅ DRAM Throughput: 0.1%（无 DRAM 瓶颈）
⚠️  L1TEX / on-chip memory path: 78-93%（接近上限）
⚠️  SM Throughput: 77%（仍有优化空间，但不代表性能可线性提升 23%）
⚠️  FMA Heavy Active: 7-51%（只能说明该管线平均活跃不高，不能单独代表全部计算单元）
```

### 10.2 优化 1：检查 Shared Memory Bank Conflict

Tiled kernel 中，`As[TILE_SIZE][TILE_SIZE]` 访问模式：
- `As[threadIdx.y][k]`：同一 Warp 内 32 个线程同时读同一行的 k 列 → **列广播，无 conflict**
- `Bs[k][threadIdx.x]`：同一 Warp 内 32 个线程读同一行的连续列 → 对 32-bit float 通常会落到不同 bank，理论上也不应有严重 conflict

```bash
# 检测 bank conflict
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --kernel-name matmul_tiled_kernel ./matmul
```

因此这里不应该先假设一定有 bank conflict。应该先用 NCU 指标验证；如果后续改成转置 shared tile、非连续访问、不同 dtype packing 或 `ldmatrix` 路径，再考虑 padding / swizzle。若指标确实显示 conflict，可尝试：
```cpp
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
__shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
```

### 10.3 优化 2：1D Block Tiling（每线程计算 8 行）

目标：提高计算/内存访问比（Arithmetic Intensity），让每次 shared memory 读取服务更多 FMA。它不一定减少 `__syncthreads()` 次数，但能降低“每次计算需要多少 shared 读”的压力。

参考 `matmul_1dblocktiling_kernel` 的思路（详见 `/share/zhenwei/Codes/lectures/demo/cuda/matmul/matmul.cu`）：
- 每线程负责 TM=8 行 × 1 列
- B 的一个值被 8 次 FMA 复用
- Shared Memory 读:计算比从 2:1 改善到 ~1:1

预期效果：`sm__pipe_fma_cycles_active` 和 SM Throughput 有望提升，但需要用 NCU 复测确认。

### 10.4 优化 3：2D Block Tiling（每线程计算 4×4 子矩阵）

参考 `matmul_2dblocktiling_kernel`：
- 每线程计算 TM=4 × TN=4 = 16 个输出
- 使用寄存器存储 `a_frag[4]` 和 `b_frag[4]`，外积计算
- 16 次 FMA 只需 8 次 Shared Memory 读

### 10.5 优化 4：Double Buffering（流水线化）

```cpp
__shared__ float As[2][TILE_SIZE][TILE_SIZE];
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

// 预加载第 0 个 tile
load_tile(0, 0);
__syncthreads();

for (int t = 1; t < num_tiles; t++) {
    int curr = (t - 1) % 2, next = t % 2;
    // 异步加载下一个 tile（与当前 tile 的计算并行）
    async_load_tile(next, t);  // 使用 cuda::memcpy_async
    compute(curr);
    __syncthreads();
}
compute(num_tiles % 2);  // 处理最后一个 tile
```

预期效果：把 global-to-shared 搬运和当前 tile 计算重叠，减少等待搬运的空洞。它不能完全消除同步，仍需要正确的 async copy wait / barrier 机制；Issue Active 是否提升要以 SchedulerStats 和 WarpStateStats 为准。

### 10.6 优化路线预期收益

| 优化 | 解决问题 | 预期指标改善 |
|------|---------|------------|
| Bank Conflict 验证/消除 | 可能的 Smem bank 冲突 | conflict 指标下降，shared 访问更稳定 |
| 1D/2D Block Tiling | 计算/访存比低 | FMA Active / SM Throughput 有望提升 |
| Double Buffering | load tile 与 compute 串行 | issue 空洞减少，barrier/wait stall 下降 |
| float4 向量化读 | load/store 指令和请求开销 | 指令数或 request 数下降；L1TEX throughput 不一定下降 |
| cuBLAS 对标 | — | SM Throughput 85%+，GFLOPS 2-3x |

---

## 11. ncu 进阶使用技巧

### 11.1 只分析特定 Kernel

```bash
# 按 kernel 名字过滤
ncu --kernel-name matmul_naive_kernel ./matmul

# 按正则表达式
ncu --kernel-name-base regex --kernel-name "matmul.*" ./matmul

# 按 launch 次序跳过 warmup，只采一次
ncu --launch-skip 10 --launch-count 1 ./matmul
```

### 11.2 控制 Replay 次数（加速分析）

ncu 通过多次"回放"（replay）kernel 来收集不同指标，`--set full` 需要更多 replay：

```bash
# 只收集特定 section（更快）
ncu --section SpeedOfLight ./matmul
ncu --section MemoryWorkloadAnalysis ./matmul
ncu --section ComputeWorkloadAnalysis ./matmul

# 查看 SM 相关可用指标
ncu --query-metrics-mode suffix --query-metrics sm__
```

### 11.3 对比两个报告

```bash
# 在 CLI 中对比（需要 ncu >= 2023）
ncu --import report_naive.ncu-rep --import report_tiled.ncu-rep --diff

# 导出 CSV 后用脚本对比
ncu --import report_naive.ncu-rep --csv > naive.csv
ncu --import report_tiled.ncu-rep --csv > tiled.csv
```

### 11.4 Source Correlation（关联到源码行）

需要编译时加 `-lineinfo`（我们的 Makefile 已加）：

```bash
ncu --set full -o report_full --source-level-analysis global_access ./matmul
# 在 ncu-ui 中可以看到每行代码对应的内存访问热点
```

### 11.5 Roofline 分析

```bash
ncu --set roofline -o report_roofline ./matmul
# 在 ncu-ui 中查看 Roofline 图
# 横轴：Arithmetic Intensity (FLOP/Byte)
# 纵轴：实际 GFLOPS
# 理论上限：min(峰值计算性能, 带宽 × AI)
```

### 11.6 常用指标速查

```bash
# 计算利用率
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed

# 内存层次
ncu --metrics \
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
  lts__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__throughput.avg.pct_of_peak_sustained_active

# Warp 效率
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__issue_active.avg.pct_of_peak_sustained_elapsed

# Shared Memory bank conflict
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

# FMA 吞吐
ncu --metrics \
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed
```

---

## 12. 参考与口径依据

- [[Nsight Compute NCU 分析方法与优化思路]]：本仓库维护的 NCU 指标阅读主文档。
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：SpeedOfLight、throughput metric、L1TEX、MemoryWorkloadAnalysis 的官方口径。
- [NVIDIA Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：命令行参数、section、metric 查询和 report 导出。

---

## 附录：B300 硬件参数（从报告提取）

> **口径说明（2026-05-30 修订）**：下表数字来自一次 NCU 报告中 Device Attributes 直接读取，反映该次运行的**实测/驱动报告值**；与 NVIDIA 官方页发布的 Blackwell Ultra 名义规格可能存在小幅差异（例如 SM 数量取决于具体 SKU，sm_103 的 CC 数字仍以最新 [CUDA Programming Guide 附录 H](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 与 [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) 为准）。用于解读本 NCU 案例时直接使用；用于他处文档引用时优先查 [[NVIDIA GPU 架构与规格]]。

| 参数 | 值 |
|------|-----|
| 计算能力 | 10.3 (sm_103) |
| SM 数量 | 148 |
| 每 SM 调度器 | 4 |
| 最大 Warp/SM | 64 |
| 最大 Thread/Block | 1024 |
| 最大 Block/SM | 32 |
| 寄存器/SM | 65536 |
| 最大 Register/Thread | 255 |
| Shared Memory/SM (可配) | 最大 233,472 bytes (~228 KB) |
| L2 Cache 大小 | 132,644,864 bytes (~126 MB) |
| L2 分区数 | 192 |
| 总显存 | 287,431,131,136 bytes (~268 GB) |
| SM 频率（测试时） | 1.09 GHz |
| DRAM 频率（测试时） | 4.00 GHz |
| FP32 / FP64 性能比 | 64:1 |
| PCIe Gen/Width | Gen 5 × 16 |
