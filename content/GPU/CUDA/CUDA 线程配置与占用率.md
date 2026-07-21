---
aliases:
  - CUDA线程配置与SM占用率
updated: 2026-05-30
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
---
# CUDA 线程配置与占用率

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA Kernel 示例：向量加法]]
- [[CUDA Kernel 示例：归约求和]]

## 1. 基本概念回顾

### 层次结构

```
Grid
 └── Block（线程块）
      └── Warp（32 个线程，硬件调度单元）
           └── Thread（线程）
```

- **Warp**：GPU 的最小调度单元，始终 32 个线程一起执行同一条指令（SIMT）
- **Block**：被调度到同一个 SM 上执行，block 内线程共享 Shared Memory 和同步屏障
- **Grid**：所有 block 的集合，block 之间无法直接通信（除 Cooperative Groups）

### SM 资源上限：硬件上限还是软件规则？

这些值大多是 **compute capability 暴露出来的硬件/架构上限**，CUDA runtime 会在 launch 时检查它们。它们可以分成三类：

| 类型 | 例子 | 怎么理解 |
|------|------|----------|
| SM 驻留上限 | 每 SM 最大 warp/thread/block 数 | 一个 SM 同时能保留多少 warp/block 的执行上下文。 |
| 片上资源容量 | 寄存器文件、shared memory/L1 carveout | 每个 resident block/thread 要消耗寄存器和 shared memory，放不下就不能同时驻留。 |
| CUDA 编程模型限制 | 每 block 最大线程数 1024、block 维度上限 | 一个 CTA/block 必须整体驻留在一个 SM 上，硬件和 runtime 都要求它不要无限大。 |

注意区分两个概念：

- **架构上限**：某个 compute capability 给出的绝对上限，例如每 SM 最多 64 warps。
- **实际驻留上限**：某个 kernel 在当前 block size、寄存器数、shared memory、barrier、cluster size 下实际能驻留多少 block/warp。

举例：硬件允许每 SM 最多 32 个 resident blocks，但如果你的 kernel 每个 block 用了很多寄存器，Nsight Compute 里可能显示 `Block Limit Registers = 2`，也就是实际只能驻留 2 个 block/SM。

### 常见架构资源上限

| 资源 | Ampere A100，CC 8.0 | Hopper H100/H200，CC 9.0 | Blackwell B200，CC 10.0 | Blackwell Ultra B300/GB300，CC 10.3 | Blackwell RTX/桌面类，CC 12.x |
|------|---------------------|---------------------------|--------------------------|-------------------------------|------------------------|
| Warp size | 32 | 32 | 32 | 32 | 32 |
| 每 SM 最大 resident warp 数 | 64 | 64 | 64 | 64 | 48 |
| 每 SM 最大 resident thread 数 | 2048 | 2048 | 2048 | 2048 | 1536 |
| 每 SM 最大 resident block 数 | 32 | 32 | 32 | 32 | 24（见下方说明） |
| 每 SM 寄存器文件 | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit |
| 每线程最大寄存器数 | 255 | 255 | 255 | 255 | 255 |
| 每 block 最大线程数 | 1024 | 1024 | 1024 | 1024 | 1024 |
| 每 SM shared memory 容量 | 164 KB | 228 KB | 228 KB | 228 KB | 100 KB |
| 每 block 最大 shared memory | 163 KB | 227 KB | 227 KB | 227 KB | 99 KB |

> `KB` 在 CUDA 技术规格表里按 1024 bytes 理解。CC 12.x 的 100 KB 指可配置成 shared memory 的最大容量；128 KB 是 unified L1/Texture/Shared Memory 总池口径。CUDA Programming Guide 13.2 的总表把 CC 12.x 的 resident block 上限列为 24；Blackwell Tuning Guide 13.2 对 CC 12.0 仍写 32。做严谨调优时不要死背表格，优先用 `cudaGetDeviceProperties()`、`deviceQuery` 或 Nsight Compute 报告里的设备属性确认。

参考：

- [CUDA C++ Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA Hopper Tuning Guide: Occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
- [NVIDIA Blackwell Tuning Guide: Occupancy](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)

查询当前 GPU：

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, device);

printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
printf("maxThreadsPerMultiProcessor = %d\n", prop.maxThreadsPerMultiProcessor);
printf("maxBlocksPerMultiProcessor = %d\n", prop.maxBlocksPerMultiProcessor);
printf("regsPerMultiprocessor = %d\n", prop.regsPerMultiprocessor);
printf("sharedMemPerMultiprocessor = %zu\n", prop.sharedMemPerMultiprocessor);
printf("sharedMemPerBlockOptin = %zu\n", prop.sharedMemPerBlockOptin);
```

为什么要有这些限制？

1. **每个 resident warp 都要保存状态**：PC、active mask、寄存器映射等都要占硬件资源。
2. **每个 resident block 都要保存 CTA 级资源**：shared memory、barrier、block 调度状态都必须留在同一个 SM 内。
3. **寄存器和 shared memory 是有限片上资源**：用得越多，能同时驻留的 block/warp 越少。
4. **调度器需要有限队列和可预测延迟**：warp scheduler、scoreboard、barrier 追踪、CTA slot 都有固定容量。
5. **限制 block 大小能保持 CUDA 模型简单**：一个 block 不跨 SM，block 内 `__syncthreads()` 和 shared memory 才能高效实现。

---

## 2. Block Size 的选择

### 2.1 核心原则：通常选择 Warp 大小的整数倍

Block size 不强制必须是 **32 的整数倍**，但通常应该这样选。否则最后一个不足 32 线程的 warp 会有 idle 线程，浪费 warp slot。

```
block_size = 100 → 实际调度 4 个 warp（128 线程），28 个线程 idle
block_size = 128 → 实际调度 4 个 warp，无浪费 ✓
```

### 2.2 常见 Block Size 选择

| Block Size | Warp 数 | 适用场景 |
|-----------|---------|---------|
| 64 | 2 | 寄存器/SMEM 压力极大时 |
| 128 | 4 | 寄存器较多或 SMEM 较大时 |
| 256 | 8 | 最常用默认值，适合大多数场景 |
| 512 | 16 | 访存密集型 kernel |
| 1024 | 32 | 每 SM 只放 2 个 block，要求精确控制 |

**经验法则**：从 **256** 开始，若 occupancy 低再调整。

### 2.3 多维 Block（2D/3D）

用于图像处理、矩阵运算等天然二维的问题：

```cuda
dim3 blockDim(32, 8, 1);   // 256 线程，对应 32 列 × 8 行
dim3 blockDim(16, 16, 1);  // 256 线程，正方形 tile
dim3 blockDim(8, 8, 4);    // 256 线程，3D 体素
```

**注意**：硬件始终将线程按 `threadIdx.x` 优先展开成 warp。常见高效做法是让同一 warp 内的线程在 x 方向连续访问，必要时再把 `blockDim.x` 设计成 32 的整数倍。

---

## 3. Grid Size 的选择

### 3.1 刚好覆盖问题规模

```cuda
int grid_x = (N + block_size - 1) / block_size;  // 向上取整
dim3 gridDim(grid_x);
```

### 3.2 Grid 不宜过小

Grid 太小（block 数少于 SM 数）会导致部分 SM 空闲，无法充分利用 GPU。

```
A100: 108 个 SM
如果 grid_size = 64，只有 64 个 SM 工作，44 个 SM 空转
建议: grid_size >= 数倍于 SM 数（通常 >= 4 × SM_count）
```

### 3.3 Grid 不宜过大（对于 persistent kernel）

对于 persistent kernel（kernel 内循环处理多个 tile）：

```
理想 grid_size = SM 数 × 每 SM 驻留 block 数
```

这样每个 SM 的 block 槽位被恰好填满，减少调度开销，且能精确控制波次（wave）。

### 3.4 Wave 量化

整个 GPU 一次能并发执行的 block 数称为一个 **wave**：

```
wave_size = SM 数 × 每 SM 驻留 block 数

total_waves = ceil(grid_size / wave_size)
```

若 `grid_size` 不是 `wave_size` 的整数倍，最后一个不完整的 wave（tail wave）会导致大量 SM 空转。应尽量使 `grid_size` 为 `wave_size` 的整数倍，或者使 tail wave 尽量大。

---

## 4. 线程总数与问题规模

### 4.1 一线程一元素 vs 一线程多元素（Thread Coarsening）

**一线程一元素**（适合访存密集型）：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) output[idx] = f(input[idx]);
```

**Thread Coarsening**（适合计算密集型或寄存器复用场景）：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    output[i] = f(input[i]);
}
```

Coarsening 的优点：
- 减少 block 总数，降低调度开销
- 同一线程复用寄存器中的中间结果，减少重复计算
- 对 reduction、scan 等操作可减少 warp 同步次数

Coarsening 的缺点：
- block 数减少可能降低 GPU 利用率（occupancy 下降）
- 不适用于访存成为瓶颈的场景

---

## 5. SM Occupancy（占用率）

### 5.1 定义：理论 vs 实测

Occupancy 有两个口径，NCU 中分别对应不同指标，**调优时不能混淆**：

| 口径 | 含义 | NCU 指标 |
|------|------|---------|
| **Theoretical Occupancy** | 按 kernel 资源用量（寄存器、SMEM、block size）和硬件上限计算出的"最多能驻留多少 warp/SM"，是上限。 | `Launch Statistics → Theoretical Occupancy` 或 CUDA Occupancy Calculator |
| **Achieved Occupancy** | kernel 实际运行中"平均每个 cycle 真正活跃的 warp 数"，受 warp 间负载不均、block 进度差异、tail 阶段、依赖等影响，**通常 < theoretical**。 | `Occupancy → Achieved Active Warps Per SM` 或 `sm__warps_active.avg.pct_of_peak_sustained_active` |

公式（理论值）：

$$\text{Theoretical Occupancy} = \frac{\text{受资源限制的活跃 warp 数}}{\text{SM 最大 warp 数}}$$

例如 A100 每 SM 最大 64 个 warp，若资源限制下能驻留 32 个 warp，则 theoretical occupancy = 50%；但若 warp 间负载不均或 tail wave 太长，achieved occupancy 可能只有 35%。

### 5.2 限制 Occupancy 的三大资源

#### 寄存器（Registers）

每个 SM 的寄存器总数固定（A100: 65,536 个 32-bit 寄存器）。

**关键：寄存器分配是粒度对齐的，不能简单按"每线程 × block size"算**：

- **每线程寄存器数**按 8 reg 粒度向上取整（A100/Hopper）；
- **每 warp 寄存器数**按 256 reg 粒度向上取整（即 `ceil(per_thread_regs / 8) × 8 × 32 = ceil(per_thread_regs × 32 / 256) × 256`）；
- **每 block 寄存器数**等于 `warps_per_block × per_warp_reg_alloc`。

伪公式：

```
per_thread_reg_alloc = ceil_to(per_thread_regs_used, 8)
per_warp_reg_alloc   = ceil_to(per_thread_reg_alloc * 32, 256)
per_block_reg_alloc  = warps_per_block * per_warp_reg_alloc
max_resident_blocks  = floor(64K / per_block_reg_alloc)
```

示例（A100，更严谨版）：

```
每线程 64 寄存器 → 每线程取整 ceil(64, 8) = 64
                  → 每 warp 取整 ceil(64*32, 256) = ceil(2048, 256) = 2048
block_size = 256 (8 warp)
每 block 寄存器 = 8 × 2048 = 16384
每 SM 最多驻留 block = 65536 / 16384 = 4
活跃 warp = 4 × 8 = 32 / 64 → Theoretical Occupancy = 50%
```

**只有当 `per_thread_regs` 恰好是 8 的倍数、且 `per_thread_regs × 32` 恰好是 256 的倍数（即 per_thread_regs 是 8 的倍数）时，简化公式 `每线程 × block_size` 才与对齐后结果相等**。在 A100/H100 上 8 reg 粒度通常会"吞掉" 1~7 reg 的差异，因此用 `--ptxas-options=-v` 看到 65 reg 时实际按 72 算，看到 72 reg 时按 72 算。

可用 `__launch_bounds__` 提示编译器限制寄存器用量：

```cuda
__global__ __launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerMultiprocessor=4
void my_kernel(...) { ... }
```

#### Shared Memory

```
每 block SMEM 用量 × 每 SM 驻留 block 数 ≤ 每 SM SMEM 总量
```

示例（A100，SMEM = 164 KB，配置为 100 KB 用于 SMEM）：
```
每 block SMEM = 32 KB
每 SM 最多驻留 block = 100 / 32 = 3（取整）
活跃 warp = 3 × (block_size/32)
```

#### Block 数量限制

每 SM 最多同时驻留 block 数有硬性上限（A100: 32），即使资源充足也不能超过。

### 5.3 使用 CUDA Occupancy Calculator

```cuda
int blockSize = 256;
int minGridSize, gridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, my_kernel, 0, 0);

int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, my_kernel, blockSize, 0);
float occupancy = (maxActiveBlocks * blockSize / 32.0f) / max_warps_per_sm;
```

或使用 Nsight Compute 直接查看 `sm__warps_active.avg.pct_of_peak_sustained_active`（achieved）与 `Launch Statistics → Theoretical Occupancy`（theoretical）。

---

## 5bis. Wave Quantization：小问题尺寸下的隐藏成本

当 `grid_size` 不是 `wave_size`（GPU 一次能并发的 block 数）的整数倍时，**最后一个 wave 称为 tail wave**，它只填充部分 SM，其余 SM 空转直到 tail wave 结束。这种现象叫做 **wave quantization**。

示例（A100，108 SM，每 SM 驻留 2 block → `wave_size = 216`）：

| grid_size | wave 数 | 占满情况 | 有效率 |
|-----------|--------|---------|--------|
| 216 | 1.00 | 1 wave 满 | 100% |
| 256 | 1.19 | 1 满 + tail 40/216 | ≈ 84% |
| 432 | 2.00 | 2 wave 满 | 100% |
| 500 | 2.32 | 2 满 + tail 68/216 | ≈ 86% |

实战影响：

- **小问题尺寸**（如 batch 小、序列短、token 数小）GEMM/attention，tail wave 浪费很大，会让 Tensor Core 利用率明显低于理论值。
- **应对**：
  1. 选择能让 `grid_size` 接近 `wave_size` 整数倍的 tile shape；
  2. 用 `persistent kernel` 模式让一组 long-living block 自己 loop 处理多个 tile（避免 tail）；
  3. 在 cuBLASLt/CUTLASS 中开启 `swizzle` / `cooperative` scheduling。
- **观测**：NCU `Launch Statistics → Waves Per SM` 是否接近整数；`Achieved Occupancy` 在 tail wave 中是否骤降。

---

## 6. Occupancy 与性能的关系

### 6.1 高 Occupancy 并非万能

**高 occupancy 的作用**：通过 warp 切换（latency hiding）来掩盖访存延迟。当一个 warp 等待内存数据时，SM 切换到另一个就绪的 warp 执行，维持流水线满载。

**但高 occupancy 不等于高性能**：
- 计算密集型 kernel（如 tensor core GEMM）：occupancy ~25% 即可达峰值，因为计算延迟极短
- 寄存器密集型 kernel：强行提高 occupancy 会导致寄存器 spill 到 local memory（L1/L2），反而降低性能
- SMEM 密集型 kernel：过多 block 驻留会导致 SMEM 无法装下足够大的 tile

### 6.2 实际设计思路

```
访存密集型（Memory-bound）kernel：
  目标：充分 latency hiding
  策略：提高 occupancy → 增大 block_size 或减少每线程寄存器

计算密集型（Compute-bound）kernel（如矩阵乘）：
  目标：最大化计算吞吐（MMA 利用率）
  策略：接受低 occupancy，换取更大 tile/更多寄存器
  典型值：A100 GEMM kernel occupancy ~25%~50%

访存 + 计算均衡型：
  需要 profile，用 roofline model 判断瓶颈
```

### 6.3 Roofline Model 与 Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs / Bytes

AI < Ridge Point → Memory-bound，提高 occupancy 有效
AI > Ridge Point → Compute-bound，优化计算效率更重要
```

对于 A100 FP16：
- 峰值算力：312 TFLOPS
- 内存带宽：2 TB/s
- Ridge Point：312 / 2 = 156 FLOP/Byte

---

## 7. 实际调优流程

```
1. 初始配置
   - block_size = 256，grid_size = (N + 255) / 256

2. 用 Nsight Compute 分析
   - 查看 Theoretical Occupancy vs Achieved Occupancy
   - 查看 Limiting Factor（寄存器 / SMEM / block 数上限）
   - 查看是否 Memory-bound 或 Compute-bound

3. 根据瓶颈调整
   - 寄存器限制 → 用 __launch_bounds__ 或减少中间变量
   - SMEM 限制 → 减小 tile 大小或改用 L1 cache
   - Memory-bound → 提高 occupancy 或优化访存模式（coalescing）
   - Compute-bound → 增大 tile，接受低 occupancy

4. 迭代验证
   - 每次只改一个变量
   - 对比 kernel time，以实测性能为准
```

---

## 8. 常见配置模式

### 向量化操作（elementwise）

```cuda
// 推荐：256 线程，grid 覆盖全部元素
dim3 block(256);
dim3 grid((N + 255) / 256);
```

### 矩阵乘（GEMM）

```cuda
// 典型 tile 配置：每 block 处理 128×128 输出 tile
// 寄存器密集，occupancy 较低（25%~50%）
dim3 block(256);  // 内部 8×4 warp layout
dim3 grid((N + 127) / 128, (M + 127) / 128);
```

### Reduction

```cuda
// 推荐：256 或 512 线程，grid = problem_size / (block_size * coarsening_factor)
dim3 block(256);
dim3 grid((N + block.x * COARSE - 1) / (block.x * COARSE));
```

### 2D 图像处理

```cuda
// x 方向保持 32 的整数倍，保证 coalesced 访问
dim3 block(32, 8);  // 256 线程
dim3 grid((W + 31) / 32, (H + 7) / 8);
```

---

## 9. 快速参考

| 设计目标 | 建议 |
|---------|------|
| block_size 初始值 | 256 |
| block_size 常见选择 | 32 的整数倍 |
| grid_size 下限 | ≥ 4 × SM 数（充分并行） |
| 避免 tail wave 浪费 | grid_size 为 wave_size 整数倍 |
| Memory-bound kernel | 提高 occupancy（减少寄存器/SMEM 用量） |
| Compute-bound kernel | 接受低 occupancy，最大化 tile 大小 |
| 检测 occupancy 限制因素 | Nsight Compute → Occupancy 面板 |
| 限制寄存器溢出 | `__launch_bounds__(block_size, min_blocks)` |
| 2D block 的 x 维度 | 通常让 warp 内线程在 x 方向连续 |
