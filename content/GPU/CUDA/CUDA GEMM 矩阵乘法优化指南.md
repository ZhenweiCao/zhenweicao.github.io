---
aliases:
  - CUDA矩阵乘法优化指南
  - CUDA GEMM 优化
updated: 2026-06-14
tags:
  - gpu-computing
  - cuda-programming
  - gemm-optimization
---
# CUDA GEMM 矩阵乘法优化指南

> 作者：caozhenwei  
> 目标读者：CUDA 初学者 ~ 中高级开发者  
> 涵盖：通用优化手段 + Hopper (H100) + Blackwell (B100/B200) 架构特性

> Canonical：这是当前 vault 中 GEMM 优化的完整主文档，位于 `GPU/CUDA/`。其他笔记涉及 GEMM 优化时应链接回本文，避免重复维护两份长文。

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[GPU 硬件架构背景与编程范式]]
- [[CUDA Kernel 示例：矩阵乘法]]

---

## 目录

1. [GPU 架构基础与 CUDA 编程模型](CUDA%20GEMM%20矩阵乘法优化指南.md#1-gpu-架构基础与-cuda-编程模型)
2. [矩阵乘法问题定义](CUDA%20GEMM%20矩阵乘法优化指南.md#2-矩阵乘法问题定义)
3. [Naive 实现：最基础的 CUDA GEMM](CUDA%20GEMM%20矩阵乘法优化指南.md#3-naive-实现最基础的-cuda-gemm)
4. [内存层次结构与访问模式](CUDA%20GEMM%20矩阵乘法优化指南.md#4-内存层次结构与访问模式)
5. [优化一：Shared Memory Tiling（共享内存分块）](CUDA%20GEMM%20矩阵乘法优化指南.md#5-优化一shared-memory-tiling共享内存分块)
6. [优化二：Bank Conflict 消除](CUDA%20GEMM%20矩阵乘法优化指南.md#6-优化二bank-conflict-消除)
7. [优化三：Warp-Level 并行与 Register Tiling](CUDA%20GEMM%20矩阵乘法优化指南.md#7-优化三warp-level-并行与-register-tiling)
8. [优化四：向量化访存（Vectorized Memory Access）](CUDA%20GEMM%20矩阵乘法优化指南.md#8-优化四向量化访存vectorized-memory-access)
9. [优化五：双缓冲流水线（Double Buffering）](CUDA%20GEMM%20矩阵乘法优化指南.md#9-优化五双缓冲流水线double-buffering)
10. [优化六：Tensor Core 与 WMMA API](CUDA%20GEMM%20矩阵乘法优化指南.md#10-优化六tensor-core-与-wmma-api)
11. [优化七：MMA PTX 指令（低级 Tensor Core 编程）](CUDA%20GEMM%20矩阵乘法优化指南.md#11-优化七mma-ptx-指令低级-tensor-core-编程)
12. [Hopper 架构深度优化](CUDA%20GEMM%20矩阵乘法优化指南.md#12-hopper-架构深度优化)
    - [TMA（Tensor Memory Accelerator）](CUDA%20GEMM%20矩阵乘法优化指南.md#121-tmatensor-memory-accelerator)
    - [Warpgroup MMA（WGMMA）](CUDA%20GEMM%20矩阵乘法优化指南.md#122-warpgroup-mmawgmma)
    - [异步流水线（cp.async + Pipeline 对象）](CUDA%20GEMM%20矩阵乘法优化指南.md#123-异步流水线cpasync--pipeline-对象)
    - [Persistent Kernel 设计](CUDA%20GEMM%20矩阵乘法优化指南.md#124-persistent-kernel-设计)
13. [Blackwell 架构深度优化](CUDA%20GEMM%20矩阵乘法优化指南.md#13-blackwell-架构深度优化)
    - [Blackwell 架构概述](CUDA%20GEMM%20矩阵乘法优化指南.md#131-blackwellsm-100架构概述)
    - [第五代 Tensor Core 与 tcgen05.mma](CUDA%20GEMM%20矩阵乘法优化指南.md#132-第五代-tensor-core-与-tcgen05mma)
    - [Distributed Shared Memory（分布式共享内存）](CUDA%20GEMM%20矩阵乘法优化指南.md#133-distributed-shared-memory分布式共享内存)
    - [Blackwell Block Cluster 改进](CUDA%20GEMM%20矩阵乘法优化指南.md#134-blackwell-block-cluster-改进)
14. [性能分析方法论](CUDA%20GEMM%20矩阵乘法优化指南.md#14-性能分析方法论)
15. [完整优化路径总结](CUDA%20GEMM%20矩阵乘法优化指南.md#15-完整优化路径总结)
16. [原 CUDA 目录文章整合](CUDA%20GEMM%20矩阵乘法优化指南.md#16-原-cuda-目录文章整合)

---

## 1. GPU 架构基础与 CUDA 编程模型

### 1.1 GPU 的并行哲学

GPU 与 CPU 的核心设计差异在于：

| 指标 | CPU (如 Intel Xeon) | GPU (如 H100 SXM) |
|------|--------------------|--------------------|
| 核心数 | 16–128 cores | 16,896 CUDA cores (H100) |
| 时钟频率 | 3–5 GHz | ~1.8 GHz |
| 缓存 | 大（L3 可达 64MB） | 小（L2 约 50MB，无大 L3） |
| 设计目标 | 低延迟单线程 | 高吞吐并行任务 |
| 内存带宽 | ~100–200 GB/s | ~3350 GB/s (H100 HBM3) |

GPU 通过**海量并行线程**来隐藏内存延迟（Latency Hiding），而不是依赖大缓存或高频率。

### 1.2 CUDA 线程层次结构

```
Grid（整个 Kernel 的工作）
  └── Block（线程块，调度到一个 SM）
        └── Warp（32 个线程，GPU 最小调度单位）
              └── Thread（单个执行单元）
```

**关键概念：**

- **Thread**：最小执行单元，有私有寄存器。
- **Warp**：32 个线程，**物理上同步执行**（SIMT，Single Instruction Multiple Threads）。同一 Warp 内的线程在同一时刻执行同一条指令，但作用于不同数据。
- **Block**：可以包含最多 1024 个线程，调度到同一个 SM（Streaming Multiprocessor）。Block 内线程可通过**共享内存（Shared Memory）**通信和同步（`__syncthreads()`）。
- **Grid**：所有 Block 的集合，构成整个 Kernel 的工作。

### 1.3 SM（Streaming Multiprocessor）内部结构

以 H100 的 SM 为例：

```
SM (Streaming Multiprocessor)
├── 4x Warp Schedulers（每个 SM 有 4 个 Warp 调度器）
├── 128x CUDA Cores（FP32/INT32）
├── 4x Tensor Core（第四代，支持 FP16/BF16/FP8/INT8/TF32）
├── 256KB Shared Memory / L1 Cache（可配置）
├── 64KB Register File（每个 SM 约 65536 个 32-bit 寄存器）
└── L1 Cache（与 Shared Memory 共享物理存储）
```

H100 SXM5 共有 **132 个 SM**。

### 1.4 内存层次结构总览

```
访问延迟（从低到高）            容量（从大到小）
                                            
寄存器 (Register)  ~1 cycle               ~256KB / SM
共享内存 (Shared Memory) ~20–30 cycles    ~228KB / SM (H100)
L1 Cache          ~20–30 cycles           与 Shared Memory 共享
L2 Cache          ~100–200 cycles         ~50MB (H100)
HBM (全局内存)    ~400–700 cycles         ~80GB (H100)
NVLink / PCIe     ~微秒级                 其他 GPU / CPU
```

---

## 2. 矩阵乘法问题定义

### 2.1 GEMM（General Matrix Multiplication）

标准 GEMM 定义：

```
C = α × A × B + β × C
```

其中：
- A: M × K 矩阵
- B: K × N 矩阵  
- C: M × N 矩阵

计算量：O(M × N × K × 2)（每个输出元素需要 K 次乘加，即 2K 次浮点操作）

**典型规模**（深度学习场景）：
- M=4096, N=4096, K=4096 → 约 137 GFLOPs
- 在 H100 上（FP16 峰值约 989 TFLOPs），理论执行时间 < 0.15 ms

### 2.2 计算密度与 Roofline Model

**算术强度（Arithmetic Intensity）**：
```
AI = FLOPs / Bytes_moved
```

对于 GEMM：
- FLOPs = 2 × M × N × K
- 内存访问（理想情况）= (M×K + K×N + M×N) × sizeof(float)
- 当 M=N=K=4096，FP32：AI ≈ 2730 / 192 ≈ **1365 FLOPs/Byte**

H100 SXM5 的 HBM 带宽峰值约 3.35 TB/s，**FP16 Tensor Core dense 峰值 989 TFLOPS（sparse 1,979 TFLOPS）**：
```
Roofline 转折点（dense） = 989 TFLOPs / 3.35 TB/s ≈ 295 FLOPs/Byte
Roofline 转折点（sparse） = 1,979 TFLOPs / 3.35 TB/s ≈ 591 FLOPs/Byte
```

> 算力数字以 [[NVIDIA GPU 架构与规格]] §"Tensor Core 代际：硬件原生形状 vs 算力峰值" 为唯一来源；其他文档（包括本文）涉及具体 TFLOPS 时必须标注 dense / sparse 口径，并指出对应硬件型号。NCU 默认报 dense flops，sparse 路径需单独走 `__nv_sparse_meta` 体系。

GEMM 的算术强度远超转折点，因此 GEMM 是**计算密集型**任务，优化目标是让 Tensor Core 持续满负荷运行。

---

## 3. Naive 实现：最基础的 CUDA GEMM

### 3.1 代码实现

```cuda
// Naive GEMM：每个线程计算 C 的一个元素
__global__ void gemm_naive(
    float* A, float* B, float* C,
    int M, int N, int K)
{
    // 计算当前线程负责的输出位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // 遍历 K 维度，完成点积
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 调用配置
void launch_naive(float* A, float* B, float* C, int M, int N, int K) {
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    gemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
```

### 3.2 性能瓶颈分析

**问题：每次访问 A 和 B 都直接从 HBM（全局内存）读取。**

设 M=N=K=1024，分析内存流量：
- 矩阵 A: 共 M×N 个线程，每个线程计算 C 的一个元素，需读 A 的一整行（K 个元素）。A 的同一行会被 N 个线程（同行不同列）各读一遍，即 A 的每一行被重复读 **N 次**，总读取量 = M × K × N × 4B = 1024 × 1024 × 1024 × 4B ≈ 4 GB
- 矩阵 B: 类似地，B 的同一列会被 M 个线程（同列不同行）各读一遍，即 B 的每一列被重复读 **M 次**，总读取量 = K × N × M × 4B ≈ 4 GB
- 实际只需各读一次 = 4 MB（A）+ 4 MB（B）
- **内存流量放大 1024 倍！**

等效算术强度：2×1024³ FLOPs / 8 GB ≈ **0.27 FLOPs/Byte**（远低于内存带宽极限）

---

## 4. 内存层次结构与访问模式

### 4.1 全局内存访问合并（Coalesced Access）

GPU 的内存控制器以 **128 字节**（Cache Line 大小）为单位访问内存。一个 Warp（32 线程）同时访问内存时，如果 32 个地址落在连续的 128 字节内，只需 1 次内存事务（最优）；否则可能需要多次。

**合并访问示意图：**

```
连续访问（合并）：Warp 中线程 i 访问地址 base + i * 4
Thread:  0   1   2   3  ... 31
Address: 0   4   8   12 ... 124  ← 1次 128B 内存事务 ✓

跨步访问（非合并）：线程 i 访问地址 base + i * 128
Thread:  0    1    2    3   ... 31
Address: 0   128  256  384 ...    ← 32次内存事务 ✗
```

**对 Naive GEMM 的影响：**

读取 B 矩阵时：`B[k * N + col]`
- col 随线程 x 方向递增，连续线程读连续地址 ✓（合并）

读取 A 矩阵时：`A[row * K + k]`
- 同一 Warp 内 32 个线程的 threadIdx.y 相同，因此 row 相同，k 也相同
- 同一 Warp 内所有线程读取 A 的**同一个地址**，触发广播（Broadcast）
- 广播虽然不会串行化，但 32 个线程只取 4 字节，128B Cache Line 利用率极低（1/32）

### 4.2 共享内存的角色

Shared Memory（SMEM）是片上高速存储，延迟约 20-30 cycles（vs HBM 的 400-700 cycles），且**带宽高出约 100 倍**。

设计思路：**将数据从 HBM 搬到 SMEM，多次复用后再搬走** → 降低 HBM 访问量。

---

## 5. 优化一：Shared Memory Tiling（共享内存分块）

### 5.1 核心思想

将 M × K × N 的大矩阵乘法分解为多个 **BM × BK × BN** 的小块乘法：

```
C[BM×BN] = Σ_k A[BM×BK] × B[BK×BN]
```

Block 内所有线程协作将 A 的一块和 B 的一块加载到 Shared Memory，然后每个线程从 SMEM 中读取数据完成计算。每块数据被**复用 BM 或 BN 次**，大幅减少 HBM 访问。

### 5.2 代码实现

```cuda
template<typename T, typename AccT, int BM, int BN, int BK>
__global__ void gemm_smem_tiled(
    // const T* __restrict__ 解释：
    // 1. const 保证不修改输入数据。
    // 2. __restrict__ (或 C99 的 restrict) 向编译器承诺这些指针不会产生“别名”（Aliasing），
    //    即 A、B、C 指向的内存区域互不重叠。这允许编译器执行更激进的指令重排和缓存优化
    //    （例如，不用每次写完 C 都重新从内存读 A/B，并且可以直接使用只读缓存）。
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K)
{
    // 隐式限制提示：
    // 1. 本代码在“计算阶段”仍然假设一个线程计算一个输出元素，
    //    因此必须满足：blockDim.y == BM 且 blockDim.x == BN。
    // 2. 对于最初未修改的数据加载逻辑，甚至要求 blockDim.x == BK 和 blockDim.y == BK（即 BM=BN=BK=线程块边长）。
    // 3. 这里为加载数据引入了一维化（flatten）循环，解耦了对 BK 的大小限制，
    //    只要总线程数足够或通过循环，任何尺寸的 (BMxBK) Tile 都可以被正确加载。
    
    // 声明共享内存 tile
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // 当前 Block 负责计算 C 的 [by*BM : (by+1)*BM, bx*BN : (bx+1)*BN] 子块
    int row = by * BM + ty;
    int col = bx * BN + tx;
    
    // 拍平线程 ID 用于通用化数据搬运
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;
    
    AccT acc = static_cast<AccT>(0);
    
    // 沿 K 维度分块迭代
    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        // ---- 协作加载 A 的 tile 到 SMEM (通用循环加载) ----
        for (int i = tid; i < BM * BK; i += total_threads) {
            int load_row = i / BK;
            int load_col = i % BK;
            int global_row = by * BM + load_row;
            int global_col = tile * BK + load_col;
            if (global_row < M && global_col < K) {
                As[load_row][load_col] = A[global_row * K + global_col];
            } else {
                As[load_row][load_col] = static_cast<T>(0); // 越界补零
            }
        }
        
        // ---- 协作加载 B 的 tile 到 SMEM (通用循环加载) ----
        for (int i = tid; i < BK * BN; i += total_threads) {
            int load_row = i / BN;
            int load_col = i % BN;
            int global_row = tile * BK + load_row;
            int global_col = bx * BN + load_col;
            if (global_row < K && global_col < N) {
                Bs[load_row][load_col] = B[global_row * N + global_col];
            } else {
                Bs[load_row][load_col] = static_cast<T>(0); // 越界补零
            }
        }
        
        // 确保所有线程都完成了数据加载
        __syncthreads();
        
        // ---- 计算当前 tile 的点积 ----
        // 只有当当前线程负责的输出坐标落在 M/N 内时，才需要进行计算
        if (row < M && col < N) {
            for (int k = 0; k < BK; k++) {
                acc += static_cast<AccT>(As[ty][k]) * static_cast<AccT>(Bs[k][tx]);
            }
        }
        
        // 确保所有线程完成计算后再加载下一个 tile
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = static_cast<T>(acc);
    }
}
```

### 5.3 数据复用分析

设 BM=BN=BK=16，Block 有 16×16=256 个线程：

| 数据 | 从 HBM 读取次数 | 从 SMEM 读取次数 |
|------|----------------|------------------|
| A tile (BM×BK) | 1次 | BN 次（tile 内每行被 BN 个列方向线程各读一遍）|
| B tile (BK×BN) | 1次 | BM 次（tile 内每列被 BM 个行方向线程各读一遍）|

- A 的 HBM 访问减少为 Naive 的 **1/BN**（每份 A tile 被 BN 列线程复用）
- B 的 HBM 访问减少为 Naive 的 **1/BM**（每份 B tile 被 BM 行线程复用）
- BM=BN=16 时，两者均降低至原来的 **1/16**！

### 5.5 特殊情况：BM = BN = BK（方形 Tile）

当 BM=BN=BK=BLOCK_SIZE 时，线程块恰好是 BLOCK_SIZE×BLOCK_SIZE 个线程，且 A tile、B tile 都是方阵，可以做一个更简洁的实现：**每个线程负责加载一个 A 元素和一个 B 元素**（1:1:1 对应），省去通用循环，代码更直观，也是教学中最常见的形式。

```cuda
template<typename T, typename AccT, int BLOCK_SIZE>
__global__ void gemm_smem_tiled_square(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K)
{
    // 线程块配置：dim3(BLOCK_SIZE, BLOCK_SIZE)
    // 每个线程负责 C 的一个元素，同时负责加载一个 A 元素和一个 B 元素
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];  // A tile: BM×BK = BLOCK_SIZE×BLOCK_SIZE
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];  // B tile: BK×BN = BLOCK_SIZE×BLOCK_SIZE

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 当前线程负责的输出元素全局坐标
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    AccT acc = static_cast<AccT>(0);

    // 沿 K 维度分块迭代，每次步进 BLOCK_SIZE
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {

        // 每个线程加载一个 A 元素：A[row][tile*BLOCK_SIZE + tx]
        // ty 对应 A 的行（0..BLOCK_SIZE-1），tx 对应 A 的列（K 方向）
        int a_col = tile * BLOCK_SIZE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col]
                                             : static_cast<T>(0);

        // 每个线程加载一个 B 元素：B[tile*BLOCK_SIZE + ty][col]
        // ty 对应 B 的行（K 方向），tx 对应 B 的列（0..BLOCK_SIZE-1）
        int b_row = tile * BLOCK_SIZE + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col]
                                             : static_cast<T>(0);

        __syncthreads();

        // 用 SMEM 中的 tile 做点积
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += static_cast<AccT>(As[ty][k]) * static_cast<AccT>(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = static_cast<T>(acc);
    }
}
```

**与通用版本的对比：**

| 特性 | 通用版（BM/BN/BK 独立）| 方形版（BM=BN=BK）|
|:--|:--|:--|
| 线程块大小 | `dim3(BN, BM)`，与 BK 无关 | `dim3(BLOCK_SIZE, BLOCK_SIZE)` |
| 数据加载方式 | tid 循环，支持任意 BK | 每线程加载 1 个 A + 1 个 B 元素 |
| 代码复杂度 | 较高（需处理循环步长）| 简洁，逻辑清晰 |
| 限制 | BM×BN 须 ≥ BM×BK 和 BK×BN（线程数需够用）| 必须 BM=BN=BK，线程块为方形 |
| SMEM 用量 | `(BM×BK + BK×BN) × sizeof(T)` | `2 × BLOCK_SIZE² × sizeof(T)` |

**调用方式：**

```cuda
constexpr int BLOCK_SIZE = 16;
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
             (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
gemm_smem_tiled_square<float, float, BLOCK_SIZE>
    <<<gridDim, blockDim>>>(A, B, C, M, N, K);
```



两个同步点各有职责：
1. **加载后同步**：防止某些线程还没加载完数据就被其他线程使用。
2. **计算后同步**：防止某些线程提前开始下一轮覆盖 SMEM，而其他线程还在读当前数据。

---

## 6. 优化二：Bank Conflict 消除

### 6.1 什么是 Bank Conflict

Shared Memory 物理上被划分为 **32 个 Bank**（每个 bank 宽度 4 字节）。  
同一个 Warp 中的多个线程如果访问**同一个 Bank 内的不同地址**，就发生 Bank Conflict，这些访问被**串行化**。

```
Shared Memory 布局（以 float 为例）：
Bank:   0    1    2    3  ...  31   0    1  ...
Index:  0    1    2    3  ...  31   32   33 ...
```

**导致 Bank Conflict 的情形：**

```cuda
__shared__ float As[16][16];
// 线程 i 访问 As[i][j]（同一列）
// 列索引 j 相同时，行索引 i 步长为 1，Bank = i % 32
// 若 i 连续，不同线程访问不同 bank → 无冲突 ✓

// 但如果 16 列的矩阵按列方向访问（转置）：
// 线程 i 访问 As[k][i]（同一行）
// Bank = (k*16 + i) % 32
// 若 16 是 32 的因数，可能导致 2-way 或更多 Bank Conflict！
```

### 6.2 解决方案：Padding

```cuda
// 添加 1 个 padding 列，打破 bank 对齐
__shared__ float As[BK][BM + 1];  // +1 padding
__shared__ float Bs[BK][BN + 1];  // +1 padding
```

**Padding 原理：** 增加 1 列后，矩阵行偏移从 `BM * 4B` 变为 `(BM+1) * 4B`，原本对齐到同一 Bank 的元素错开了 1 个位置，有效避免冲突。

### 6.3 另一种方案：数据重排（Swizzle）

更高级的方法是在**写入 Shared Memory 时**对物理地址做 XOR（异或）变换。这**不修改全局内存（HBM）的布局**，仅仅改变数据在 SMEM 中的存放位置：

```cuda
// 假设有 As[BM][BK]，为了避免按列读取时的 Bank Conflict
// 存储时（从 HBM 读入 SMEM）：
// 逻辑坐标为 (row, col) 的元素，存放在 SMEM 的物理坐标为 (row, col ^ (row % 8))
int physical_col = col ^ (row % 8);
As[row][physical_col] = A_global[...];

// 读取时（从 SMEM 读到寄存器参与计算）：
// 按照完全相同的 XOR 公式计算物理地址，即可取回正确的元素
int physical_col_read = col ^ (row % 8);
float a_reg = As[row][physical_col_read];
```

**为什么是 XOR（异或）？**
1. **错位效果**：`col ^ (row % 8)` 会根据行号的不同，把同一列的数据“斜向”打散到不同的 Bank 里。当一个 Warp 的 32 个线程去读**同一逻辑列**时，它们实际上访问的是 SMEM 中**不同的物理列**（不同的 Bank），从而完美消除排队（Bank Conflict）。
2. **对称性**：XOR 操作具有自反性（`A ^ B ^ B = A`），所以存和取用的是同一个极简公式，不需要写两套寻址逻辑，且 GPU 计算 XOR 只需要 1 个周期。
3. **零空间浪费**：相比 Padding 需要在每行末尾多分配内存，Swizzle 完全不增加 SMEM 占用。

**开发者感知程度（Ampere vs Hopper）：**
- **Ampere 及以前（软件级 Swizzle）**：开发者**强感知**。必须在 CUDA 代码中手动写出 `row ^ ...` 的寻址逻辑，计算 `physical_col`，这增加了指令开销。
- **Hopper 及以后（硬件级 Swizzle）**：开发者**弱感知**。Hopper 引入了 TMA（Tensor Memory Accelerator）硬件单元。在 Host 端配置 TMA 描述符时，只需开启 Swizzle 标志（如 128-byte swizzle）。TMA 硬件在把数据从 HBM 搬到 SMEM 时会**自动在硬件层面完成 XOR 变换**；同理，WGMMA 计算指令去读 SMEM 时，硬件也会自动反向解析。整个过程对 Kernel 内的线程指令是透明的，真正实现了“零软件开销”。
CUTLASS 库和 CuTe 工具包大量使用并封装了 Swizzle 布局。

---

## 7. 优化三：Warp-Level 并行与 Register Tiling

### 7.1 问题：仅 SMEM Tiling 不够

当 BM=BN=BK=16 时，Block 有 256 线程，每个线程只计算 1 个输出元素（1 次乘加）。  
这导致：
- **指令并行度低**：每个线程太少工作，Warp 调度开销大。
- **Register 利用率低**：寄存器是最快的存储，没有充分利用。

### 7.2 Register Tiling（每线程计算多个输出）

让每个线程负责 **TM × TN** 个输出元素（通常 TM=TN=4 或 8）：

```
Block 大小：BM × BN = 128 × 128 （总输出）
每线程输出：TM × TN = 8 × 8 = 64 个元素
线程数：(BM/TM) × (BN/TN) = 16 × 16 = 256 线程
```

```cuda
template<typename T, typename AccT, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register_tiled(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K)
{
    // 前置条件说明：
    // 本实现的线程分配强假设 Block 内的总线程数 == (BM/TM) * (BN/TN)
    // 且线程块配置为 dim3(BN/TN, BM/TM)
    
    __shared__ T As[BK][BM];
    __shared__ T Bs[BK][BN];
    
    // 线程在 Block 内的位置
    int thread_row = threadIdx.y;  // 取决于线程块配置为 dim3(BN/TN, BM/TM)
    int thread_col = threadIdx.x;  
    
    // 或者如果你用的是 1D Block dim3((BN/TN) * (BM/TM))，那么应这样解算：
    // int thread_row = threadIdx.x / (BN / TN);
    // int thread_col = threadIdx.x % (BN / TN);
    
    // 每个线程维护 TM × TN 个寄存器累加器
    AccT acc[TM][TN] = {static_cast<AccT>(0)};
    T a_reg[TM];   // A 的寄存器缓存
    T b_reg[TN];   // B 的寄存器缓存
    
    // 注意：load_tile 需要考虑 M/N/K 维度越界（类似前述 SMEM tiling 的补零逻辑）
    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        // 协作加载 tile 到 SMEM（省略边界处理代码，实际需判断行列越界并补零）
        load_tile_A_to_smem(A, As, ...);
        load_tile_B_to_smem(B, Bs, ...);
        __syncthreads();
        
        // 计算 TM × TN 的输出
        for (int k = 0; k < BK; k++) {
            // 从 SMEM 加载到寄存器
            for (int m = 0; m < TM; m++)
                a_reg[m] = As[k][thread_row * TM + m];
            for (int n = 0; n < TN; n++)
                b_reg[n] = Bs[k][thread_col * TN + n];
            
            // 外积：TM × TN 次乘加
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += static_cast<AccT>(a_reg[m]) * static_cast<AccT>(b_reg[n]);
        }
        __syncthreads();
    }
    
    // 获取当前 Block 计算子块的全局起点的偏置
    int by = blockIdx.y;
    int bx = blockIdx.x;
    
    // 写回结果（需要检查输出元素是否越界 M 和 N）
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int global_row = by * BM + thread_row * TM + m;
            int global_col = bx * BN + thread_col * TN + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = static_cast<T>(acc[m][n]);
            }
        }
    }
}
```

### 7.3 特殊情况：BM = BN = BK（方形 Tile，教学常见形式）

令 BM=BN=BK=BLOCK_SIZE，每个线程计算 TM×TN 个输出。此时线程块为 `(BLOCK_SIZE/TN, BLOCK_SIZE/TM)` 个线程，数据加载与 5.5 节的方形 Tile 完全一致——**每个线程恰好负责加载 TM 个 A 元素和 TN 个 B 元素**，无需通用循环。

这是 Simon Boehm 的 [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMul) 等主流教学博客中采用的标准形式。

```cuda
template<typename T, typename AccT, int BLOCK_SIZE, int TM, int TN>
__global__ void gemm_register_tiled_square(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K)
{
    // 线程块配置：dim3(BLOCK_SIZE / TN, BLOCK_SIZE / TM)
    // 线程数 = (BLOCK_SIZE/TM) * (BLOCK_SIZE/TN)
    // 每个线程计算 TM × TN 个输出，负责加载 TM 个 A 元素 + TN 个 B 元素

    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];  // BK × BM = BLOCK_SIZE × BLOCK_SIZE
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];  // BK × BN = BLOCK_SIZE × BLOCK_SIZE

    // 线程在 Block 内的逻辑位置（以"输出子块"为单位）
    int thread_row = threadIdx.y;  // 范围 [0, BLOCK_SIZE/TM)
    int thread_col = threadIdx.x;  // 范围 [0, BLOCK_SIZE/TN)

    int by = blockIdx.y, bx = blockIdx.x;

    // 每个线程的累加器寄存器
    AccT acc[TM][TN] = {};
    T a_reg[TM];
    T b_reg[TN];

    // 数据加载时，把线程 ID 重新映射为"扁平 tile 加载者"
    // 加载 A tile（BLOCK_SIZE × BLOCK_SIZE 个元素），每个线程加载 TM × TN 个
    // 用 (thread_row * TN + thread_col) 作为扁平 ID，步长为总线程数
    // load_tid：Block 内的扁平线程 ID，等价于 threadIdx.y * blockDim.x + threadIdx.x
    // 线程数 = (BLOCK_SIZE/TM) × (BLOCK_SIZE/TN)，而 tile 元素数 = BLOCK_SIZE²
    // 每个线程需要加载 TM×TN 个元素，用 load_tid 做步长循环均匀分配加载任务
    int load_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = (BLOCK_SIZE / TM) * (BLOCK_SIZE / TN);

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {

        // 协作加载 A tile：每个线程加载 TM × TN 个元素（用扁平循环）
        for (int i = load_tid; i < BLOCK_SIZE * BLOCK_SIZE; i += total_threads) {
            int r = i / BLOCK_SIZE;
            int c = i % BLOCK_SIZE;
            int global_row = by * BLOCK_SIZE + r;
            int global_col = tile * BLOCK_SIZE + c;
            As[r][c] = (global_row < M && global_col < K)
                       ? A[global_row * K + global_col]
                       : static_cast<T>(0);
        }

        // 协作加载 B tile
        for (int i = load_tid; i < BLOCK_SIZE * BLOCK_SIZE; i += total_threads) {
            int r = i / BLOCK_SIZE;
            int c = i % BLOCK_SIZE;
            int global_row = tile * BLOCK_SIZE + r;
            int global_col = bx * BLOCK_SIZE + c;
            Bs[r][c] = (global_row < K && global_col < N)
                       ? B[global_row * N + global_col]
                       : static_cast<T>(0);
        }

        __syncthreads();

        // 每个 k 步：从 SMEM 取 TM 个 A 元素 + TN 个 B 元素，做 TM×TN 外积
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for (int m = 0; m < TM; m++)
                a_reg[m] = As[k][thread_row * TM + m];
            for (int n = 0; n < TN; n++)
                b_reg[n] = Bs[k][thread_col * TN + n];

            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += static_cast<AccT>(a_reg[m]) * static_cast<AccT>(b_reg[n]);
        }

        __syncthreads();
    }

    // 写回
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int global_row = by * BLOCK_SIZE + thread_row * TM + m;
            int global_col = bx * BLOCK_SIZE + thread_col * TN + n;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = static_cast<T>(acc[m][n]);
        }
    }
}
```

**调用方式（BLOCK_SIZE=64, TM=TN=4）：**

```cuda
constexpr int BLOCK_SIZE = 64, TM = 4, TN = 4;
// 线程块：(64/4, 64/4) = (16, 16) = 256 线程
// 每个线程计算 4×4 = 16 个输出元素
dim3 blockDim(BLOCK_SIZE / TN, BLOCK_SIZE / TM);
dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
             (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
gemm_register_tiled_square<float, float, BLOCK_SIZE, TM, TN>
    <<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

**关键设计解读：**

- `load_tid = threadIdx.y * blockDim.x + threadIdx.x`：Block 内的扁平线程 ID。线程数为 `(BLOCK_SIZE/TM) × (BLOCK_SIZE/TN)`，而 tile 元素数为 `BLOCK_SIZE²`，每个线程需要加载 `TM×TN` 个元素。用 `load_tid` 做步长为 `total_threads` 的循环，把加载任务均匀分配给所有线程
- `As[k][thread_row * TM + m]`：同一列的 TM 个 A 元素在 SMEM 中地址连续，线程内顺序读取，无 Bank Conflict
- `Bs[k][thread_col * TN + n]`：同理，TN 个 B 元素连续
- 外积循环 `acc[m][n] += a_reg[m] * b_reg[n]` 全在寄存器中完成，不碰 SMEM
- SMEM 大小 = `2 × BLOCK_SIZE² × sizeof(T)`，BLOCK_SIZE=64 时约 32 KB，在 H100 的 228 KB SMEM 内远未饱和

### 7.4 算术强度分析（Register Tiling 后）

每个线程每轮 tile（BK 次迭代）：
- SMEM 读取：TM + TN 次（加载 a_reg 和 b_reg）
- 计算：TM × TN × BK 次乘加

**SMEM 算术强度** = `(TM × TN × BK) / (TM + TN)` FLOPs/SMEM_access  
当 TM=TN=8, BK=16：= `8×8×16 / (8+8)` = **64 FLOPs/SMEM_access**

这意味着从 SMEM 读来的每个数据被复用 64 次，显著减少了 SMEM 带宽压力。

---

## 8. 优化四：向量化访存（Vectorized Memory Access）

### 8.1 原理

GPU 支持 128-bit 向量加载指令（`LDG.128`、`float4`），一次加载 4 个 float（或 8 个 float16）。  
相比逐元素加载，向量加载：
- 指令数减少 4 倍
- 提升内存带宽利用率（减少指令开销占比）
- 确保 128-byte Cache Line 对齐访问

### 8.2 代码实现

```cuda
// 使用 float4 向量加载
// 每次加载 4 个 float（16 字节），对应 LDG.128 指令

// 加载全局内存 → 共享内存时使用向量化
float4 tmp = reinterpret_cast<float4*>(A + row * K + tile * BK)[thread_col];
As[ty][tx * 4 + 0] = tmp.x;
As[ty][tx * 4 + 1] = tmp.y;
As[ty][tx * 4 + 2] = tmp.z;
As[ty][tx * 4 + 3] = tmp.w;

// 或者直接使用 __ldg（通过只读缓存加载）
float4 tmp = __ldg(reinterpret_cast<const float4*>(&A[...]));
```

### 8.3 对齐要求

- 数据起始地址必须对齐到 16 字节（float4）或 8 字节（float2）
- 矩阵列维度最好是 4 的倍数（float4）或 8 的倍数（float16×8）
- **边界尾部（Tail）处理**：如果最后剩余不足 `4` 个元素，就无法使用 `float4`。实际生产库（如 CUTLASS）会采用对底层内存（分配时）额外填充，或者使用单独的标量加载代码块专门处理这些不满足对齐的部分（Predicated Load）。

---

## 9. 优化五：双缓冲流水线（Double Buffering）

### 9.1 计算与访存的重叠

到目前为止，每个 tile 的流程是：
```
加载数据到 SMEM → 同步 → 计算 → 同步 → 加载下一个 tile
```

数据加载和计算**串行执行**，浪费了 GPU 的并行能力。

**双缓冲（Double Buffering / Software Pipelining）**思想：
```
加载 tile[0] → 同步 → 计算 tile[0] + 加载 tile[1] → 同步 → 计算 tile[1] + 加载 tile[2] → ...
```
即当前 tile 的计算与下一个 tile 的加载**并行进行**。

### 9.2 实现方式

#### 方式一：纯 SMEM 双缓冲

```cuda
__shared__ float As[2][BK][BM];  // 2 个缓冲区
__shared__ float Bs[2][BK][BN];

int cur = 0;  // 当前使用的缓冲区
// 预加载第一个 tile
load_to_smem(A, B, As[0], Bs[0], tile=0);
__syncthreads();

for (int tile = 1; tile < num_tiles; tile++) {
    // 异步加载下一个 tile 到备用缓冲区
    load_to_smem(A, B, As[1-cur], Bs[1-cur], tile);
    
    // 同时计算当前缓冲区的 tile
    compute_tile(As[cur], Bs[cur], acc);
    
    __syncthreads();  // 等待加载完成 + 计算完成
    cur = 1 - cur;   // 切换缓冲区
}
compute_tile(As[cur], Bs[cur], acc);  // 计算最后一个 tile
```

#### 方式二：利用 `cp.async`（Ampere+）

从 Ampere 架构（A100）开始，CUDA 支持 `cp.async` 指令，可以**异步地从全局内存复制到共享内存**，不需要经过寄存器，且不阻塞当前线程的计算：

```cuda
// cp.async：异步复制，返回后数据不一定就绪
#include <cuda/pipeline>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__shared__ float As[2][BK][BM];

// 异步加载指令
cuda::memcpy_async(thread_rank, As[next], src_ptr, BM * BK * sizeof(float), pipeline);

// 发出屏障
pipeline.producer_commit();

// ... 计算当前 tile（使用 As[cur]）...

// 等待异步加载完成
pipeline.consumer_wait();
```

`cp.async` 的优势：
- 数据不经过寄存器，节省寄存器压力
- SM 可在等待 DRAM 的同时继续执行计算指令

---

## 10. 优化六：Tensor Core 与 WMMA API

### 10.1 Tensor Core 是什么

Tensor Core 是 NVIDIA 从 Volta 架构（2017）引入的**专用矩阵乘法硬件单元**，用于高效执行小型矩阵乘加操作：

```
D = A × B + C
```

每个 Tensor Core 每个 clock 可完成多个 FMA（Fused Multiply-Add），具体形状和吞吐随代际变化。

> **Tensor Core 硬件原生 MMA 形状（V100 `m8n8k4`、Ampere `m16n8k16`、Hopper wgmma `m64nNk16/32`、Blackwell `tcgen05.mma`）与各代 dense / sparse 峰值算力，统一以 [[NVIDIA GPU 架构与规格]] §"Tensor Core 代际：硬件原生形状 vs 算力峰值" 为唯一来源**。本文不在此处再维护一份代际表，避免出现混淆 dense / sparse、把 WMMA API tile 当成硬件原生形状（如把 V100 写成 `16×16×16`）的错误。

要点提醒：

- V100 硬件原生 mma 是 `m8n8k4`，`16×16×16` 是 WMMA API tile 形状；
- A100 FP16 dense 156 TFLOPS / sparse 312 TFLOPS；H100 SXM5 FP16 dense 989 TFLOPS / sparse 1,979 TFLOPS；
- Ampere A100 不支持 FP8 Tensor Core，FP8 从 Hopper / Ada 之后进入主线；
- Blackwell 引入 FP4 / FP6 / 块缩放（block scaling），见下文 §13.2。

### 10.2 WMMA（Warp Matrix Multiply-Accumulate）API

WMMA 是 CUDA 提供的 Warp 级别 Tensor Core 编程接口（`nvcuda::wmma`）：

```cuda
#include <mma.h>
using namespace nvcuda;

// 每个 Warp 计算一个 16×16×16 的矩阵乘法
__global__ void gemm_wmma(half* A, half* B, float* C, int M, int N, int K) {
    // 声明 Fragment（Warp 中 32 个线程共同持有的矩阵）
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int warp_M = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_N = blockIdx.x * 16;
    
    for (int k = 0; k < K; k += 16) {
        // 从全局内存加载 fragment（数据按 Warp 所需布局分布到 32 个线程）
        wmma::load_matrix_sync(a_frag, A + warp_M * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_N, N);
        
        // 执行 Tensor Core 矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
// 写回结果
    wmma::store_matrix_sync(C + warp_M * N + warp_N, c_frag, N, wmma::mem_row_major);
}

// **特别注意**：
// 这个基础 WMMA 示例严格假设矩阵维度 M、N、K 都是 16 的倍数（因为 fragment 大小固定为 16x16x16）。
// 当尺寸不满足时，要么需要在内存中提前 padding，要么必须使用更小/可变大小的 mma 变种，处理尾部逻辑极为复杂。
```

**注意**：WMMA API 是高级封装，Tensor Core 的数据布局由驱动自动处理，但灵活性有限。

---

## 11. 优化七：MMA PTX 指令（低级 Tensor Core 编程）

### 11.1 为什么需要 PTX 级别

WMMA API 的问题：
- 数据布局不透明，难以精确控制
- 无法充分利用寄存器复用
- Hopper 新特性（WGMMA）没有 WMMA 对应接口

PTX `mma.sync` 指令直接对应 Tensor Core 硬件操作，允许程序员精确控制数据在寄存器中的分布。

### 11.2 Ampere 时代的 MMA 指令

```cuda
// PTX inline assembly: 16x8x16 的 FP16 mma 操作
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, "      // D (output accumulator, 4 个 float32)
    "{%4,%5,%6,%7}, "      // A (16 个 FP16 = 4 个 32-bit 寄存器)
    "{%8,%9}, "            // B (8 个 FP16 = 2 个 32-bit 寄存器)
    "{%10,%11,%12,%13};"   // C (input accumulator, 4 个 float32)
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3)
);
```

每条 `mma.sync.m16n8k16` 指令由整个 Warp 协作完成，数据分散在 32 个线程的寄存器中。

### 11.3 数据布局（Register Layout）

以 `mma.sync.m16n8k16` 为例，**A 矩阵的 256 个 FP16 元素**在 Warp 中的分布：

```
Thread  0: 持有 A 的 (0,0), (0,1), (8,0), (8,1), (0,8), (0,9), (8,8), (8,9) 等
Thread  1: 持有 A 的 (0,2), (0,3), ...
Thread 16: 持有 A 的 (1,0), (1,1), ...
```

这种复杂的分布模式是 PTX 编程的难点，也是 CUTLASS 等库抽象掉的部分。

---

## 12. Hopper 架构深度优化

### 12.1 Hopper 架构关键变化

H100（Hopper, SM 9.0）引入的革命性特性：

| 特性 | 描述 |
|------|------|
| **TMA（Tensor Memory Accelerator）** | 专用硬件单元，独立处理 SMEM ↔ HBM 的数据移动 |
| **WGMMA（Warpgroup MMA）** | Warpgroup（128 线程）级别的大型 Tensor Core 操作 |
| **FP8 精度** | 新增 E4M3 和 E5M2 格式，Tensor Core 吞吐翻倍 |
| **异步事务屏障（mbarrier）** | 细粒度异步同步原语 |
| **Thread Block Cluster** | 跨 SM 的 Block 分组，共享 DSMEM（Distributed SMEM） |

### 12.2 TMA（Tensor Memory Accelerator）

#### 核心思想

传统方式：CPU（线程）负责计算搬运哪些数据 + 执行搬运  
TMA 方式：线程只需**描述搬运任务**，TMA 硬件单元**独立异步执行**搬运

这使得 SM 可以将全部资源用于计算，消除了数据搬运的软件开销。

#### TMA 描述符（TMA Descriptor）

在 Host 端创建描述符，描述内存布局（多维 tensor 形状、步长、数据类型等）：

```cpp
#include <cute/arch/copy_sm90_tma.hpp>
// 使用 CuTe 库创建 TMA 描述符

// 创建 2D tensor 布局
auto A_layout = make_layout(make_shape(M, K), make_stride(K, 1));
auto A_smem_layout = make_layout(make_shape(BM, BK));

// 创建 TMA 对象（描述如何从全局 tensor 拷贝 tiles 到 SMEM）
auto tma_A = make_tma_copy(SM90_TMA_LOAD{}, A_ptr, A_layout, A_smem_layout, ...);
```

#### TMA 在 Kernel 中的使用

```cuda
// 在 kernel 中，只有 1 个线程负责发起 TMA
if (threadIdx.x == 0) {
    // 使用 cute 描述符发起异步 TMA 加载
    cute::copy(tma_A, 
               tma_A.get_tma_descriptor(),
               make_coord(block_row * BM, tile_k * BK),  // 源坐标
               smem_A,                                    // 目标 SMEM
               mbar                                       // mbarrier 同步对象
    );
}

// 其他线程等待 mbarrier 完成
mbarrier::wait(mbar, phase);
```

**TMA 的优势：**
- 节省约 100+ 个 SM 线程用于数据搬运的开销
- 支持多维 tensor（自动处理 stride、padding）
- 支持 Swizzle（自动优化 SMEM bank 布局）
- 带宽可接近 HBM 理论峰值

**TMA 指令变种**（按 tensor 维数和广播能力）：

| 指令 | 维数 | 说明 |
|------|-----|------|
| `cp.async.bulk` | 1D | 一维线性区域批量异步拷贝 |
| `cp.async.bulk.tensor.1d` | 1D | 走 tensor descriptor 路径的 1D 拷贝 |
| `cp.async.bulk.tensor.2d` | 2D | GEMM A/B tile 最常用 |
| `cp.async.bulk.tensor.3d` | 3D | 常用于 3D conv / attention head |
| `cp.async.bulk.tensor.4d` | 4D | 高维 tensor（batch + spatial） |
| `cp.async.bulk.tensor.5d` | 5D | 上限；维度由 PTX/Driver 版本决定 |
| `cp.async.bulk.tensor.*.multicast` | 1D~5D | **Cluster multicast** 变体：一次 HBM 读取，同时写入 cluster 内多个 CTA 的 SMEM（详见 §13.3 DSMEM） |

### 12.3 Warpgroup MMA（WGMMA）

#### 背景

Ampere 的 MMA：每个 Warp（32 线程）执行 `mma.sync.m16n8k16`  
Hopper 的 WGMMA：4 个 Warp 组成 Warpgroup（128 线程）执行更大的矩阵操作

#### WGMMA 特点

```
WGMMA 操作形状（按精度）：
  FP16/BF16: m=64 (固定), n ∈ {8, 16, 24, ..., 256}（8 的倍数）, k=16
  FP8 / INT8: m=64,                n 同上,                       k=32
  TF32:      m=64,                n 同上,                       k=8
```

常用 tile 是 `m64n128k16` 与 `m64n256k16`（写文档时不要把 `m64n256k16` 当作唯一形状）。

**操作数来源（两种变体）**：

- **SS 变体**：A 和 B 都来自 shared memory，通过 matrix descriptor 描述（含 swizzle 模式）；
- **RS 变体**：A 来自寄存器（fragment），B 来自 shared memory，适合把 A fragment 留在寄存器中跨多个 wgmma 复用。

**因此"wgmma 要求 A 和 B 必须来自 Shared Memory"是不准确的**——RS 变体下 A 可以来自寄存器；但 B 始终来自 SMEM。TMA 的重要性来自：无论 SS 还是 RS，B（以及 SS 下的 A）都必须先高效搬到 SMEM。

#### PTX WGMMA 指令

```cuda
// wgmma.mma_async（SS 变体示意）
// 标量参数顺序：scale-d, scale-a, scale-b, trans-a(仅 SS), trans-b(仅 SS)
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
    "{%0,%1,...,%63}, "   // D: 64 个 float32 寄存器
    " %64, "              // A: matrix descriptor（SS 变体）或 fragment 寄存器（RS 变体）
    " %65, "              // B: matrix descriptor（始终 SMEM）
    "1, 1, 1, 0, 0;"      // scale_D, scale_A, scale_B, trans_A, trans_B
    ...
);

// 提交一组 wgmma 指令
asm volatile("wgmma.commit_group.sync.aligned;");

// 等待前 N 组完成
asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(0));
```

> 上面是简化示意，操作数寄存器数量随 N 维变化；详细的标量参数语义、descriptor 字段（swizzle / leading dim / stride）以 PTX ISA `wgmma.mma_async` 章节为准。

#### CuTe 封装（推荐方式）

直接操作 PTX 过于繁琐，CUTLASS 3.x / CuTe 提供了更高层的封装：

```cpp
// 使用 CuTe 的 MMA atom
using MMA_Atom = MMA_Atom<SM90_64x256x16_F32F16F16_SS>;  // 指定 WGMMA 变体
using TiledMMA = TiledMMA<MMA_Atom, ...>;

// 执行矩阵乘法（CuTe 自动生成正确的 WGMMA 指令序列）
cute::gemm(tiled_mma, acc_C, thr_A, thr_B, acc_C);
```

### 12.4 异步流水线（cp.async + Pipeline 对象）

Hopper 使用 **mbarrier（memory barrier）** 替代传统的 `__syncthreads()`，实现更细粒度的异步控制：

```
传统流水线（Ampere）：
[Load tile i] → sync → [Compute tile i] → sync → [Load tile i+1] → ...

Hopper 异步流水线：
[TMA Load tile i]  ─────────────────→ 完成后 signal mbar[i]
                   ↘
                    [WGMMA Compute tile i-1]  当 mbar[i] ready 后开始
                                              ↘
                                               [TMA Load tile i+1] ...
```

**多级流水线（Pipeline Depth）设计：**

```cpp
// 通常使用 4–8 级流水线缓冲
constexpr int STAGES = 4;
__shared__ float As[STAGES][BK][BM];  // 4 份 SMEM 缓冲
__shared__ float Bs[STAGES][BK][BN];
__shared__ cuda::barrier<cuda::thread_scope_block> mbar[STAGES];

// 预填充流水线（填入前 STAGES-1 个 tile）
for (int s = 0; s < STAGES - 1; s++) {
    // 异步加载
    copy_tma_to_smem(tma_A, As[s], tile_k = s, mbar[s]);
}

// 主流水线循环
for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
    int consume_stage = tile_k % STAGES;
    int produce_stage = (tile_k + STAGES - 1) % STAGES;
    
    // 加载下一个 tile（超前 STAGES-1 步）
    copy_tma_to_smem(tma_A, As[produce_stage], tile_k + STAGES - 1, mbar[produce_stage]);
    
    // 等待当前消费 stage 的数据就绪
    mbar[consume_stage].wait(phase[consume_stage]);
    
    // WGMMA 计算
    wgmma_compute(As[consume_stage], Bs[consume_stage], acc);
}
```

### 12.5 Persistent Kernel 设计

#### 背景

传统 GEMM Kernel：每个 CTA（Block）处理固定的一块 C，完成后退出，调度器再分配新 CTA。  
对于大型 GEMM，这种**波次（Wave）调度**会导致：
- 最后一波 CTA 数量不足 SM 数量 → **Tail Effect（尾效应）**，GPU 部分空闲

#### Persistent Kernel 方案

```cpp
// 线程块不退出，持续从全局任务队列中取工作
__global__ void persistent_gemm_kernel(...) {
    // 每个 SM 上的 CTA 持续工作
    int tile_m, tile_n;
    while (get_next_tile(&tile_m, &tile_n)) {  // 原子获取下一个输出 tile
        compute_output_tile(tile_m, tile_n, ...);
    }
}
```

**优势：**
- 消除 Tail Effect
- Block 重用（减少初始化开销）
- 结合 TMA 的 prefetch 可以在 tile 切换时零等待

Hopper 的 **Grid Stream** 和 CUTLASS 3.x 的 **Stream-K** 算法是这一思想的工程实现。

---

## 13. Blackwell 架构深度优化

### 13.1 Blackwell（SM 10.0）架构概述

Blackwell（B100/B200/B300，2024–2025 年）在 Hopper 基础上大幅升级：

| 特性 | Hopper (H100) | Blackwell (B200) |
|------|--------------|------------------|
| FP8 Tensor Core 峰值 | ~1979 TFLOPs | ~4.5 PFLOPs |
| FP4 支持 | ❌ | ✓（FP4 / NVFP4，B200 单 GPU dense ~10 PFLOPS，sparse ~20 PFLOPS） |
| FP8 算力（参考） | dense 1,979 TFLOPS / sparse 3,958 TFLOPS | B200 单 GPU dense ~5 PFLOPS / sparse ~10 PFLOPS |
| HBM 带宽 | 3.35 TB/s | ~8 TB/s（HBM3e，单 GPU 口径） |
| Shared Memory / L1 | 228 KB shared memory opt-in / 256 KB combined L1+SMEM | 具体随 compute capability 和产品形态变化，B200 combined L1+SMEM 仍为 256 KB 口径 |
| GPU-to-GPU 带宽 | NVLink 4 双向 900 GB/s（单 GPU 端口聚合） | NVLink 5 双向 1.8 TB/s（单 GPU 端口聚合） |
| 显存容量 | 80 GB H100 / 141 GB H200 | B200 单 GPU 192 GB HBM3e；rack / DGX 系统口径详见规格表 |
| 第二代 MIG | 7 实例 | 7 实例（单实例资源更大） |
| `tcgen05.mma` / Tensor Memory | ❌ | ✓ |
| 第五代 Tensor Core | ❌ | ✓ |

> 这张表仅作方向对比，不作为产品规格表。Blackwell 单 GPU、Grace Blackwell Superchip、NVL72 和 DGX 系统的规格口径不同，精确数值优先查 [[NVIDIA GPU 架构与规格]]。所有 TFLOPS / PFLOPS **都要标注 dense 还是 sparse**；NVLink/PCIe 带宽**都要标注单向还是双向**，本表的 NVLink 数字均为双向聚合。

### 13.2 第五代 Tensor Core 与 `tcgen05.mma`

#### 从 WGMMA 到 `tcgen05`

Blackwell 的低层主线建议按官方 PTX 的 `tcgen05.mma` / `tcgen05.*` 指令族理解。很多资料会用"新一代 MMA"或"UMMA"概括这类能力，但写文档和查资料时应优先使用官方指令名。

**核心改变：**

1. **Tensor Memory（TMEM）进入主路径**：
   - Blackwell 在 SM 内引入新的片上存储 **Tensor Memory**，**每个 SM 约 256 KB**，与 register file、shared memory/L1 并列；
   - `tcgen05.mma` 的 accumulator / destination 默认位于 Tensor Memory；A/B 操作数可以来自 SMEM descriptor，也可以来自 TMEM；
   - 把大规模 accumulator 的存放从通用寄存器压力中部分解放出来，但也要求 kernel 正确管理 TMEM 地址、descriptor 和等待语义；
   - 配套的搬运指令 **`tcgen05.ld`** / **`tcgen05.st`** 负责在 TMEM 与寄存器之间搬数据；TMEM 的对齐、行/列布局由 descriptor 控制。

2. **CTA group 与 descriptor 更重要**：
   - `tcgen05.mma` 通过 `.cta_group::1` / `.cta_group::2` 描述参与范围；
   - `cta_group::2` 让同 cluster 内两个 CTA 协作完成一次更大的 MMA，是 Blackwell 上提升单次 MMA 吞吐的关键手段；
   - A/B 矩阵通过 Tensor Memory 或 shared memory descriptor 组织，具体形式取决于指令 kind 和 CUTLASS kernel。

3. **FP4/FP6 精度支持**：
   ```
   FP4 (E2M1) 或 NVFP4：4-bit，Tensor Core 吞吐再翻倍
   FP6 (E3M2 / E2M3)：6-bit，精度/吞吐平衡
   FP8 (E4M3 / E5M2)：8-bit，沿用 Hopper
   ```

4. **Block-Scale 量化（MXFP / NVFP4）**：配合 FP4/FP6，引入块级缩放因子。**缩放因子格式由 OCP Microscaling spec 定义为 E8M0（8-bit exponent-only）或 FP8(E4M3)——不是 FP16**；块大小：
   - **MXFP4 / MXFP6 / MXFP8**：每 32 个元素共享 1 个 E8M0 缩放因子（OCP MX 标准）；
   - **NVFP4**：每 16 个元素共享 1 个 FP8(E4M3) 缩放因子（NVIDIA 自定义）；
   - 详见 [OCP Microscaling Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

#### `tcgen05.mma` PTX 指令格式

```cuda
// Blackwell tcgen05.mma（概念性示意，实际约束以 PTX ISA 和 CUTLASS 文档为准）
asm volatile(
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
    : 
    : "l"(acc_desc),   // 输出/累加位置，指向 Tensor Memory
      "l"(a_desc),     // A matrix descriptor 或 TMEM 地址
      "l"(b_desc),     // B matrix descriptor
      "r"(idesc),      // instruction descriptor：编码 MMA kind / tile / scale 路径
      "r"(scale_factor)
);
```

> 说明：上面仍是概念性示意。`tcgen05.mma` 的实际参数序列包含 instruction descriptor (idesc) 字段，编码 tile shape、是否带 block scale、转置等；CUDA 12.8 / 13.x 的 PTX ISA 仍在演进，**写代码请以最新 PTX ISA `tcgen05` 章节与 CUTLASS Blackwell 模板为准**，不要把本示意当成最终签名。

### 13.3 Distributed Shared Memory（分布式共享内存）

#### 背景

Hopper 引入了 **Thread Block Cluster** 机制：将多个 CTA 组成一个 Cluster，调度到同一个 **GPC（GPU Processing Cluster）** 内相邻的 SM 上。

**Cluster 关键限制**（详见 [[CUDA CTA 与 Thread Block Cluster 入门]]）：

- **Cluster size**：portable 上限 = 8 CTA；通过 `cudaFuncSetAttribute(cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` 可 opt-in 到 16 CTA。Hopper 与 Blackwell 都遵守这一上限——**Blackwell 没有把 cluster 再扩大**。
- **物理位置**：cluster 内所有 CTA 必须落在**同一个 GPC** 的不同 SM 上。
- **DSMEM 路径**：cluster 内 CTA 间的 SMEM 访问走**片内 SM-to-SM fabric**（cluster local network），**不走 NVLink**——NVLink 是 GPU 之间的互联，与 cluster 内通信无关。

Cluster 内的 SM 可以**互相访问对方的 Shared Memory**，形成 **Distributed Shared Memory（DSMEM）**：

```
Cluster（4 个 CTA，调度到 4 个相邻 SM）：
┌──────────────┐  ┌──────────────┐
│  SM 0        │  │  SM 1        │
│  SMEM(228KB) │←→│  SMEM(228KB) │  低延迟互联（SM-to-SM fabric，非 NVLink）
│  CTA 0       │  │  CTA 1       │
└──────────────┘  └──────────────┘
        ↕                ↕
┌──────────────┐  ┌──────────────┐
│  SM 2        │  │  SM 3        │
│  SMEM(228KB) │←→│  SMEM(228KB) │
│  CTA 2       │  │  CTA 3       │
└──────────────┘  └──────────────┘

每个 CTA 自己的 SMEM：仍是 228 KB（Hopper / Blackwell 上限）
Cluster 可见 SMEM 总量：N × 228 KB（N = cluster size，本例 4 × 228 = 912 KB）
```

> **常见误读**："cluster 可见 SMEM = 912 KB" 描述的是 4-CTA cluster 的**集合容量**——CTA 0 可以通过 DSMEM 访问其他 3 个 CTA 的 SMEM，相当于片上有 912 KB 可达数据；**但单个 CTA 自己的 SMEM 仍是 228 KB**，不会变大。寄存器分配、occupancy 等仍按 228 KB 计。

#### 对 GEMM 的意义

在矩阵乘法中，cluster + DSMEM 相当于扩大数据复用半径，提高 HBM 带宽利用率：

```
传统单 CTA：BM=128, BN=256, BK=64 → 数据复用仅限于 CTA 内的 228 KB SMEM
Cluster 扩展：cluster 内 2 个 CTA 共享部分 tile，跨 CTA 复用 A 或 B
              → 等效 tile 更大，HBM 访问次数下降
```

#### DSMEM 访问方式

```cuda
// 在 Hopper Kernel 中访问其他 CTA 的 SMEM
// 使用 cute::cluster_wait 和 cute::cluster_sync
#include <cute/arch/cluster_sm90.hpp>

// 获取 Cluster 内某个 CTA 的 SMEM 指针（Rank 为该 CTA 在 Cluster 内的索引）
void* peer_smem_ptr = cute::cluster_local_ptr(peer_cta_rank, local_smem_ptr);

// 发起对 peer SMEM 的原子操作或直接访问
// （实际通过 TMA + multicast 实现最高效的 DSMEM 访问）
```

#### TMA Multicast（多播）

配合 Cluster，TMA 支持**广播模式**：从 HBM 加载一份数据，同时写入 Cluster 内多个 CTA 的 SMEM：

```cuda
// TMA multicast：一次 HBM 读取，写入 cluster_size 个 SMEM
cute::copy(SM90_TMA_LOAD_MULTICAST{},  // 多播 TMA
           tma_descriptor,
           source_coord,
           smem_ptrs,              // 指向多个 CTA SMEM 的指针数组
           cluster_mask,           // 哪些 CTA 参与广播
           mbarrier);
```

有效带宽节省：每份数据只从 HBM 读一次，但被 N 个 CTA 使用 → **HBM 带宽需求降低 N 倍**。

### 13.4 Blackwell Block Cluster 改进

Blackwell 仍遵守 Hopper 的 cluster size 上限（portable 8 / opt-in 16），并未将其扩大。Blackwell 的改进集中在调度和搬运：

- **TMEM-aware 调度**：调度器更优先将同 cluster 的 CTA 落在 TMEM 资源充足的 SM 上；
- **`tcgen05.mma.cta_group::2`**：允许同 cluster 内两个 CTA 协作完成更大的 MMA，等效提高单次 MMA 的 N 维 tile；
- **NVLink 5.0**：单 GPU 端口双向聚合带宽提升到 1.8 TB/s——但**这是 GPU 之间**的互联，与 cluster 内 CTA 间通信（走片内 SM-to-SM fabric）**无关**。常见误读："Cluster 间通信带宽翻倍到 1.8 TB/s" 是错的。

---

## 14. 性能分析方法论

### 14.1 Roofline 分析

判断 Kernel 是**计算瓶颈**还是**内存带宽瓶颈**：

```python
# 伪代码
arithmetic_intensity = total_flops / total_bytes_accessed

if arithmetic_intensity > ridge_point:
    bottleneck = "计算瓶颈，需要提升 FLOPs/cycle"
    action = "增大 Tile 尺寸，使用 Tensor Core，减少非计算指令"
else:
    bottleneck = "带宽瓶颈，需要减少内存访问"
    action = "提高数据复用，向量化访问，使用 L2 cache"
```

### 14.2 关键性能指标

| 指标 | 说明 | 工具 |
|------|------|------|
| **SM Utilization** | SM 活跃时间占比 | Nsight Compute |
| **Warp Occupancy** | 活跃 Warp 数 / 最大 Warp 数 | Nsight Compute |
| **Memory Throughput** | 实际 HBM 带宽利用率 | Nsight Compute |
| **L1/L2 Hit Rate** | 缓存命中率 | Nsight Compute |
| **Bank Conflicts** | SMEM Bank 冲突次数 | Nsight Compute |
| **Achieved Occupancy** | 实际 Warp 占用率 | Nsight Compute |
| **Pipe Stalls** | 流水线暂停次数 | Nsight Compute |

### 14.3 Nsight Compute 关键 Section

```bash
# 使用 ncu 分析 CUDA Kernel 性能
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_active,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  smsp__warp_issue_stalled_math_throttle_per_warp_active.pct,\
  gpu__time_duration.sum \
  --target-processes all \
  ./your_gemm_binary

# 生成完整报告
ncu -o report --set full ./your_gemm_binary
```

### 14.4 常见性能陷阱

**寄存器溢出（Register Spilling）**：
```
问题：Kernel 使用寄存器超出限制 → 溢出到 L1 Cache 或 Local Memory（速度等同 HBM！）
检测：ncu 中查看 l1tex__data_pipe_lsu_wavefronts_mem_local.sum
解决：减少 TM/TN tile 大小，或使用 __launch_bounds__ 限制寄存器使用
```

**低 Warp Occupancy**：
```
问题：每个 SM 上活跃 Warp 数太少 → 无法隐藏内存延迟
原因：Shared Memory 用量过大 或 寄存器用量过大 导致 Block 数量受限
解决：权衡 SMEM 大小 vs Block 数量（Occupancy Calculator）
```

**非对齐访问**：
```
问题：float4 加载地址不是 16 字节对齐 → 降级为多次 32-bit 加载
检测：ncu 中 l1tex__data_bank_conflicts_pipe_lsu 升高
解决：使用 cudaMalloc 保证对齐（默认 256 字节对齐）
```

---

## 15. 完整优化路径总结

![CUDA GEMM 优化路径栈](/gpu/drawings/gemm-优化路径栈.svg)

### 15.1 优化层次图

```
Level 0: Naive GEMM
         └── 性能：~1% 峰值（纯 HBM 访问，极低复用）

Level 1: SMEM Tiling (BM×BN×BK)
         └── 性能提升：~8–16x（减少 HBM 流量）

Level 2: + Register Tiling (TM×TN)
         └── 性能提升：~2–4x（提升寄存器复用，减少 SMEM 压力）

Level 3: + Vectorized Access (float4/float8)
         └── 性能提升：~1.2–1.5x（提升内存带宽利用率）

Level 4: + Bank Conflict 消除 (Padding/Swizzle)
         └── 性能提升：~1.1–1.3x（消除 SMEM 串行化）

Level 5: + Double Buffering (SMEM / cp.async)
         └── 性能提升：~1.2–1.5x（计算与访存重叠）

Level 6: + Tensor Core (WMMA / MMA PTX)
         └── 性能提升：~4–16x（Tensor Core vs CUDA Core）

Level 7 [Hopper]: + TMA + WGMMA + mbarrier 流水线
         └── 性能提升：~2–3x（更高效的数据搬运 + 更大 MMA）

Level 8 [Blackwell]: + tcgen05.mma + TMEM + FP8/FP4 + DSMEM
         └── 性能提升：~2–4x（更大算力 + 更高效的内存架构）

最终性能：~90% 硬件峰值（cuBLAS / CUTLASS 水平）
```

### 15.2 各优化维度对比

| 优化手段 | 解决的问题 | 实现复杂度 | 效果 |
|---------|-----------|-----------|------|
| SMEM Tiling | HBM 带宽瓶颈 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Register Tiling | SMEM 带宽 + 指令效率 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Vectorized Access | 内存事务效率 | ⭐⭐ | ⭐⭐⭐ |
| Bank Conflict 消除 | SMEM 串行化 | ⭐⭐ | ⭐⭐ |
| Double Buffering | 计算-访存流水线 | ⭐⭐⭐ | ⭐⭐⭐ |
| Tensor Core (WMMA) | 计算吞吐 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| TMA (Hopper) | 数据搬运效率 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| WGMMA (Hopper) | 计算吞吐 + 流水线 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| DSMEM + Cluster | 大 Tile 数据复用 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `tcgen05.mma` + FP8/FP4 (Blackwell) | 极致算力利用 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 15.3 推荐学习路径

1. **入门**：理解 SMEM Tiling，手写 Level 1–2 版本
2. **进阶**：学习 CUTLASS 2.x，理解 Register Tiling 和 Tensor Core 编程
3. **高阶（Hopper）**：
   - 阅读 CUTLASS 3.x 源码（重点：`include/cutlass/gemm/collective/`）
   - 学习 CuTe 库（Tensor 抽象，Swizzle，TMA 封装）
   - 阅读 FlashAttention-2/3 源码（工业界 TMA+WGMMA 最佳实践）
4. **极致（Blackwell）**：
   - 研究 CUTLASS 3.6+ 的 Blackwell backend
   - 阅读 NVIDIA 官方 Blackwell Tuning Guide

### 15.4 关键参考资料

| 资源 | 说明 |
|------|------|
| [CUTLASS 3.x](https://github.com/NVIDIA/cutlass) | NVIDIA 官方高性能 GEMM 库，代码即文档 |
| [CuTe 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) | CUTLASS 3.x 的核心 Tensor 抽象层 |
| [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) | Hopper TMA+WGMMA 工业级实践 |
| [siboehm/CUDA-Learn-Notes](https://github.com/siboehm/SGEMM_CUDA) | 从 Naive 到优化的逐步教程 |
| [Reed et al., "Stream-K" (2023)](https://arxiv.org/abs/2301.03598) | Persistent GEMM 调度论文 |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | PTX 指令集参考 |
| [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) | GPU 性能分析工具 |
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 官方编程指南 |

## 16. 原 CUDA 目录文章整合

### 16.1 来源与定位

原 `CUDA/` 目录中有一篇中文整理和一篇英文版整理，主题都是 **CUDA GEMM Optimization: From Naive Implementation to Ultimate Optimization**。中文旧文来自微信公众号“算力基建洞察”，原始链接为：

https://mp.weixin.qq.com/s/EtQ7XQWc4HL36C1hg830ZA

旧文的价值在于用更短的路径解释“从朴素 GEMM 到接近 cuBLAS 的优化阶梯”。本主文档保留更完整的 Hopper/Blackwell、Tensor Core、TMA/WGMMA/`tcgen05.mma` 内容；旧文中的五阶段路径、性能倍率、关键概念和实践建议整合在本节，作为快速复盘入口。

相关笔记：

- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA Kernel 示例：矩阵乘法]]

### 16.2 五阶段优化路径

| 阶段 | 核心技术 | 相对性能 | 主要瓶颈 | 对应章节 |
|------|----------|----------|----------|----------|
| v1 Naive | 每线程计算一个 C 元素 | 1x | 非合并全局内存访问 | [第 3 节](CUDA%20GEMM%20矩阵乘法优化指南.md#3-naive-实现最基础的-cuda-gemm) |
| v2 Shared Memory Tiling | Block 协作加载 A/B tile 到 SMEM | 5-10x | Shared Memory Bank Conflict | [第 5 节](CUDA%20GEMM%20矩阵乘法优化指南.md#5-优化一shared-memory-tiling共享内存分块) |
| v3 Padding + float4 | Padding 消除 bank conflict，向量化加载 | 15-25x | 每线程计算量不足 | [第 6 节](CUDA%20GEMM%20矩阵乘法优化指南.md#6-优化二bank-conflict-消除)、[第 8 节](CUDA%20GEMM%20矩阵乘法优化指南.md#8-优化四向量化访存vectorized-memory-access) |
| v4 Thread-level Tiling | 每线程计算 TM x TN 输出子块 | 30-50x | 计算和访存仍有串行段 | [第 7 节](CUDA%20GEMM%20矩阵乘法优化指南.md#7-优化三warp-level-并行与-register-tiling) |
| v5 Double Buffering + cp.async | 双缓冲流水线，加载下一 tile 与当前 tile 计算重叠 | 80-100x | 接近硬件峰值，需要精细调度 | [第 9 节](CUDA%20GEMM%20矩阵乘法优化指南.md#9-优化五双缓冲流水线double-buffering) |

这条路径和第 15 节的层次图是一致的，只是旧文停在 CUDA Core + SMEM 优化的高阶实现；本主文档继续延展到 Tensor Core、WMMA/MMA PTX、Hopper TMA/WGMMA 与 Blackwell `tcgen05.mma`。

### 16.3 旧文中的关键概念

**内存合并访问（Memory Coalescing）**

同一 warp 内 32 个线程访问连续地址时，硬件可以合并成更少的内存事务。若访问形态类似 `A[tid * stride]` 且 `stride != 1`，通常会触发非合并访问。GEMM 的第一步优化常常不是改计算，而是先把线程到数据的映射调整成连续访问。

**Shared Memory Bank Conflict**

Shared Memory 可近似理解为 32 个 bank。常用判断式是：

```text
bank_id = address_offset_in_float % 32
```

同一条 SMEM 指令中，如果同一个 warp 的多个线程访问同一 bank 的不同地址，就会串行化。常见修复方式是 padding、转置存储或 swizzle。更详细的 bank conflict 例子放在 [[CUDA Shared Memory 与 Bank Conflict]]。

**算术强度**

算术强度是 FLOP/Byte。v1-v3 常见问题是数据搬运太重、复用不足，kernel 更容易 memory bound。Thread-level tiling 和 register tiling 的本质，是让每次从 SMEM/GMEM 搬来的数据服务更多 FMA。

**Occupancy 不是越高越好**

提高每线程寄存器使用量会降低 occupancy，但如果它显著提升寄存器复用和 CUDA Core 利用率，整体吞吐仍可能更高。GEMM 优化要看实际瓶颈和硬件计数器，不能只看 occupancy。

### 16.4 实操建议

1. 先用 Nsight Compute 判断瓶颈：global memory、shared memory、instruction issue、tensor pipe 还是 occupancy。
2. 每次只改一个优化维度，保留 baseline 和中间版本，避免性能回退后找不到原因。
3. 对 CUDA Core GEMM，优先按 `coalescing -> SMEM tiling -> bank conflict -> register tiling -> vectorized access -> double buffering` 的顺序推进。
4. 对深度学习 GEMM，最终应理解 Tensor Core 路线：WMMA、MMA PTX、CUTLASS/CuTe，以及 Hopper/Blackwell 上的 TMA、WGMMA、`tcgen05.mma`。
5. cuBLAS 是性能标杆，CUTLASS 是学习工业级 GEMM 设计最好的代码资料。

---

*文档持续更新 | 最后修订：2026年5月16日*
