---
title: "第3章：GPU 硬件原理"
content_type: guide
maturity: reviewed
updated: 2026-07-27
publish: true
tags:
  - gpu-computing
  - gpu-programming
  - concept-note
---
# 第3章：GPU 硬件原理

> 理解 GPU 内部结构，才能写出高效的代码

## 本章定位

本章负责解释 CUDA 代码为什么会快或慢：SM、warp、SIMT、memory hierarchy、occupancy 是后续所有优化的基础。严格定义和架构代际演进以 [[GPU 硬件架构背景与编程范式]] 为准。

配套主文档：

- [[GPU 硬件架构背景与编程范式]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA CTA 与 Thread Block Cluster 入门]]

## 学习目标

- 理解 SM（Streaming Multiprocessor）结构
- 掌握 Warp 执行模型
- 理解内存层次和访问模式
- 知道为什么某些代码更快

## 3.1 GPU 硬件架构概览

### 3.1.1 从程序员视角看 GPU

```text
你写的代码:
┌─────────────────────────────────────┐
│  Kernel<<<Grid of Blocks>>>         │
│                                      │
│  Block 0   Block 1   Block 2   ...   │
│  ┌─────┐   ┌─────┐   ┌─────┐        │
│  │T0..N│   │T0..N│   │T0..N│        │
│  └─────┘   └─────┘   └─────┘        │
└─────────────────────────────────────┘
              ↓
           如何映射到
              ↓
实际硬件:
┌─────────────────────────────────────┐
│              GPU 芯片               │
├─────────┬─────────┬─────────┬───────┤
│  SM 0   │  SM 1   │  SM 2   │  ...  │
├─────────┼─────────┼─────────┼───────┤
│  SM 4   │  SM 5   │  SM 6   │  ...  │
└─────────┴─────────┴─────────┴───────┘
```

### 3.1.2 SM 是什么？

**SM (Streaming Multiprocessor)** = GPU 的计算核心

一个 GPU 包含多个 SM（A100 有 108 个，RTX 4090 有 128 个）

每个 SM 就像一个"小型 CPU"，可以同时执行多个 Block。

### 3.1.3 SM 内部结构

```text
单个 SM 的结构（简化版）:
┌────────────────────────────────────────────────┐
│                   SM                            │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │        Warp 调度器 (Warp Scheduler)       │ │
│  │   负责选择哪个 Warp 在哪个时钟周期执行    │ │
│  └──────────────────────────────────────────┘ │
│                      ↓                         │
│  ┌──────────────────────────────────────────┐ │
│  │           CUDA 核心 (CUDA Cores)         │ │
│  │   ┌────┐ ┌────┐ ┌────┐ ┌────┐          │ │
│  │   │ FP │ │ FP │ │ INT │ │INT │   ...    │ │
│  │   └────┘ └────┘ └────┘ └────┘          │ │
│  │   (浮点运算核心) (整数运算核心)          │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │          Tensor Cores (可选)              │ │
│  │   专门做矩阵乘法的硬件加速器              │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │         共享内存 (Shared Memory)          │ │
│  │         ~100KB，非常快                    │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │            寄存器文件 (Registers)         │ │
│  │         ~256KB，最快                      │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │             L1 Cache                      │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

## 3.2 Warp 执行模型

### 3.2.1 什么是 Warp？

**Warp = 32 个线程组成的执行单位**

GPU 不是逐个线程执行的，而是以 Warp 为单位执行的。

```text
一个 Block:
┌────────────────────────────────────────┐
│ 256 个线程                              │
│                                        │
│ Warp 0: Thread 0-31   (同时执行)       │
│ Warp 1: Thread 32-63  (同时执行)       │
│ Warp 2: Thread 64-95  (同时执行)       │
│ Warp 3: Thread 96-127 (同时执行)       │
│ ...                                    │
│ Warp 7: Thread 224-255 (同时执行)      │
└────────────────────────────────────────┘
```

### 3.2.2 SIMT 执行模型

**SIMT = Single Instruction, Multiple Threads**

所有 Warp 内的 32 个线程：
- **同时执行相同的指令**
- 但处理不同的数据

```text
Warp 执行过程:
时钟周期 1: 所有 32 个线程执行 "加载 A[i]"
时钟周期 2: 所有 32 个线程执行 "加载 B[i]"
时钟周期 3: 所有 32 个线程执行 "C[i] = A[i] + B[i]"
...
```

### 3.2.3 分支分歧（重要！）

问题：如果 Warp 内的线程执行不同的代码？

```cpp
__global__ void bad_branch(int* data, int n) {
    int idx = threadIdx.x;
    
    if (idx < 16) {
        data[idx] = data[idx] * 2;  // 前16个线程执行
    } else {
        data[idx] = data[idx] + 1;  // 后16个线程执行
    }
}
```

**发生了什么？**

```text
Warp 0 (Thread 0-31):
                    
时间片 1: 执行 if 分支
         Thread 0-15 执行 data[idx] = data[idx] * 2
         Thread 16-31 空闲等待 (masked out)
         
时间片 2: 执行 else 分支
         Thread 0-15 空闲等待 (masked out)
         Thread 16-31 执行 data[idx] = data[idx] + 1
         
结果：执行时间翻倍！
```

### 3.2.4 如何避免分支分歧

**方法 1：重排数据**

```cpp
// 不好：相邻线程执行不同分支
for (int i = 0; i < 32; i++) {
    if (i < 16) A[i] = B[i] * 2;
    else A[i] = B[i] + 1;
}

// 好：相邻线程执行相同分支
for (int i = 0; i < 16; i++) {
    A[i] = B[i] * 2;      // Warp 0 全部执行这个
}
for (int i = 16; i < 32; i++) {
    A[i] = B[i] + 1;      // Warp 1 全部执行这个
}
```

**方法 2：使用条件表达式代替分支**

```cpp
// 不好
if (condition) {
    result = a + b;
} else {
    result = a - b;
}

// 好（编译器可能自动优化）
float add_result = a + b;
float sub_result = a - b;
result = condition ? add_result : sub_result;
```

## 3.3 内存层次详解

### 3.3.1 完整的内存层次

```text
速度从快到慢:
┌─────────────────────────────────────────────┐
│ 1. 寄存器 (Registers)                       │ ← 最快
│    - 每个 Thread 私有                       │
│    - 延迟: 1 周期                           │
│    - 容量: 每线程 ~255 个                   │
├─────────────────────────────────────────────┤
│ 2. 共享内存 (Shared Memory)                 │
│    - 每个 Block 内共享                      │
│    - 延迟: ~20 周期                         │
│    - 容量: 每 SM ~100KB                     │
├─────────────────────────────────────────────┤
│ 3. L1 Cache                                 │
│    - 硬件自动管理                           │
│    - 延迟: ~30 周期                         │
├─────────────────────────────────────────────┤
│ 4. L2 Cache                                 │
│    - 所有 SM 共享                           │
│    - 延迟: ~200 周期                        │
│    - 容量: ~40MB (A100)                     │
├─────────────────────────────────────────────┤
│ 5. 全局内存 (Global Memory/HBM)             │ ← 最慢
│    - 所有 Thread 可访问                     │
│    - 延迟: ~400 周期                        │
│    - 容量: ~40-80GB                         │
│    - 带宽: 1-3 TB/s                         │
└─────────────────────────────────────────────┘
```

### 3.3.2 内存访问延迟对比

用人来比喻：

| 内存类型 | 延迟 | 人类类比 |
|----------|------|----------|
| 寄存器 | 1 周期 | 从口袋拿东西（瞬间） |
| 共享内存 | 20 周期 | 从桌上拿东西（几秒） |
| L1/L2 Cache | 30-200 周期 | 从房间另一头拿（几分钟） |
| 全局内存 | 400 周期 | 从隔壁房间拿（几小时） |

### 3.3.3 全局内存访问模式

**合并访问 (Coalesced Access)**

最好的情况：相邻线程访问相邻内存

```cpp
__global__ void good_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 相邻线程访问相邻地址
    // Thread 0 -> data[0]
    // Thread 1 -> data[1]
    // Thread 2 -> data[2]
    // ...
    float val = data[idx];  // ✓ 合并访问
}
```

**为什么合并访问快？**

```text
GPU 内存控制器一次读取连续的 128 字节
（32 个线程 × 4 字节/float）

合并访问:
┌────────────────────────────────────────┐
│ Thread 0-31 同时请求 data[0-31]        │
│         ↓                              │
│ 内存控制器一次读取 128 字节            │
│         ↓                              │
│ 所有线程同时获得数据                   │
└────────────────────────────────────────┘

跨步访问 (Stride Access):
┌────────────────────────────────────────┐
│ Thread 0 请求 data[0]                  │
│ Thread 1 请求 data[32]    ← 跨了 32 个元素  │
│ Thread 2 请求 data[64]                 │
│ ...                                    │
│ 内存控制器需要多次读取                 │
│ 效率大幅下降                           │
└────────────────────────────────────────┘
```

### 3.3.4 共享内存和 Bank Conflict

**共享内存的结构**

共享内存被分成 32 个 Bank（为了匹配 Warp 的 32 个线程），**每个 Bank 宽度固定为 4 字节**。"bank 宽度可配为 8 字节" 是 Kepler 时代的旧 API，Maxwell 起已弃用。

```text
Bank 0   Bank 1   Bank 2   ...  Bank 31
  ↓        ↓        ↓             ↓
[0]      [1]      [2]    ...   [31]
[32]     [33]     [34]    ...   [63]
[64]     [65]     [66]    ...   [95]
...
```

**什么是 Bank Conflict？**

多个线程同时访问同一个 Bank 的**不同 word** → 串行化。注意：命中**同一 Bank 的同一 word** 会触发 broadcast/multicast，不冲突。

```cpp
__shared__ float data[128];

// ✗ 严重 Bank Conflict
// 所有 32 个线程都访问 Bank 0 的不同 word
float val = data[threadIdx.x * 32];  // Thread 0 访问 data[0]
                                      // Thread 1 访问 data[32]
                                      // 都在 Bank 0！

// ✓ 无 Bank Conflict
float val = data[threadIdx.x];  // 每个线程访问不同 Bank

// ✓ Broadcast / Multicast：无冲突
float val = data[5];  // 整个 warp 命中同一个 word，硬件广播
```

**解决方法：Padding（Ampere 及之前）**

```cpp
// ✗ 可能 Bank Conflict
__shared__ float matrix[32][32];

// ✓ 使用 Padding 避免 Bank Conflict
__shared__ float matrix[32][33];  // 多加一列
// 现在 matrix[i][j] 和 matrix[i+1][j] 在不同 Bank
```

> Hopper / Blackwell 上推荐用 TMA + SMEM swizzle 取代 padding。详细：[[CUDA Shared Memory 与 Bank Conflict]]。

## 3.4 占用率 (Occupancy)

### 3.4.1 什么是占用率？

**占用率 = 活跃 Warp 数 / 最大 Warp 数**

高占用率意味着 GPU 资源得到充分利用。

### 3.4.2 影响占用率的因素

```text
每个 SM 的限制:
1. 最大 Warp 数: 64 (A100)
2. 最大线程数: 2048
3. 最大 Block 数: 32
4. 最大寄存器数: 65536
5. 最大共享内存: ~160KB
```

### 3.4.3 计算示例

假设每个 Block:
- 256 个线程 = 8 个 Warp
- 使用 32 个寄存器/线程 = 8192 个寄存器
- 使用 4KB 共享内存

在 A100 (最大 2048 线程, 65536 寄存器, 160KB 共享内存):
- 按 Warp 限制: 64 / 8 = 8 个 Block
- 按线程限制: 2048 / 256 = 8 个 Block
- 按寄存器限制: 65536 / 8192 = 8 个 Block
- 按共享内存限制: 160KB / 4KB = 40 个 Block

**最小值是 8，所以最多同时运行 8 个 Block**

占用率 = 8 × 8 / 64 = 100%

### 3.4.4 使用 CUDA Occupancy Calculator

NVIDIA 提供工具帮助计算：
```cpp
int num_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, threads, shared_mem);
float occupancy = (num_blocks * threads / 32) / (float)max_warps;
```

## 3.5 Tensor Core

### 3.5.1 什么是 Tensor Core？

Tensor Core 是专门做矩阵乘法的硬件单元。**不同代际的硬件原生 MMA 形状不同**：

```text
普通 CUDA Core:
  1 次运算 = 1 个乘加

Tensor Core（硬件原生 MMA，按代际）:
  Volta V100:        m8n8k4 （WMMA API 暴露为 16×16×16 是抽象 tile，不是硬件原生）
  Ampere A100:       m16n8k16 (FP16/BF16), m16n8k8 (TF32), m16n8k32 (INT8)
  Hopper H100/H200:  wgmma m64n{8..256}k16 (FP16), k32 (FP8/INT8), k8 (TF32)
  Blackwell B200/B300: tcgen05.mma，围绕 Tensor Memory + descriptor + block scaling

公式：D = A × B + C
```

> 严格的硬件形状与代际算力（dense / sparse）以 [[GPU 硬件架构背景与编程范式]] §"指令与能力演进" 和 [[NVIDIA GPU 架构与规格]] §"Tensor Core 代际" 为权威。本章只给入门直觉，不重复维护精确数字。

### 3.5.2 Tensor Core 的优势（代表性参考值）

下表为不同硬件 dense FP16 算力参考，**精确数字（含 dense / sparse 双口径）以规格表为唯一来源**：

| GPU | FP16 普通 Core (TFLOPS) | FP16 Tensor Core dense (TFLOPS) | 加速比 |
|-----|------------------------|---------------------------------|--------|
| A100 SXM | 39 | 156（sparse 312） | ~4× |
| H100 SXM5 | 67 | 989（sparse 1,979） | ~15× |
| B200 (1 GPU) | ~75 | ~2,500（sparse ~5,000，含 NVFP4 路径再翻倍） | ~33× |

> Ampere A100 不支持 FP8 Tensor Core；FP8 从 Hopper / Ada 起进入主线。Blackwell 进一步引入 FP4 / FP6 与块缩放（block scaling）。

### 3.5.3 使用 Tensor Core

最简单的方法：使用 CUTLASS 库或 cuBLAS 的 Tensor Core 版本。

高级方法：使用 WMMA API（详见高级章节）

## 3.6 性能分析工具

### 3.6.1 Nsight Compute

详细分析单个 Kernel 的性能：

```bash
ncu --set full ./my_program
```

输出包括：
- 内存吞吐量
- 计算吞吐量
- 占用率
- Warp 执行效率
- 等等

### 3.6.2 Nsight Systems

分析整个程序的时间线：

```bash
nsys profile ./my_program
```

可以看到：
- 各个 Kernel 的执行时间
- CPU-GPU 之间的数据传输
- 并行情况

### 3.6.3 nvprof (旧工具)

```bash
nvprof ./my_program
```

## 💡 本章要点

1. **SM 是 GPU 的计算单元**，一个 GPU 有多个 SM
2. **Warp 是执行单位**，32 个线程同时执行相同指令
3. **分支分歧会降低性能**，Warp 内线程走不同分支会串行
4. **内存有层次**：寄存器 > 共享内存 > L1/L2 > 全局内存
5. **合并访问最重要**：相邻线程访问相邻内存
6. **Bank Conflict 会降低共享内存性能**
7. **占用率反映 GPU 利用率**

## 📝 课后练习

1. 编写一个有分支分歧的 Kernel，观察性能
2. 编写一个有 Bank Conflict 的 Kernel，然后修复它
3. 使用 Nsight Compute 分析你的 Kernel
4. 计算你的 Kernel 的理论占用率

---

[上一章：CUDA 入门 ←](第2章-CUDA入门.md) | [下一章：优化技巧 →](第4章-优化技巧.md)
