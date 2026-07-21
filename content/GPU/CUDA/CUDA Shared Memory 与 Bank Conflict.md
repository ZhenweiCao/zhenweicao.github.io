---
aliases:
  - Bank Conflict
updated: 2026-05-30
tags:
  - gpu-computing
  - cuda-programming
---
# CUDA Shared Memory 与 Bank Conflict

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA Kernel 示例：归约求和]]

## 1. 共享内存的物理结构

GPU 的 Shared Memory（共享内存）在物理上被划分为若干个**内存 Bank**，每个 Bank 是一个独立的存储单元，可以在同一个时钟周期内独立响应访问请求。

- **CUDA 架构（Maxwell 及之后，含 Volta/Turing/Ampere/Hopper/Blackwell）**：始终 **32 个 Bank**，**Bank 宽度固定为 4 字节（32-bit）**。
- **Bank 编号规则**：地址 `addr` 对应的 Bank 编号为 `(addr / 4) % 32`。
- **历史注记**：Kepler 时代曾通过 `cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)` 切到 8 字节 bank 模式以缓解 `double` 访问冲突；该 API 在 Maxwell 之后被弃用，现代架构上**不再有 8 字节 bank 模式可配**。读到老资料里"bank 宽度可配为 8 字节"时直接忽略。

```
Shared Memory 物理布局（32-bit 模式）：

Bank ID:  0    1    2    3    4    5  ...  31   0    1    2  ...
         ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
地址：    │ 0  │ 4  │ 8  │ 12 │ 16 │ 20 │...│124 │128 │132 │136 │
         └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
         ←————————————— 第 0 行（128 字节）—————————————→←— 第 1 行 —
```

一个 Warp（32 个线程）同时访问共享内存时，如果 32 个线程访问的是 **32 个不同的 Bank**，则可以在 **1 次 shared memory request** 内全部完成——这是最理想的情况。

> 术语说明：NVIDIA 文档对 SMEM 用 "shared memory request"，对 global memory 才用 "transaction"。本文为了简洁，下文仍偶用"事务"指代 request，但两者不要与 global memory transaction 混淆。

---

## 2. 什么是 Bank Conflict

**Bank Conflict** 是指同一个 Warp 中有多个线程访问**同一个 Bank 中的不同 4 字节 word**，导致这些访问无法并行，必须**串行化**执行。

> 关键修正（2026-05-30）：判定的是"同一 Bank 的**不同 32-bit word**"。若多个线程命中**同一 Bank 的同一个 word**（即使读取该 word 内不同 byte 字段），硬件会触发 **broadcast / multicast** 而不是冲突。这一点在矩阵广播、查表型 kernel 中非常关键。

### 2.1 无 Bank Conflict（理想情况）

线程 `i` 访问 `smem[i]`，每个线程访问不同的 Bank：

```
线程：  T0   T1   T2   T3   T4  ...  T31
        │    │    │    │    │         │
        ▼    ▼    ▼    ▼    ▼         ▼
Bank:   0    1    2    3    4   ...   31

✓ 32 个线程同时完成，1 次 shared memory request
```

### 2.2 2-Way Bank Conflict

线程 `i` 访问 `smem[i * 2]`，步长为 2，每隔一个 Bank 访问：

```
线程：  T0   T1   T2   T3   T4   T5  ...  T15  T16  T17 ...
        │    │    │    │    │    │          │    │    │
        ▼    ▼    ▼    ▼    ▼    ▼          ▼    ▼    ▼
地址：   0    8   16   24   32   40   ...   120   0    8  ...
Bank:   0    2    4    6    8   10   ...   30    0    2  ...

! T0 和 T16 都访问 Bank 0（不同 word）→ 2-way conflict
! T1 和 T17 都访问 Bank 2 → 2-way conflict
  ……所有 Bank 都有 2 个线程竞争

✗ 需要 2 次 request 才能完成
```

### 2.3 32-Way Bank Conflict（最坏情况）

线程 `i` 访问 `smem[i * 32]`，步长为 32（所有线程访问同一个 Bank 的不同 word）：

```
线程：  T0    T1    T2    T3  ...  T31
        │     │     │     │         │
        ▼     ▼     ▼     ▼         ▼
地址：   0   128   256   384   ...  3968
Bank:   0    0     0     0    ...   0

! 所有线程都访问 Bank 0 的不同 word

✗ 需要 32 次 request 串行执行，性能下降 32 倍
```

### 2.4 Broadcast / Multicast —— 无 Conflict 的特例

Volta 之后硬件对"同一 Bank 同一 32-bit word"的多线程访问有两类合并行为：

| 情形 | 行为 | 是否冲突 |
|------|------|---------|
| 多个线程访问**完全相同的地址** | broadcast：1 次 request | 否 |
| 多个线程访问**同一个 4B word 的不同 byte 字段** | multicast：1 次 request | 否 |
| 多个线程访问**同一 Bank 的不同 word** | serialize：N 次 request | **是** |

```
线程：  T0    T1    T2  ...  T31
        │     │     │         │
        └─────┴─────┴────···──┘
                    │
                    ▼
地址：            smem[5]（Bank 5 的同一 word）

✓ 触发 broadcast/multicast，1 次 request 完成，无 Conflict
```

实战意义：查表型 kernel 中"全 warp 读同一权重"无需担心 bank conflict；只要不退化为读同一 Bank 不同 word 即可。

---

## 3. 矩阵转置中的经典 Bank Conflict

以 32×32 的共享内存 tile 转置为例，这是最常见的 Bank Conflict 场景。

### 3.1 有 Conflict 的写入

```cuda
__shared__ float tile[32][32];

// 线程 (tx, ty) 写入列优先
tile[tx][ty] = input[...];  // 写时无 conflict（行连续）
output[...] = tile[ty][tx]; // 读时：同一行的线程读同一列 → conflict！
```

读取阶段，Warp 中线程 `T0~T31` 均有 `ty` 相同（同一行），读取 `tile[ty][0]` ~ `tile[ty][31]`：

```
读取 tile[0][0..31]，即访问：

地址偏移： 0    4    8   12  ...  124
Bank：     0    1    2    3  ...   31   ✓ 无 Conflict

读取 tile[0..31][0]（转置后的列访问），即访问：

地址偏移：  0   128  256  384  ...  3968
Bank：      0    0    0    0   ...   0   ✗ 32-way Conflict！
```

### 3.2 Padding 解决方案

在 tile 的列维度加 1 个 padding，使步长错开：

```cuda
__shared__ float tile[32][33];  // 每行多 1 个元素（padding）
```

加 padding 后的 Bank 分布：

```
不加 padding（tile[32][32]）：
行 0:  Bank  0  1  2  3 ...31  (偏移 0~124)
行 1:  Bank  0  1  2  3 ...31  (偏移 128~252)  ← 列 0 永远在 Bank 0

加 padding（tile[32][33]）：
行 0:  Bank  0  1  2  3 ...31 32%32=0  (偏移 0~128，共 33×4=132 字节)
行 1:  Bank  1  2  3  4 ... 0  1  ...  (偏移 132 开始，132/4=33, 33%32=1)
行 2:  Bank  2  3  4  5 ...           (偏移 264 开始，264/4=66, 66%32=2)
...
行 k:  列 0 在 Bank k%32

✓ 不同行的列 0 位于不同 Bank，32-way Conflict 彻底消除
```

---

## 4. 完整示例对比

```cuda
// ===================== 有 Bank Conflict =====================
__global__ void transpose_conflict(float* out, float* in, int N) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // 32-way conflict
}

// ==================== 无 Bank Conflict（Padding）====================
__global__ void transpose_no_conflict(float* out, float* in, int N) {
    __shared__ float tile[32][33]; // +1 padding

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // ✓ 无 conflict
}
```

---

## 5. 检测与分析

使用 **Nsight Compute** 检测 Bank Conflict：

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
./your_kernel
```

| 指标 | 含义 |
|------|------|
| `...op_ld.sum` | 读操作产生的 Bank Conflict 次数 |
| `...op_st.sum` | 写操作产生的 Bank Conflict 次数 |

---

## 6. 避免 Bank Conflict 的常用策略

| 策略 | 适用场景 |
|------|----------|
| **Padding**（列+1） | 矩阵转置、按列访问 SMEM |
| **调整访问步长** | 使步长与 32 互质（如奇数步长） |
| **数据布局重排** | 加载时重排为对 Bank 友好的顺序 |
| **使用向量类型** | `float4` 读写合并访问，减少事务数 |
| **Swizzle 寻址** | CuTe/Triton 中的高级技术，通过异或重映射地址 |

---

## 7. Case 分析：Tiled GEMM 中的 Bank Conflict

### 7.1 代码

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 7.2 加载阶段：无 Bank Conflict

加载 `As` 和 `Bs` 时，线程按 `(ty, tx)` 写入 `smem[ty][tx]`，同一 warp 内 `ty` 相同、`tx` 从 0 到 31 连续递增，访问连续地址，落在 32 个不同 Bank。

```
加载 As[ty][0..31]（ty 固定，tx = 0..31）：

threadIdx.x:   0    1    2    3  ...  31
写入地址偏移:   0    4    8   12  ...  124
Bank:          0    1    2    3  ...   31

✓ 每个线程访问不同 Bank，1 个事务完成
```

### 7.3 计算阶段：Bs 存在 32-way Bank Conflict

计算内层循环 `sum += As[ty][k] * Bs[k][tx]` 时：

- **读 `As[ty][k]`**：`ty` 固定，`k` 是公共循环变量，整个 warp 读的是**同一个地址** `As[ty][k]`，触发广播，无 Conflict。
- **读 `Bs[k][tx]`**：`k` 固定，`tx` 从 0 到 31 连续递增，访问 `Bs[k][0..31]`，连续地址，无 Conflict。

**等等——真的没有问题吗？** 我们来仔细看 `As` 的读取：

```
内层 k 循环，某一步 k=5，整个 block 所有 warp 同时运行：

Warp 0（ty=0）: As[0][5]  → 偏移 = (0*32 + 5)*4 = 20,  Bank = 5
Warp 1（ty=1）: As[1][5]  → 偏移 = (1*32 + 5)*4 = 148, Bank = 148/4 % 32 = 37 % 32 = 5
Warp 2（ty=2）: As[2][5]  → 偏移 = (2*32 + 5)*4 = 276, Bank = 276/4 % 32 = 69 % 32 = 5
...
Warp 31（ty=31）:As[31][5]→ 偏移 = (31*32+5)*4 = 4004, Bank = 1001 % 32 = 5
```

同一个 warp 内所有线程读的是**完全相同的地址**（广播，无 Conflict）。  
但是：**不同 warp 之间不共享执行单元，Bank Conflict 只在 warp 内计算，因此 warp 间互不影响。**

结论：**这段代码读 `As` 和 `Bs` 均无 Bank Conflict。**

### 7.4 真正的隐患：TILE_SIZE 不是 32 时

当 `TILE_SIZE = 16` 时，`As[16][16]`，每行 16 个 float = 64 字节。

```
As[16][16] 的 Bank 分布：

行 0:  Bank  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
行 1:  Bank 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
行 2:  Bank  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  ← 与行 0 完全重叠！
行 3:  Bank 16 17 ...
...
偶数行都在 Bank 0~15，奇数行都在 Bank 16~31

读 As[ty][k]（广播，同 warp 内无 Conflict）✓
```

但若改为**按列访问** `As[k][tx]`（某些变体的 GEMM kernel 会这样写）：

```
读 As[0..15][0]，即 warp 内 ty=0..15，tx 固定：

ty=0:  As[0][0]  偏移 = 0,   Bank = 0
ty=1:  As[1][0]  偏移 = 64,  Bank = 16
ty=2:  As[2][0]  偏移 = 128, Bank = 0   ← 与 ty=0 同 Bank！
ty=3:  As[3][0]  偏移 = 192, Bank = 16  ← 与 ty=1 同 Bank！
...

✗ 8-way Bank Conflict（TILE_SIZE=16 时，8 个线程争同一 Bank）
```

### 7.5 TILE_SIZE=32 时计算阶段完整示意图

以 `k=3` 时读取 `Bs[3][tx]` 为例：

```
Bs[TILE_SIZE][TILE_SIZE] = Bs[32][32]，每行 128 字节，恰好覆盖全部 32 个 Bank

读 Bs[3][tx]，tx = 0..31：

tx:          0     1     2     3   ...   31
偏移(bytes): 384  388   392   396  ...   508
Bank:         0    1     2     3   ...   31

✓ 32 个线程各占一个 Bank，1 个事务完成，无 Conflict
```

```
Bs 在 SMEM 中的布局（每格代表一个 float，数字为 Bank ID）：

         col→  0   1   2   3   4   5  ...  31
row↓  0      [ 0][ 1][ 2][ 3][ 4][ 5]...[31]
      1      [ 0][ 1][ 2][ 3][ 4][ 5]...[31]
      2      [ 0][ 1][ 2][ 3][ 4][ 5]...[31]
      ...
      31     [ 0][ 1][ 2][ 3][ 4][ 5]...[31]

读某一行（k 固定，tx=0..31）→ 每列对应不同 Bank ✓
读某一列（tx 固定，k=0..31）→ 所有行的同一列在同一 Bank ✗（但此处 tx 是 warp 内广播，无 Conflict）
```

### 7.6 总结

| 访问 | 模式 | Bank 情况 | 结论 |
|------|------|-----------|------|
| 加载 `As[ty][tx]` | ty 固定，tx=0..31 连续 | 32 个不同 Bank | ✓ 无 Conflict |
| 加载 `Bs[ty][tx]` | ty 固定，tx=0..31 连续 | 32 个不同 Bank | ✓ 无 Conflict |
| 计算读 `As[ty][k]` | ty 固定，k 为公共值，全 warp 同地址 | 广播 | ✓ 无 Conflict |
| 计算读 `Bs[k][tx]` | k 固定，tx=0..31 连续 | 32 个不同 Bank | ✓ 无 Conflict |
| **若改写为 `As[k][ty]`** | **ty=0..31，k 固定，按列访问** | **32 行同列 → 同 Bank** | **✗ 32-way Conflict** |

**该 kernel 在 `TILE_SIZE=32` 时实际上没有 Bank Conflict**，其设计（行优先加载 + 广播读）正好规避了所有冲突。  
风险点在于：若将内层循环改为 `As[k][ty]` 的列访问形式，或使用非 32 倍数的 `TILE_SIZE`，就会引入 Conflict。

---

## 8. Hopper / Blackwell：TMA + swizzle 如何天然避免 Bank Conflict

Hopper 引入 **TMA（Tensor Memory Accelerator）** 后，"如何把 global memory 中的矩阵 tile 搬到 shared memory 又不踩 bank conflict"这个老问题被硬件层重新设计了。要点：

- **swizzled SMEM layout**：TMA descriptor 支持 `SWIZZLE_32B / 64B / 128B` 几种内置 swizzle 模式（CUTLASS 中对应 `Layout_K_SW32`、`Layout_K_SW64`、`Layout_K_SW128`）。被 swizzle 后，原本"按列访问会落在同一 Bank"的 32×32 tile 自动重排，列方向访问也分散到 32 个不同 Bank。
- **`cp.async.bulk.tensor` 配合 swizzle**：写入 SMEM 的 byte stride 与 swizzle 模式必须匹配；CUTLASS / cuDNN / FlashAttention-3 已经把这套约束封装进 layout 类，新代码很少需要手工对齐。
- **`ldmatrix` / `stmatrix`**：与 swizzled SMEM 配合时，硬件按 fragment 重排顺序读写，避免 warp 内的同 Bank 冲突。
- **结果**：Hopper 之后的高性能 GEMM/attention kernel 通常**不再需要 `+1 padding` 这种 trick**——padding 反而会破坏 swizzle 的对齐前提，导致 TMA 路径退化或失效。

详细如何把 swizzle 与 tile shape / `wgmma` 组合，见 [[CUDA GEMM 矩阵乘法优化指南]] §"Hopper TMA + WGMMA"。如果你写的是 Ampere 及之前的 kernel，传统的 `+1 padding` 仍然有效；从 Hopper 开始优先考虑 TMA + swizzle。

