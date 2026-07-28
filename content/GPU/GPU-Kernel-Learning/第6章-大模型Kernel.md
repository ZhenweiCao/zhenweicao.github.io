---
title: "第6章：大模型推理 Kernel"
content_type: guide
maturity: reviewed
created: 2026-05-17
updated: 2026-07-27
publish: true
featured: true
tags:
  - gpu-computing
---
# 第6章：大模型推理 Kernel

> 实现 GPT、LLaMA 等大模型的核心算子，掌握推理优化技巧

## 本章定位

本章负责把基础 CUDA kernel 能力映射到大模型推理：GEMM、softmax、norm、attention、RoPE、KV cache。这里先建立数据流和优化方向，具体算法语义可回到 LLM 目录，底层 GEMM/硬件细节回到 GPU 主文档。

配套主文档：

- [[GPU 学习任务]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件架构背景与编程范式]]
- [[Nsight Compute NCU 分析方法与优化思路]]

## 学习目标

- 实现大模型推理核心 Kernel（Softmax、LayerNorm、Attention 等）
- 理解各 Kernel 的优化要点和数值稳定性
- 掌握 Flash Attention 等高效算法
- 了解 KV Cache、RoPE 等推理关键技术

---

## 6.1 矩阵乘法（GEMM）

矩阵乘法是大模型推理的核心操作，占据 90% 以上的计算量。

### 6.1.1 基础实现

```cpp
// 基础 GEMM: C = A × B + C
// A: M × K, B: K × N, C: M × N
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] += sum;
    }
}
```

### 6.1.2 共享内存优化版本

```cpp
#define BLOCK_SIZE 16

__global__ void gemm_shared(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 分块计算
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // 加载到共享内存
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
                                        A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
                                        B[b_row * N + col] : 0.0f;
        __syncthreads();

        // 块内矩阵乘法
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 6.1.3 Tensor Core 版本

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_tensor_core(half* A, half* B, float* C,
                                  int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        store_matrix_sync(C + cRow * N + cCol, c_frag, N, mem_row_major);
    }
}
```

---

## 6.2 Softmax Kernel

### 6.2.1 基础实现（数值不稳定）

```cpp
__global__ void softmax_naive(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 计算 max（数值稳定性问题）
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max_val);
    }

    // 输出
    output[idx] = expf(input[idx] - max_val) / sum;
}
```

### 6.2.2 Warp 级优化版本

> **关键修订点（2026-05-30）**：`__shfl_down_sync` 做的是 **reduce-to-lane-0**，归约完成后只有 lane 0 持有最终值，其他 lane 还是部分和。如果不广播就直接 `expf(x - max_val) / sum`，**lane 1..31 用错误的 max/sum 写回，结果错**。必须用 `__shfl_sync(0xffffffff, val, 0)` 把 lane 0 的最终值广播给整个 warp，再写回。

```cpp
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;  // 只有 lane 0 是最终结果
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // 只有 lane 0 是最终结果
}

__device__ float warp_broadcast(float val) {
    // 把 lane 0 的值广播给整个 warp
    return __shfl_sync(0xffffffff, val, 0);
}

__global__ void softmax_optimized(float* input, float* output, int seq_len) {
    // 每个 warp 处理一行
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (row >= seq_len) return;

    float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    // 每个线程处理部分元素，求局部 max
    float max_val = -INFINITY;
    for (int i = lane; i < seq_len; i += 32) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // Warp 归约求 max，再广播给所有 lane
    max_val = warp_reduce_max(max_val);
    max_val = warp_broadcast(max_val);

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = lane; i < seq_len; i += 32) {
        sum += expf(row_input[i] - max_val);
    }

    // Warp 归约求 sum，再广播给所有 lane
    sum = warp_reduce_sum(sum);
    sum = warp_broadcast(sum);

    // 输出 softmax 结果（此时所有 lane 都拿到正确的 max_val / sum）
    for (int i = lane; i < seq_len; i += 32) {
        row_output[i] = expf(row_input[i] - max_val) / sum;
    }
}
```

### 6.2.3 Online Softmax（单次遍历）

> ℹ️ 下面是经典 online softmax 状态更新公式：`m_new = max(m, x)`，`d_new = d * exp(m - m_new) + exp(x - m_new)`。先用 `m_new` 把旧的 `d` 重缩放，再加入新项，最终 `softmax(x_i) = exp(x_i - m_final) / d_final`。
>
> ⚠️ 本示例假设 **blockDim.x ≤ 32**（单 warp 处理一行），因此 warp shuffle 即可完成跨线程归约。若 `blockDim.x > 32`，需要再加一层 SMEM 的跨 warp 归约，参考 §6.2.2 的写法。

![[GPU/Drawings/Online Softmax 与 Flash Attention 状态更新.svg]]

可编辑源图：[[GPU/Drawings/Online Softmax 与 Flash Attention 状态更新.excalidraw]]

```cpp
// 避免 overflow 的在线 softmax，单次遍历完成
// 前提：blockDim.x == 32（单 warp / row）
__global__ void online_softmax(float* input, float* output, int seq_len) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    // 在线算法：同时跟踪 max 和 sum
    float max_val = -INFINITY;
    float sum = 0.0f;

    for (int i = lane; i < seq_len; i += blockDim.x) {
        float new_max = fmaxf(max_val, row_input[i]);
        float correction = expf(max_val - new_max);
        sum = sum * correction + expf(row_input[i] - new_max);
        max_val = new_max;
    }

    // Warp 归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum, offset);

        float new_max = fmaxf(max_val, other_max);
        float correction1 = expf(max_val - new_max);
        float correction2 = expf(other_max - new_max);

        sum = sum * correction1 + other_sum * correction2;
        max_val = new_max;
    }

    // 广播到 Warp 内所有线程
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    sum = __shfl_sync(0xffffffff, sum, 0);

    // 输出
    for (int i = lane; i < seq_len; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - max_val) / sum;
    }
}
```

---

## 6.3 LayerNorm Kernel

### 6.3.1 基础实现

```cpp
__global__ void layernorm_naive(float* input, float* output,
                                 float* gamma, float* beta,
                                 int batch_size, int hidden_size) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    if (batch >= batch_size) return;

    float* in = input + batch * hidden_size;
    float* out = output + batch * hidden_size;

    // 计算 mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += in[i];
    }
    sum = warp_reduce_sum(sum);
    float mean = sum / hidden_size;

    // 计算 variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = in[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    float var = var_sum / hidden_size;
    float inv_std = rsqrtf(var + 1e-5f);

    // 归一化
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

> **正确性边界（重要）**：上面的 `layernorm_naive` 在 **`blockDim.x ≤ 32`**（即只有 1 个 warp）时才正确——一个 warp 内的 `warp_reduce_sum` 能覆盖全部数据，且 lane 0 写回时其他 lane 还在用错的 mean/var。要支持 `blockDim.x > 32`，必须：
> 1. 跨 warp 归约：用 shared memory 把每个 warp 的局部和汇总（见 §6.3.2 的写法）；
> 2. 把 `mean`/`inv_std` 通过 shared memory 广播给整个 block，再做归一化。
>
> 课程示例之所以保留这个"看似简洁"的 naive 版本，是为了让你**先看到 bug 再看修复**——如果直接复制到 `blockDim.x = 256` 的 launch 上跑会得到错误结果。

### 6.3.2 高性能版本

```cpp
template<int THREADS_PER_BLOCK>
__global__ void layernorm_optimized(float* __restrict__ input,
                                     float* __restrict__ output,
                                     float* __restrict__ gamma,
                                     float* __restrict__ beta,
                                     int hidden_size) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    float* in = input + batch * hidden_size;
    float* out = output + batch * hidden_size;

    // 使用共享内存存储中间结果
    __shared__ float shared_sum[THREADS_PER_BLOCK / 32];
    __shared__ float shared_var[THREADS_PER_BLOCK / 32];

    // 计算 mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += THREADS_PER_BLOCK) {
        sum += in[i];
    }
    sum = warp_reduce_sum(sum);

    int warp_id = tid / 32;
    if (tid % 32 == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    // 跨 Warp 归约
    if (tid < THREADS_PER_BLOCK / 32) {
        sum = shared_sum[tid];
    }
    if (warp_id == 0) {
        sum = warp_reduce_sum(sum);
    }
    float mean = sum / hidden_size;

    // 计算 variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += THREADS_PER_BLOCK) {
        float diff = in[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);

    if (tid % 32 == 0) {
        shared_var[warp_id] = var_sum;
    }
    __syncthreads();

    if (tid < THREADS_PER_BLOCK / 32) {
        var_sum = shared_var[tid];
    }
    if (warp_id == 0) {
        var_sum = warp_reduce_sum(var_sum);
    }
    float inv_std = rsqrtf(var_sum / hidden_size + 1e-5f);

    // 广播 mean 和 inv_std
    __shared__ float shared_mean, shared_inv_std;
    if (tid == 0) {
        shared_mean = mean;
        shared_inv_std = inv_std;
    }
    __syncthreads();

    // 归一化
    for (int i = tid; i < hidden_size; i += THREADS_PER_BLOCK) {
        out[i] = (in[i] - shared_mean) * shared_inv_std * gamma[i] + beta[i];
    }
}
```

---

## 6.4 Flash Attention

Flash Attention 是大模型推理的核心优化技术，通过分块计算和重计算减少内存访问。

### 6.4.1 Flash Attention 核心思想

```text
传统 Attention: O(N²) 显存，需要存储完整的 attention matrix
Flash Attention: O(N) 显存，分块计算，不存储中间结果

关键思想:
1. 分块计算，避免存储完整的 attention matrix
2. 使用在线 softmax 算法
3. 重计算策略（用计算换内存）
```

### 6.4.2 Flash Attention V2 实现

> **⚠️ 重要警告（2026-05-30 修订）**：下面的代码是**说明性伪代码**，**不能直接复制编译运行**，存在以下已知正确性问题，读者按字面理解即可：
>
> 1. **`O_local[HEAD_DIM]` 每线程一份完整 HEAD_DIM 数组**，但内层用 `for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)` 按 lane 跨步访问 `O_local[d]`——多个线程会写到自己数组的不同 index，但每个线程实际只拥有 HEAD_DIM 个槽位中的 `HEAD_DIM/blockDim.x` 个"真正应该负责"的位置，索引语义混乱；
> 2. **m_new / sum_exp 在跨 warp / 跨线程时未做 reduce + broadcast**，每个 q_row 的状态需要在所有参与的线程间一致，否则 `correction = expf(max_val - new_max)` 中 `max_val` 是 stale 的；
> 3. **`block_sum` 同样需要跨参与维度归约**，否则 sum_exp 是部分和。
>
> 真正可运行的 Flash Attention V2 实现见 [FlashAttention-2 paper / official repo](https://github.com/Dao-AILab/flash-attention)，工程上以 CUTLASS + Hopper TMA/WGMMA 为底层。本课程示意只用于建立"分块 + online softmax + 重计算"的心智模型。

```cpp
// Flash Attention V2 - 说明性伪代码（不可直接运行，见上面警告）
// Q: (batch, heads, seq_len, head_dim)
// K: (batch, heads, seq_len, head_dim)
// V: (batch, heads, seq_len, head_dim)
// O: (batch, heads, seq_len, head_dim)

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_kernel_pseudocode(
    float* Q, float* K, float* V, float* O,
    int batch_size, int num_heads, int seq_len
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int block_row = blockIdx.x;

    int q_offset = (batch * num_heads + head) * seq_len * HEAD_DIM;
    float* Q_block = Q + q_offset + block_row * BLOCK_SIZE * HEAD_DIM;
    float* K_block = K + q_offset;
    float* V_block = V + q_offset;
    float* O_block = O + q_offset + block_row * BLOCK_SIZE * HEAD_DIM;

    // 共享内存
    __shared__ float Q_s[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_s[BLOCK_SIZE][HEAD_DIM];
    __shared__ float V_s[BLOCK_SIZE][HEAD_DIM];

    // 加载 Q 块到共享内存
    for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_row = block_row * BLOCK_SIZE + row;
        if (q_row < seq_len) {
            Q_s[row][col] = Q_block[i];
        } else {
            Q_s[row][col] = 0.0f;
        }
    }

    // ⚠️ 概念上：每个 q_row 应有独立的 (max_val, sum_exp, O_local[HEAD_DIM]) 状态，
    //    并且 O_local 应该放在 shared memory 中由该 q_row 对应的一组线程共同维护，
    //    而非简单的"每个线程的私有 HEAD_DIM 数组"
    float O_local[HEAD_DIM] = {0.0f};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // 遍历 K/V 块
    for (int block_col = 0; block_col < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; block_col++) {
        // 加载 K, V 块
        for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int k_row = block_col * BLOCK_SIZE + row;
            if (k_row < seq_len) {
                K_s[row][col] = K_block[k_row * HEAD_DIM + col];
                V_s[row][col] = V_block[k_row * HEAD_DIM + col];
            } else {
                K_s[row][col] = 0.0f;
                V_s[row][col] = 0.0f;
            }
        }
        __syncthreads();

        // 计算 Q × K^T（点积部分用 warp reduce 后需 broadcast，此处省略）
        for (int q_row = threadIdx.y; q_row < BLOCK_SIZE; q_row += blockDim.y) {
            float scores[BLOCK_SIZE];
            float row_max = -INFINITY;

            for (int k_row = 0; k_row < BLOCK_SIZE; k_row++) {
                float score = 0.0f;
                for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                    score += Q_s[q_row][d] * K_s[k_row][d];
                }
                // 概念性：warp reduce 后必须 broadcast 给所有 lane
                score = warp_reduce_sum(score);
                score = __shfl_sync(0xffffffff, score, 0);  // broadcast
                score /= sqrtf((float)HEAD_DIM);
                scores[k_row] = score;
                row_max = fmaxf(row_max, score);
            }

            // 在线 softmax 状态更新（概念性，跨线程需 reduce + broadcast）
            float new_max = fmaxf(max_val, row_max);
            float correction = expf(max_val - new_max);

            float block_sum = 0.0f;
            for (int k_row = 0; k_row < BLOCK_SIZE; k_row++) {
                block_sum += expf(scores[k_row] - new_max);
            }
            float new_sum = sum_exp * correction + block_sum;

            // 更新输出（O_local 索引语义见上方警告）
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                float o_val = O_local[d] * correction;
                for (int k_row = 0; k_row < BLOCK_SIZE; k_row++) {
                    o_val += expf(scores[k_row] - new_max) * V_s[k_row][d];
                }
                O_local[d] = o_val;  // ⚠️ 越界风险
            }

            max_val = new_max;
            sum_exp = new_sum;
        }
        __syncthreads();
    }

    // 归一化输出
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        O_local[d] /= sum_exp;
    }

    // 写回全局内存
    for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_row = block_row * BLOCK_SIZE + row;
        if (q_row < seq_len) {
            O_block[i] = O_local[col];
        }
    }
}
```

---

## 6.5 RoPE（旋转位置编码）

### 6.5.1 RoPE 原理

RoPE 通过旋转矩阵编码位置信息：
- 将位置编码为旋转角度
- 应用到 query 和 key
- 保持相对位置信息

### 6.5.2 RoPE 实现

```cpp
__device__ void apply_rotary_emb(
    float* query, float* key,
    int head_dim, int pos, float rope_theta
) {
    // 每两个维度一组进行旋转
    for (int i = 0; i < head_dim; i += 2) {
        float freq = pos / powf(rope_theta, (float)(i / 2) / head_dim);
        float cos_freq = cosf(freq);
        float sin_freq = sinf(freq);

        // 旋转 query
        float q0 = query[i];
        float q1 = query[i + 1];
        query[i] = q0 * cos_freq - q1 * sin_freq;
        query[i + 1] = q0 * sin_freq + q1 * cos_freq;

        // 旋转 key
        float k0 = key[i];
        float k1 = key[i + 1];
        key[i] = k0 * cos_freq - k1 * sin_freq;
        key[i + 1] = k0 * sin_freq + k1 * cos_freq;
    }
}

__global__ void rope_kernel(
    float* query, float* key,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float rope_theta
) {
    // 注意：此实现采用 RoFormer 原始论文的"相邻对"约定（[i, i+1] 配对）。
    // LLaMA / HuggingFace 的常见实现使用"half-rotation"约定（[i, i + head_dim/2] 配对）。
    // 两种约定数学上等价但内存布局不同，对接预训练权重时需匹配模型自身的约定，
    // 否则 attention 输出在数值上正确但语义错位。

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq = blockIdx.z;

    int offset = (batch * num_heads + head) * seq_len * head_dim;
    float* q = query + offset + seq * head_dim;
    float* k = key + offset + seq * head_dim;

    // 每个线程处理一对维度（i, i+1）
    for (int i = threadIdx.x * 2; i < head_dim; i += blockDim.x * 2) {
        float freq = seq / powf(rope_theta, (float)(i / 2) / head_dim);
        float cos_freq = cosf(freq);
        float sin_freq = sinf(freq);

        // 旋转 query
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i] = q0 * cos_freq - q1 * sin_freq;
        q[i + 1] = q0 * sin_freq + q1 * cos_freq;

        // 旋转 key
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i] = k0 * cos_freq - k1 * sin_freq;
        k[i + 1] = k0 * sin_freq + k1 * cos_freq;
    }
}
```

---

## 6.6 KV Cache

### 6.6.1 KV Cache 原理

在自回归生成过程中：
- 每次生成一个 token
- 需要所有之前 token 的 K、V
- 缓存避免重复计算

KV Cache 结构：
- K_cache: (batch, heads, max_seq_len, head_dim)
- V_cache: (batch, heads, max_seq_len, head_dim)

### 6.6.2 KV Cache 实现

```cpp
// 更新 KV Cache
__global__ void update_kv_cache(
    float* K_cache, float* V_cache,
    float* K_new, float* V_new,
    int batch_size, int num_heads, int head_dim,
    int cache_len, int new_len
) {
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq = blockIdx.z;

    if (seq >= new_len) return;

    int cache_offset = (batch * num_heads + head) * cache_len * head_dim;
    int new_offset = (batch * num_heads + head) * new_len * head_dim;

    int pos = cache_len + seq;  // 新 token 的位置

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        K_cache[cache_offset + pos * head_dim + d] = K_new[new_offset + seq * head_dim + d];
        V_cache[cache_offset + pos * head_dim + d] = V_new[new_offset + seq * head_dim + d];
    }
    // 如果下一个 kernel 在同一 grid 中通过 cooperative groups 立即读取 K_cache / V_cache，
    // 需要 __threadfence() 保证写后对全局可见。如果下一个 kernel 是
    // 通过 cudaStream 顺序发起（标准做法），CUDA stream acquire/release 已经保证可见性，
    // 这里可省略 fence。
}

// 从 KV Cache 读取
__global__ void read_kv_cache(
    float* K_out, float* V_out,
    float* K_cache, float* V_cache,
    int batch_size, int num_heads, int head_dim,
    int seq_len
) {
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq = blockIdx.z;

    if (seq >= seq_len) return;

    int offset = (batch * num_heads + head) * seq_len * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        K_out[offset + seq * head_dim + d] = K_cache[offset + seq * head_dim + d];
        V_out[offset + seq * head_dim + d] = V_cache[offset + seq * head_dim + d];
    }
}
```

---

## 6.7 其他常用 Kernel

### 6.7.1 RMSNorm

> ℹ️ 标准 LLaMA RMSNorm：`y = x * rsqrt(mean(x²) + eps) * weight`。注意 `eps` 必须**加在平方根内部**，写成 `rsqrtf(sum_sq / N + eps)`；常见 bug 是写成 `rsqrtf(sum_sq / N) + eps`，会破坏数值稳定性。下面示例假设 `blockDim.x ≤ 32`（单 warp），如需更大 block 与 §6.3.2 一样追加 SMEM 跨 warp 归约。LLaMA 原版 RMSNorm **没有 bias**，这里 `bias` 仅为兼容其他变体。

```cpp
__global__ void rmsnorm_kernel(
    float* input, float* output,
    float* weight, float* bias,
    int batch_size, int hidden_size,
    float epsilon = 1e-5f
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    float* in = input + batch * hidden_size;
    float* out = output + batch * hidden_size;

    // 计算 sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum_sq += in[i] * in[i];
    }
    sum_sq = warp_reduce_sum(sum_sq);
    // 广播到 warp 内所有线程
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    // eps 必须在 sqrt 内部
    float rms = rsqrtf(sum_sq / hidden_size + epsilon);

    // 归一化
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out[i] = in[i] * rms * weight[i] + (bias ? bias[i] : 0.0f);
    }
}
```

### 6.7.2 SwiGLU 激活函数

```cpp
__global__ void swiglu_kernel(
    float* gate, float* up, float* output,
    int batch_size, int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_size;

    if (idx < total) {
        // SwiGLU = gate × sigmoid(gate) × up
        float g = gate[idx];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));  // SiLU/Swish
        output[idx] = g * sigmoid_g * up[idx];
    }
}
```

### 6.7.3 GEMV（矩阵向量乘）

推理时 `y = A · x`（A 形状 `[M, K]`，x 形状 `[K]`）退化为 GEMV。**GEMV 是带宽 bound**，roofline 上限不是 FLOPS 而是 `2 · M · K · sizeof(T) / 带宽`，所以衡量性能应该用 **GB/s** 而非 GFLOPS。

```cpp
// 每个 block 处理一行 A（blockDim.x ≤ 32，单 warp 归约）
// 若需要更大 blockDim.x，参考 §6.3.2 加 SMEM 跨 warp 归约
template<int TILE_SIZE>
__global__ void gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    __shared__ float x_shared[TILE_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作把 x 的 tile 加载到共享内存
        int col = t * TILE_SIZE + tid;
        if (tid < TILE_SIZE && col < K) {
            x_shared[tid] = x[col];
        }
        __syncthreads();

        // 计算：每个线程沿 K 维步长 blockDim.x
        for (int k = tid; k < TILE_SIZE && t * TILE_SIZE + k < K; k += blockDim.x) {
            sum += A[row * K + t * TILE_SIZE + k] * x_shared[k];
        }
        __syncthreads();
    }

    // Warp 内归约
    sum = warp_reduce_sum(sum);

    // 仅 lane 0 写回 y[row]；多 warp 时需先 SMEM 跨 warp 归约再写
    if (tid == 0) {
        y[row] = sum;
    }
}
```

> ⚠️ **常见 bug**：原始写法 `if (tid % 32 == 0)` 在 `blockDim.x > 32` 时多个 warp 的 lane 0 会**同时写同一个 `y[row]`**，互相覆盖。正确做法要么限制单 warp（`tid == 0`），要么先用 SMEM 把各 warp 的部分和归约到 lane 0 再写。

---

## 6.8 性能对比

> ⚠️ 下表是**示意数量级**，仅用于体现"naive → 优化 → 厂商库"几倍量级的关系，**不是 benchmark 报告**。真实数字取决于硬件型号、精度、形状、是否启用 Tensor Core / 稀疏，需自行用 NCU 实测。详见 [[CUDA Kernel 性能瓶颈定位流程]]。

下列数字以 **A100 SXM 80GB（dense, FP16, Tensor Core 156 TFLOPS / HBM 2 TB/s）** 为参照。

| Kernel | 类型 | Naive | 优化版 | cuBLAS/cuDNN | 备注 |
|--------|------|-------|--------|--------------|------|
| GEMM (FP16) | compute bound | ~500 GFLOPS | ~80 TFLOPS | ~120 TFLOPS | dense；Tensor Core dense 峰值 156 TFLOPS |
| Softmax | bandwidth bound | ~50 GB/s | ~1.2 TB/s | — | 峰值约 HBM 带宽 2 TB/s |
| LayerNorm | bandwidth bound | ~100 GB/s | ~1.5 TB/s | — | 同上 |
| Flash Attention V2 (FP16) | compute + bandwidth | — | ~50 TFLOPS | — | head_dim=128, seq=2048 |
| GEMV (FP16) | bandwidth bound | ~200 GB/s | ~1.6 TB/s | ~1.8 TB/s (cuBLAS) | M=K=4096，roofline 上限 ≈ HBM 带宽 |

口径说明：
- GEMM/FlashAttention 用 **TFLOPS（dense）**；
- Softmax/LayerNorm/GEMV 用 **GB/s**（带宽 bound）；
- Tensor Core sparse 数值约为 dense 的 2×，本表统一 dense 口径，详见 [[Hardware/NVIDIA GPU 架构与规格]] 的"代际峰值算力（dense/sparse 统一口径）"表。

---

## 💡 本章要点

1. **GEMM** 是大模型推理的核心，使用 Tensor Core 可以大幅加速
2. **Softmax** 要注意数值稳定性，使用 Online Softmax 单次遍历
3. **LayerNorm** 使用 Warp 级原语减少同步开销
4. **Flash Attention** 用计算换内存，避免存储完整 attention matrix
5. **RoPE** 通过旋转编码位置信息，支持任意长度序列
6. **KV Cache** 避免重复计算，是自回归生成的关键优化
7. **RMSNorm** 和 **SwiGLU** 是 LLaMA 等新模型的常用组件

---

## 实战练习

1. 实现 Softmax 并优化到接近理论峰值
2. 实现 LayerNorm 并对比不同实现方式的性能
3. 实现 Flash Attention 并分析内存带宽节省
4. 实现完整的 Attention 层（QKV 投影 + RoPE + Flash Attention + 输出投影）
5. 实现 KV Cache 管理，支持动态增长

---

[← 上一章：基础 Kernel](第5章-基础Kernel.md) | [下一章：高级主题 →](第7章-高级主题.md)
