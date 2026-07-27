---
updated: 2026-05-17
tags:
  - gpu-computing
  - learning-roadmap
  - concept-note
---
# GPU 学习任务

## 任务入口

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA Kernel 示例：归约求和]]

## Kernel 主题路线

| 主题 | 当前入口 | 为什么重要 | 第一版应产出 |
|------|----------|------------|--------------|
| online softmax | 待建主文档 | FlashAttention 的核心稳定性技巧，避免显式存完整 attention score | 推导 `m_i` / `l_i` 递推公式，给出 block-wise softmax 伪代码 |
| GEMM | [[CUDA GEMM 矩阵乘法优化指南]] | 大模型中线性层、MLP、QKV projection 的主要算力来源 | 补 CUTLASS/CuTe 最小 GEMM 示例，并结合 [[Nsight Compute NCU 分析方法与优化思路]] 记录指标 |
| MHA | 待建主文档 | attention 是推理长上下文的核心瓶颈 | 拆成 QK、mask、softmax、PV、KV cache，关联 online softmax |
| reduce | [[CUDA Kernel 示例：归约求和]] | norm、softmax、统计量计算都依赖 reduction | 对比 block reduce、warp reduce、CUB / Thrust reduce |
| RMSNorm | 待建主文档 | LLM decoder 层常见算子，典型 memory-bound kernel | 写清 `sum(x^2)`、`rsqrt`、scale 三步和向量化访存 |
| MLP | 待建主文档 | LLM 中常见 GEMM + activation + GEMM 结构 | 梳理 SwiGLU / GeLU、activation fusion、权重量化入口 |

## 主题细化

### Online Softmax

标准 softmax 需要先拿到整行最大值，再做指数求和：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

online softmax 的价值是分块扫描时维护当前最大值 `m` 和归一化项 `l`，每看到一个新 tile 就更新：

```text
m_new = max(m_old, max(tile))
l_new = l_old * exp(m_old - m_new) + sum(exp(tile - m_new))
```

这样可以在不物化完整 attention score 矩阵的情况下稳定计算 softmax，是 FlashAttention 类算法的基础。

### MHA / Attention

第一版笔记建议按算子流拆：

```text
Q, K, V projection
  -> QK^T / sqrt(d)
  -> mask / causal mask
  -> softmax
  -> P V
  -> output projection
```

推理侧还要单独记录 KV cache 的内存布局、page/block 管理、GQA/MQA 对带宽的影响。

### RMSNorm

RMSNorm 的核心公式：

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

这是一个典型的 memory-bound kernel。优化重点通常是：

- 每行一个或多个 block。
- 向量化加载 `float4` / `half2`。
- block 内 reduction 计算 `sum(x^2)`。
- 尽量把归一化和 scale 写回融合在同一个 kernel。

### MLP

LLM MLP 常见结构是：

```text
up = X W_up
gate = X W_gate
hidden = activation(gate) * up
out = hidden W_down
```

后续笔记可以重点看 GEMM 性能、activation fusion、量化权重布局，以及 tensor parallel 下的通信边界。

## 下一批文档建议

1. `CUDA Online Softmax 与 FlashAttention 基础`
2. `CUDA RMSNorm Kernel 优化`
3. `Attention Kernel 拆解：QK、Softmax、PV、KV Cache`
4. `CUTLASS CuTe 最小 GEMM 示例`
5. [[Nsight Compute NCU 分析方法与优化思路]]
