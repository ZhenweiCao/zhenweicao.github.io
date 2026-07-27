---
aliases:
  - GPU 通信
  - GPUDirect RDMA
  - NVLink 通信
updated: 2026-06-15
tags:
  - gpu-computing
  - gpu-architecture
  - rdma-networking
  - distributed-communication
---
# GPU 通信机制与优化技术

> 目标：把 GPU 多卡/多机通信的硬件通路、软件栈和常见优化手段串成一张图。查具体 NVLink 带宽、NVL72 规格时回到 [[NVIDIA GPU 架构与规格]]；做 kernel 内片内协作看 [[CUDA CTA 与 Thread Block Cluster 入门]]；做推理工程排障可对照 [[DeepEP 通信物理路径与共享内存]] 和 [[NVSHMEM_IB_GID_INDEX]]。

## 本文怎么读

- 先读总览图，建立「片内 → 同机多卡 → 跨机」三层心智模型。
- 再分别理解 **NVLink**（同机/同 rack 高带宽域）和 **GPUDirect RDMA**（跨机 GPU 显存直达网卡）。
- 最后看软件栈（NCCL / NVSHMEM / P2P API）和优化 checklist。

## 通信层次总览

![[GPU/Drawings/GPU 通信机制总览.svg]]

可编辑源图：[[GPU/Drawings/GPU 通信机制总览.excalidraw]]

![[GPU/Drawings/GPU 通信路径选择流程.svg]]

| 层次 | 典型场景 | 硬件通路 | 带宽量级（示意） | 软件入口 |
|------|----------|----------|------------------|----------|
| 片内 | 单卡 GEMM、Attention、Cluster DSMEM | SM ↔ L2 ↔ HBM；Cluster 内 CTA 经 SM-to-SM fabric 访问彼此 SMEM | HBM 数 TB/s；DSMEM 不走 NVLink | CUDA kernel、`cluster.sync()`、TMA multicast |
| 同机多卡 | TP/EP 8 卡 AllReduce、P2P KV 拷贝 | NVLink（优先）或 PCIe P2P | NVLink 数百 GB/s～TB/s 级（见规格表） | `cudaMemcpyPeer`、`cudaDeviceCanAccessPeer`、NCCL |
| 跨机 | 多节点 TP/EP、PD KV transfer、MoE A2A | RoCE/InfiniBand + GPUDirect RDMA | 单 NIC 100–400 Gb/s 量级 | NCCL、NVSHMEM、Mooncake/NIXL、DeepEP |

**关键区分**：Thread Block Cluster 的 DSMEM 是**单 GPU 片内** SM 间通信，**不走 NVLink**；NVLink 连接的是**不同 GPU**（或 Grace CPU 与 GPU 的 NVLink-C2C）。

## NVLink 通信基本原理

![[GPU/Drawings/NVLink 通信基本原理.svg]]

可编辑源图：[[GPU/Drawings/NVLink 通信基本原理.excalidraw]]

### 是什么

NVLink 是 NVIDIA GPU 之间（以及 Grace CPU 与 GPU 之间）的**点对点高速互联**，在物理层提供远高于 PCIe 的带宽和更低延迟。多卡服务器里，GPU 通过 NVLink 组成 **NVLink domain**；更大规模（如 NVL72 rack）通过 **NVSwitch** 把多块 GPU 拉进同一个全互联域。

### 数据通路（概念）

```text
GPU A 显存
  → GPU A NVLink 控制器
  → NVLink 线缆 / NVSwitch 交换 fabric
  → GPU B NVLink 控制器
  → GPU B 显存
```

与 PCIe P2P 对比：PCIe 要经过 root complex，多卡争用同一 PCIe 树时带宽和延迟都更差；同机多卡训练/推理默认应让 NCCL 走 NVLink 拓扑。

### 几种常见拓扑

| 形态 | 例子 | 特点 |
|------|------|------|
| 直连 | 2–4 卡 DGX 部分配置 | GPU 两两 NVLink 直连，domain 较小 |
| NVSwitch 全互联 | DGX H100/B200 8 卡、GB200 NVL72 | 任意 GPU 对之间经 switch 可达，aggregate 带宽见 [[NVIDIA GPU 架构与规格]] |
| NVLink-C2C | GB200 Grace Blackwell Superchip | 1 Grace CPU + 2 GPU，CPU↔GPU 统一内存语义，延迟低于传统 PCIe + 离散 CPU |

### 软件如何用到 NVLink

1. **CUDA P2P**：`cudaDeviceCanAccessPeer` 为 true 时，`cudaMemcpyPeerAsync` 可走 NVLink（若拓扑支持）。
2. **NCCL**：初始化时探测拓扑，AllReduce/AllGather 等 collective 自动选 NVLink > PCIe > NET 路径。
3. **UVM / 统一寻址**：多卡 P2P 映射后，kernel 可通过指针直接 load/store 远端 GPU 显存（性能敏感路径仍建议显式 memcpy 或 collective）。

### 优化要点

- **进程–GPU 亲和**：TP rank 与物理 GPU 编号对齐，避免跨 NUMA 或绕路 PCIe。
- **拓扑感知放置**：同 NVLink domain 内的 worker 做高频 collective；跨 domain 通信成本陡增。
- **不要混淆 DSMEM 与 NVLink**：kernel 内的 cluster 协作是片内优化；多卡同步仍靠 NCCL / 自定义 P2P。

## GPUDirect RDMA 基本原理

![[GPU/Drawings/GPUDirect RDMA 数据通路.svg]]

可编辑源图：[[GPU/Drawings/GPUDirect RDMA 数据通路.excalidraw]]

### 解决什么问题

跨机传输时，若数据在 **GPU 显存**，传统路径需要：

```text
GPU VRAM → DMA 到 Host pinned memory → RDMA NIC → 网络 → 对端 NIC → Host → GPU VRAM
```

多次拷贝、占用 CPU、抬高延迟。**GPUDirect RDMA（GDR）** 让 RDMA 网卡（Mellanox/NVIDIA ConnectX）通过 PCIe peer-to-peer / peer memory 机制，把 GPU 显存注册为 RDMA Memory Region，实现：

```text
GPU VRAM
  -> PCIe BAR / peer mapping
  -> RDMA NIC
  -> IB / RoCE 网络
  -> 对端 RDMA NIC
  -> 对端 PCIe BAR / peer mapping
  -> 对端 GPU VRAM
```

数据面绕过 host staging，是 Mooncake/NIXL KV transfer、NVSHMEM IBGDA、DeepEP 跨机路径的性能基础。

> [!NOTE] 口径提醒
> GPUDirect RDMA 不是一种新的网络协议，也不是 GPU-GPU 同机互联。它解决的是 **RDMA 设备如何直接访问 GPU memory**。真正的网络语义仍由 verbs / NCCL / NVSHMEM / Mooncake / NIXL / DeepEP 等上层栈决定；同机 GPU 之间优先走 NVLink / PCIe P2P，不经过 IB NIC。

### 关键组件

| 组件 | 作用 |
|------|------|
| NVIDIA GPU driver | 识别 CUDA UVA 中的 GPU pointer，并导出 GPU 显存映射。 |
| RDMA HCA / NIC | ConnectX / BlueField 等设备，发起 PCIe DMA 和 IB/RoCE 网络传输。 |
| RDMA 栈 | `ibverbs` / `rdma-core` / MLNX_OFED / DOCA-OFED，负责 QP、MR、CQ 等 RDMA 对象。 |
| kernel-mode GDR 路径 | Linux DMA-BUF 或 legacy `nvidia-peermem`，让 RDMA 设备拿到 GPU memory 的可 DMA 映射。 |
| 应用/中间件 | NCCL NET plugin、NVSHMEM、Mooncake、NIXL、DeepEP 等，负责 buffer 注册、连接建立、协议和调度。 |

### 控制面与数据面

![[GPU/Drawings/GPUDirect RDMA 控制面与数据面流程.svg]]

GDR 的“零拷贝”主要说的是**数据面**，不是说完全没有 CPU：

| 阶段 | 谁参与 | 做什么 |
|------|--------|--------|
| 控制面：初始化 | CPU、CUDA runtime、RDMA verbs、驱动 | 分配 GPU buffer、识别 pointer、注册 MR、创建 QP/CQ、交换 rkey/addr、建立连接。 |
| 控制面：提交传输 | CPU 或 GPU | CPU-initiated 路径由 CPU post WQE；NVSHMEM IBGDA 这类路径可由 GPU kernel 触发 NIC doorbell。 |
| 数据面：传输 | NIC DMA engine、PCIe、网络 | NIC 直接读写 GPU BAR 映射后的显存，不需要先搬到 host pinned buffer。 |
| 完成与同步 | CQ / event / CUDA sync | 通信完成后，要保证 CUDA kernel 读写这段显存前有正确 ordering。 |

所以更准确的说法是：

```text
传统跨机 GPU 通信：
  GPU -> Host bounce buffer -> NIC -> 网络 -> NIC -> Host bounce buffer -> GPU

GPUDirect RDMA：
  GPU memory 注册成 RDMA MR
  NIC 直接 DMA 读写 GPU memory
  CPU 主要留在控制面
```

### 真实工作流程

以一个跨节点 `RDMA Write` 把本机 GPU buffer 写到远端 GPU buffer 为例：

```text
1. 本机进程 cudaMalloc(src_gpu_ptr)
2. 远端进程 cudaMalloc(dst_gpu_ptr)
3. 通信库识别 src/dst 是 CUDA GPU pointer
4. 双方把 GPU buffer 注册成 RDMA Memory Region
5. 交换 QP 信息、rkey、remote address、GID/LID 等连接信息
6. 本机提交 RDMA Write
7. 本机 NIC 经 PCIe P2P 读取 src GPU memory
8. 数据经 IB/RoCE 网络到远端 NIC
9. 远端 NIC 经 PCIe P2P 写入 dst GPU memory
10. CQ / event 报告完成；后续 GPU kernel 通过 CUDA 同步边界消费数据
```

核心不是“GPU 自己在网上发包”，而是**这段 GPU memory 已经被 NIC 合法注册和映射**。谁来触发传输要看上层：

| 触发方式 | 典型栈 | 含义 |
|----------|--------|------|
| CPU-initiated RDMA | NCCL NET、MPI、Mooncake/NIXL 的常见 verbs 路径 | CPU 线程 post send/write/read；NIC 数据面直接访问 GPU memory。 |
| GPU-initiated RDMA | NVSHMEM IBGDA、DeepEP legacy low-latency 路径 | GPU kernel 直接触发 NIC 传输，减少 CPU proxy 和 launch/轮询开销。 |

### `nvidia-peermem` 工作流

legacy 路径通过 NVIDIA 私有 NV-P2P API 和 RDMA peer-memory client 把 GPU 显存接入 RDMA：

```text
User App
  -> cudaMalloc()
  -> ibv_reg_mr(gpu_ptr)
  -> libibverbs / ib_uverbs
  -> mlx5_ib
  -> nvidia-peermem
  -> NVIDIA GPU kernel driver
  -> pin GPU pages + 建立 BAR / DMA 映射
  -> mlx5_ib program MTT/MKEY
  -> NIC 可直接 DMA 读写 GPU memory
```

和普通 CPU memory 的区别在于：CPU buffer 可以走 Linux `get_user_pages()` pin 住页；GPU memory 没有普通 CPU page frame，必须由 NVIDIA 驱动提供 `nvidia_p2p_get_pages()` / DMA mapping 之类的 GPU peer mapping 信息。

### DMA-BUF 工作流

新路径把 GPU memory 作为 Linux upstream 的 dma-buf 对象共享：

```text
User App
  -> cudaMalloc() / CUDA driver allocation
  -> CUDA / NVIDIA driver 导出 dma-buf fd
  -> RDMA driver 作为 importer 导入 fd
  -> 注册为可用于 verbs 的 Memory Region
  -> NIC 可直接 DMA 读写 GPU-backed buffer
```

这条路径把“GPU driver 导出 buffer、RDMA driver 导入 buffer”放进 Linux 标准 exporter/importer 模型，减少对私有 peer-memory 胶水层的依赖。NVIDIA GPU Operator 当前也明确推荐优先使用 DMA-BUF，而不是 legacy `nvidia-peermem`。

### `nvidia-peermem` vs DMA-BUF

![[GPU/Drawings/GPUDirect RDMA peermem 与 DMA-BUF 工作流.svg]]

| 维度 | `nvidia-peermem` legacy | DMA-BUF 推荐方向 |
|------|--------------------------|------------------|
| 内核接口 | NVIDIA NV-P2P API + RDMA peer-memory client。 | Linux dma-buf exporter / importer。 |
| GPU 侧角色 | NVIDIA driver 返回 GPU page table / BAR mapping。 | NVIDIA driver 导出 dma-buf fd。 |
| RDMA 侧角色 | `mlx5_ib` 通过 peer-memory client 注册 GPU memory。 | RDMA driver 导入 dma-buf 并注册 MR。 |
| 驱动依赖 | 依赖 MLNX_OFED / DOCA-OFED 等提供 peer-memory 支持。 | 可用 Linux inbox driver；MLNX_OFED / DOCA-OFED 变成可选。 |
| NVIDIA GPU driver | 任意受支持 driver 分支，历史兼容性好。 | 需要 NVIDIA Open GPU Kernel Module。 |
| CUDA / kernel 前提 | CUDA 版本前提较宽松。 | CUDA 11.7+，Linux kernel 5.12+。 |
| 运维动作 | 常见为加载 `nvidia-peermem`；旧 `nv_peer_mem` 不能与它同时加载。 | 不需要 `nvidia-peermem`；perftest 可用 `--use_cuda_dmabuf` 验证。 |
| 生命周期 | 依赖 NV-P2P invalidation callback，第三方驱动要处理 revoke。 | 依赖 dma-buf 共享对象语义，生命周期更接近 Linux 通用框架。 |
| 适用判断 | 老系统、老内核、既有 OFED 栈常见。 | 新系统、新 GPU Operator、希望跟 Linux upstream 方向对齐。 |

### 关键硬件与系统约束

GDR 能不能工作，常常不是应用层说了算，而是受这些条件限制：

| 约束 | 说明 |
|------|------|
| PCIe 拓扑 | GPU 和 NIC 通常要共享合适的 upstream PCIe root complex；跨 CPU socket、跨不支持 P2P 的 root complex 可能失败或退化。 |
| IOMMU | GPUDirect RDMA 依赖设备看到一致的物理地址；IOMMU 需要关闭或配置为 1:1 / passthrough。 |
| ACS / 虚拟化 | ACS 可能强制 P2P 流量绕 root complex；虚拟化场景要显式允许 P2P、Relax ACS、64-bit MMIO / large BAR。 |
| BAR 空间 | GPU BAR 映射资源有限；频繁 pin/unpin 大量小 buffer 会耗尽 BAR 或引入毫秒级注册成本。 |
| 注册缓存 | 通信库通常做 registration cache / lazy unpin，复用已 pin 的 GPU MR，避免每次传输都进内核注册。 |
| 内存类型 | GDR 主要面向 `cudaMalloc()` 这类 GPU device memory；Unified Memory 与并发 RDMA 有一致性风险。 |
| Memory ordering | RDMA 写入 GPU memory 后，后续 kernel 消费前需要明确同步；并发 GPU kernel 与 NIC 写同一 buffer 是数据竞争。 |

### 启用与验证

```bash
# legacy peermem 路径：检查 peer memory 模块
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'

# 若缺失，加载（视发行版而定）
sudo modprobe nvidia-peermem

# DMA-BUF 路径：perftest 典型验证参数
ib_write_bw --use_cuda=0 --use_cuda_dmabuf -d mlx5_0 -a -F --report_gbits
```

容器环境还需：`/dev/infiniband/*` 透传、`memlock` 足够（`--ulimit memlock=-1`）、必要时 `IPC_LOCK`；否则 MR 注册失败，退化为 host bounce buffer 或直接报错。

排查时优先确认：

```bash
nvidia-smi topo -m        # GPU-NIC / GPU-GPU 拓扑
ibdev2netdev              # HCA 与 netdev 对应关系
ibv_devinfo               # RDMA 设备能力
show_gids                 # RoCE GID index
```

### 与 GPUDirect 家族的关系

| 技术 | 方向 | 典型用途 |
|------|------|----------|
| GPUDirect RDMA | GPU ↔ NIC | 多机 NCCL、KV transfer、NVSHMEM |
| GPUDirect P2P | GPU ↔ GPU（同机） | NVLink/PCIe P2P，见上一节 |
| GPUDirect Storage | GPU ↔ NVMe/存储 | 训练数据直读，推理 checkpoint 加载 |

## 软件通信栈对照

| 层次 | 组件 | 抽象 | 典型问题 |
|------|------|------|----------|
| Collective | NCCL | AllReduce / AllGather / ReduceScatter | hang、带宽低、走错网卡 |
| PGAS / GPU 发起 | NVSHMEM、IBGDA | 远端 GPU 内存 put/get | `ibv_modify_qp failed`、GID/HCA 错误 |
| 应用协议 | DeepEP、Mooncake、NIXL | MoE dispatch、PD KV slot | MR 失败、layout/rank 不一致 |
| 同机 IPC | `cudaIpc*` | 进程间共享 GPU 句柄 | fd 泄漏、设备 ordinal 不一致 |

排障时先判断失败在 **建连 / 内存注册 / collective / 业务 layout** 哪一层，再对症查环境变量（`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`、`NVSHMEM_IB_GID_INDEX` 等）。如果错误出现在 NVSHMEM/IBGDA 初始化阶段，不要立刻断定数据面依赖 `/dev/shm`；很多时候 `/dev/shm` 只是 bootstrap / rendezvous 控制面不足，详见 [[DeepEP 通信物理路径与共享内存]]。

## 常见优化技术

### 1. 拓扑与放置

- 同机 collective 走 NVLink；跨机走 RDMA，且 Prefill/Decode、MoE EP 组尽量落在**同一 RDMA fabric / 同一 ToR**。
- 多 NIC（multi-rail）时绑定 GPU–NIC 亲和，避免「有 8×400G 实际只走一条错误链路」。

### 2. 通信与计算重叠

- **CUDA Graph** 固定 launch 顺序，减少小 kernel 启动开销。
- **自定义 AR（如 symmetric memory、MSCCL++）**：小消息或特殊拓扑下替代 NCCL 默认算法。
- **DeepEP / SBO**：MoE combine 与下一层计算 overlap。

### 3. 数据面零拷贝

- 启用 GDR + 正确 peer memory / DMA-BUF，避免 host staging。
- KV / activation buffer 按 RDMA 对齐分配（页大小、注册粒度）。
- 复用长生命周期通信 buffer，依赖 registration cache / lazy unpin，避免频繁注册 GPU memory。
- 明确同步边界：RDMA 写入完成后，再 launch 消费该 buffer 的 GPU kernel；不要让 NIC 写和 GPU kernel 并发访问同一段内存。

### 4. 集合通信算法选择

- 小消息：LL / LL128（NCCL 内部低延迟协议）。
- 大消息：Ring / Tree / NVLS（NVLink Sharp，视硬件支持）。
- 多节点：关注 rail 数量与 chunk pipeline，用 `NCCL_DEBUG=INFO` 看实际选用的 channel 和算法。

### 5. 环境 checklist（跨机）

```bash
# 网卡与 socket
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0   # 按实际业务网卡

# RoCE GID（需结合 ibv_devinfo / show_gids 选择可达 GID）
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
```

## 与 LLM 推理的对应关系

| 场景 | 主要通信机制 | 备注 |
|------|--------------|------|
| 单卡推理 | 无多卡通信 | 瓶颈在 HBM 带宽与算力 |
| TP 8 同机 | NVLink + NCCL AllReduce | 注意力/MLP 列并行 |
| EP 跨机 MoE | RDMA + NVSHMEM/DeepEP | dispatch/combine token |
| PD 分离 | GDR + Mooncake/NIXL | KV cache 跨 worker |
| 长上下文 CP | NVLink/NCCL 或 DSMEM+collective | 依实现与 CP 策略而定 |

## 相关文档

- [[NVIDIA GPU 架构与规格]] — NVLink 带宽、NVL72 aggregate 数字的唯一来源
- [[GPU 硬件架构背景与编程范式]] — DSMEM / cluster 与多卡通信的边界
- [[CUDA CTA 与 Thread Block Cluster 入门]] — 片内 cluster 协作
- [[DeepEP 通信物理路径与共享内存]] — DeepEP 中 NVLink、CUDA IPC、NVSHMEM/IBGDA、`/dev/shm` 的工程边界
- [[NVSHMEM_IB_GID_INDEX]] — RoCE / GID index / NVSHMEM 环境变量解释
- [[GPU Direct RDMA 工作流程]] — Raw/Weixin 来源整理
- [[NVIDIA 实现 GDR（GPUDirect RDMA）的两种机制：peermem vs DMA-Buf]] — Raw 来源：peermem 与 DMA-BUF 对照
- [[GPU 知识库索引]] — GPU 目录总入口

官方资料：

- [NVIDIA GPUDirect RDMA documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA GPU Operator: GPUDirect RDMA and GPUDirect Storage](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
