---
title: "Modern GPU Programming For MLSys"
content_type: index
maturity: stable
updated: 2026-07-21
lang: en
publish: true
aliases:
  - "Modern GPU Programming For MLSys"
source: https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/8950d661e8499008546e3520c667c1cacec9af21/index.md
source_commit: 8950d661e8499008546e3520c667c1cacec9af21
imported: 2026-07-21
tags:
  - gpu-computing
  - gpu-kernel
  - tutorial-note
  - reference-note
---
# Modern GPU Programming For MLSys

> [!NOTE] Obsidian edition
> This is an English-only, offline reading edition imported from the upstream repository. Sphinx cross-references were converted to Obsidian wikilinks, and interactive pages use the enabled Embed HTML plugin.

Machine learning systems sit at the heart of modern AI workloads. In these systems, performance often comes down to the quality of a small number of GPU kernels. Attention kernels, LLM prefill and decode kernels, low-precision block-scaled GEMMs, fused MoE layers, and other large fused kernels all directly shape end-to-end speed in both training and serving.

To make these kernels fast, however, we need more than a list of optimization tricks. Modern GPUs are no longer simple variations of the same old design. Recent architectures introduce richer memory spaces, new access patterns, and increasingly specialized execution units. To program them well, we need both a clear mental model of the hardware and a practical understanding of how high-performance kernels are built. This book is about developing both.

The book follows a simple progression: first understand the GPU hardware, then learn the programming model we will use, and finally build state-of-the-art kernels step by step. Our main target is the Blackwell generation, and our main running examples are General Matrix-Matrix Multiplication (GEMM) and FlashAttention. Along the way, we will also study the core ingredients behind GPU optimization: data layout, asynchronous data movement, and asynchronous coordination.

The material grows out of the [Machine Learning Systems](https://mlsyscourse.org/) course series at Carnegie Mellon University. To make the ideas easier to study and easier to run, this book uses the **TIRx** Python DSL to build real GPU kernel examples step by step. TIRx stays close to the hardware, which lets us reason about low-level control while still learning through runnable code.

This book is open source. Contributions, corrections, and examples are welcome through the [GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## How This Book Is Organized

- **Part I, Understanding the GPU.** This part introduces the overall organization of the GPU, general recipes for writing fast kernels, and key concepts such as data layout, asynchronous memory operations, and coordination. It builds the hardware intuition that the rest of the book relies on.
- **Part II, TIRx Overview.** This part introduces the key elements of TIRx, which serve as the foundation for the code examples throughout the book.
- **Part III, GEMM: Tiled to SOTA.** A complete guide to optimizing a tiled GEMM, built up through TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV, Flash Attention 4.** A complete attention kernel built from the Part III techniques: two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference and compiler internals.

### Part I, Understanding the GPU

- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/01 GPU Execution Model|GPU Execution Model]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/02 What Makes a Kernel Fast|What Makes a Kernel Fast]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/03 Data Layout and Its Notation|Data Layout and Its Notation]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/04 The Evolution of Tensor Core Data Layouts|The Evolution of Tensor Core Data Layouts]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/05 Async Data Movement - TMA|Async Data Movement: TMA]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/06 Blackwell Tensor Core - tcgen05 MMA|Blackwell Tensor Core: tcgen05.mma]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/07 Tensor Memory - TMEM|Tensor Memory (TMEM)]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/08 Async Coordination - mbarrier|Async Coordination: mbarrier]]
- [[GPU/modern-gpu-programming-for-mlsys/part-1-gpu/09 Advanced Scheduling - Cluster Launch Control|Advanced Scheduling: Cluster Launch Control]]

### Part II, TIRx Overview

- [[GPU/modern-gpu-programming-for-mlsys/part-2-tirx/10 Introduction to TIRx|Introduction to TIRx]]
- [[GPU/modern-gpu-programming-for-mlsys/part-2-tirx/11 TIRx Layout API|TIRx Layout API]]

### Part III, GEMM: Tiled to SOTA

- [[GPU/modern-gpu-programming-for-mlsys/part-3-gemm/12 Building a Tiled GEMM|Building a Tiled GEMM]]
- [[GPU/modern-gpu-programming-for-mlsys/part-3-gemm/13 Pipelining GEMM with TMA|Pipelining GEMM with TMA]]
- [[GPU/modern-gpu-programming-for-mlsys/part-3-gemm/14 Scaling GEMM with Warp Specialization and Clusters|Scaling GEMM with Warp Specialization and Clusters]]

### Part IV, Flash Attention 4

- [[GPU/modern-gpu-programming-for-mlsys/part-4-flash-attention/15 Flash Attention 4|Flash Attention 4]]

### Reference

- [[GPU/modern-gpu-programming-for-mlsys/reference/Reference|Reference]]
- [[GPU/modern-gpu-programming-for-mlsys/reference/Debugging Warp-Specialized Kernels|Debugging Warp-Specialized Kernels]]
- [[GPU/modern-gpu-programming-for-mlsys/reference/Compiler Internals|Compiler Internals]]
- [[GPU/modern-gpu-programming-for-mlsys/reference/TIRx Language Reference|TIRx Language Reference]]
## Interactive demos

See [[GPU/modern-gpu-programming-for-mlsys/Interactive Demos|Interactive Demos]] for all 23 local HTML visualizations. Demos used by a chapter are also embedded at the relevant reading position.

## Edition metadata

See [[GPU/modern-gpu-programming-for-mlsys/SOURCE|SOURCE]] for provenance, source commit, scope, and refresh notes.
