---
title: Zhenwei's Blog
description: GPU 架构、CUDA 编程与性能优化技术笔记。
---

<nav class="home-nav" aria-label="首页导航">
  <a class="home-brand" href="/">
    <span class="home-brand-mark" aria-hidden="true">Z</span>
    <span>Zhenwei's Blog</span>
  </a>
  <div class="home-nav-links">
    <a href="/">首页</a>
    <a href="/gpu/gpu-知识库索引">知识库</a>
    <a href="/gpu/gpu-kernel-学习路线">学习路线</a>
    <a href="/tags/">标签</a>
  </div>
</nav>

<section class="home-cover" aria-labelledby="home-title">
  <div class="home-cover-grid" aria-hidden="true"></div>
  <div class="home-orbit home-orbit-one" aria-hidden="true"></div>
  <div class="home-orbit home-orbit-two" aria-hidden="true"></div>
  <div class="home-cover-content">
    <p class="home-kicker">GPU · CUDA · PERFORMANCE</p>
    <h1 id="home-title">Zhenwei Cao</h1>
    <p class="home-motto">生如逆旅，一苇以航</p>
    <p class="home-intro">记录 GPU 架构、CUDA 编程与性能优化的学习路径，把硬件机制、Kernel 实现和性能分析连接成一张可以持续生长的知识地图。</p>
    <div class="home-actions">
      <a class="home-action home-action-primary" href="#featured">开始阅读</a>
      <a class="home-action home-action-secondary" href="https://github.com/ZhenweiCao" target="_blank" rel="noreferrer">GitHub</a>
    </div>
  </div>
  <div class="home-signal-panel" aria-label="站点主题">
    <span class="home-signal-label">CURRENT FOCUS</span>
    <strong>GPU Systems</strong>
    <div class="home-signal-route" aria-hidden="true">
      <span>ARCH</span><i></i><span>KERNEL</span><i></i><span>PROFILE</span>
    </div>
    <p>Architecture → Runtime → Performance</p>
  </div>
  <a class="home-scroll" href="#featured" aria-label="向下浏览精选内容">
    <span></span>
  </a>
</section>

<section class="home-section home-featured" id="featured" aria-labelledby="featured-title">
  <div class="home-section-heading">
    <div>
      <p class="home-section-kicker">START HERE</p>
      <h2 id="featured-title">从一张知识地图开始</h2>
    </div>
    <p>按“硬件背景 → CUDA 基础 → Kernel 优化 → 性能分析”的顺序，逐步建立完整的 GPU 系统视角。</p>
  </div>
  <div class="home-topic-grid">
    <a class="home-topic-card home-topic-architecture" href="/gpu/hardware/gpu-硬件背景地图">
      <span class="home-topic-number">01</span>
      <span class="home-topic-icon" aria-hidden="true">ARCH</span>
      <strong>GPU 硬件与架构</strong>
      <small>理解 SM、warp、内存层级与现代 GPU 的执行路径。</small>
      <span class="home-topic-link">进入专题 <b aria-hidden="true">→</b></span>
    </a>
    <a class="home-topic-card home-topic-cuda" href="/gpu/cuda/cuda-编程基础">
      <span class="home-topic-number">02</span>
      <span class="home-topic-icon" aria-hidden="true">CUDA</span>
      <strong>CUDA 编程基础</strong>
      <small>从 grid、block、thread 到内存访问，写出第一个 kernel。</small>
      <span class="home-topic-link">进入专题 <b aria-hidden="true">→</b></span>
    </a>
    <a class="home-topic-card home-topic-kernel" href="/gpu/gpu-kernel-学习路线">
      <span class="home-topic-number">03</span>
      <span class="home-topic-icon" aria-hidden="true">KERNEL</span>
      <strong>Kernel 学习路线</strong>
      <small>沿着 GEMM、Tensor Core 与 Blackwell 机制逐层深入。</small>
      <span class="home-topic-link">进入专题 <b aria-hidden="true">→</b></span>
    </a>
    <a class="home-topic-card home-topic-profiling" href="/gpu/profiling/cuda-kernel-性能瓶颈定位流程">
      <span class="home-topic-number">04</span>
      <span class="home-topic-icon" aria-hidden="true">NCU</span>
      <strong>性能分析</strong>
      <small>用 profiler 建立证据链，定位瓶颈并验证优化收益。</small>
      <span class="home-topic-link">进入专题 <b aria-hidden="true">→</b></span>
    </a>
  </div>
</section>

<section class="home-quote" aria-label="博客寄语">
  <span aria-hidden="true">“</span>
  <blockquote>
    <p>Get busy living, or get busy dying.</p>
    <cite>把零散理解沉淀为可以回看的技术文章。</cite>
  </blockquote>
</section>

<section class="home-section home-reading" aria-labelledby="reading-title">
  <div class="home-section-heading">
    <div>
      <p class="home-section-kicker">FEATURED NOTES</p>
      <h2 id="reading-title">精选技术文章</h2>
    </div>
    <a class="home-all-notes" href="/gpu/gpu-知识库索引">浏览全部笔记 <span aria-hidden="true">→</span></a>
  </div>
  <div class="home-post-grid">
    <a class="home-post-card home-post-wide" href="/gpu/modern-gpu-programming-for-mlsys/part-4-flash-attention/15-flash-attention-4">
      <figure>
        <img src="/gpu/modern-gpu-programming-for-mlsys/assets/images/flash_attention_pipeline_v2.png" alt="Flash Attention 4 流水线示意图" loading="lazy">
      </figure>
      <div class="home-post-content">
        <span class="home-post-tag">FLASH ATTENTION</span>
        <h3>Flash Attention 4</h3>
        <p>跟随一个 tile 穿过 TMA、TMEM、softmax 与两段 MMA，理解 Blackwell 上的 attention kernel。</p>
        <span class="home-post-meta">专题教程 · 深度阅读</span>
      </div>
    </a>
    <a class="home-post-card" href="/gpu/cuda/cuda-gemm-矩阵乘法优化指南">
      <figure>
        <img src="/gpu/modern-gpu-programming-for-mlsys/assets/images/gemm_perf.png" alt="GEMM 性能对比图" loading="lazy">
      </figure>
      <div class="home-post-content">
        <span class="home-post-tag">GEMM</span>
        <h3>CUDA GEMM 优化指南</h3>
        <p>从 naive kernel 到 Tensor Core，梳理矩阵乘法的完整优化栈。</p>
        <span class="home-post-meta">CUDA · Kernel 优化</span>
      </div>
    </a>
    <a class="home-post-card" href="/gpu/profiling/cuda-kernel-性能瓶颈定位流程">
      <figure>
        <img src="/gpu/drawings/ncu-kernel-分析闭环.svg" alt="CUDA Kernel 性能分析闭环" loading="lazy">
      </figure>
      <div class="home-post-content">
        <span class="home-post-tag">PROFILING</span>
        <h3>CUDA Kernel 性能瓶颈定位</h3>
        <p>从症状到指标建立分析闭环，用 Nsight Compute 定位并验证 Kernel 瓶颈。</p>
        <span class="home-post-meta">性能分析 · 实践指南</span>
      </div>
    </a>
  </div>
</section>

<section class="home-note" aria-label="发布说明">
  <strong>公开笔记说明</strong>
  <p>本站内容由 Obsidian 中经过筛选的公开笔记构建。PDF 原件暂未发布，后续会迁移到对象存储并提供稳定的在线阅读链接。</p>
</section>
