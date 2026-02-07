# Thermodynamic Computing Research

### ⚗️ TSU vs MCMC Benchmark and P0 Optimization

Research contributions to Extropic's thermodynamic computing framework (THRML). This repository contains benchmark experiments comparing Thermodynamic Sampling Unit (TSU) simulation against traditional Markov Chain Monte Carlo (MCMC) methods, along with JIT compilation optimizations to the THRML simulator that achieved a measurable 10% speedup on core sampling operations.

---

## 📋 Quick Summary

> 🔬 **Thermodynamic Computing Research** 是針對 Extropic 熱力學計算框架 THRML 的研究貢獻專案。🌡️ 核心工作包含兩大方向：一是 TSU（熱力學取樣單元）與傳統 MCMC（馬可夫鏈蒙地卡羅）方法的嚴謹基準測試，涵蓋 5 組實驗、13 張視覺化圖表，深入比較收斂速度、分佈品質與平行效率；二是透過 JIT 編譯優化（`@eqx.filter_jit`）加速 THRML 模擬器的 `_run_blocks` 與 `sample_blocks` 核心函式，在 9×9 數獨問題上達成 ⚡ 10% 加速（2.77s → 2.52s）。🧩 附帶實用的問題求解器：數獨求解器與八皇后問題求解器，展示如何將組合最佳化問題映射為熱力學能量景觀。📊 所有實驗皆以 JAX + Matplotlib 在 macOS 上完成，適合對新型計算典範感興趣的研究者與開發者。

---

## 🤔 Why This Exists

Thermodynamic computing represents a paradigm shift in how we approach optimization and sampling problems. Instead of step-by-step serial computation, a Thermodynamic Sampling Unit uses the physics of thermal noise and energy landscapes to find solutions -- millions of probabilistic bits "settling" into optimal configurations simultaneously, the way water finds the lowest point in a landscape.

This research project explores that paradigm through hands-on experimentation:
- **Benchmarking**: Rigorous comparison of TSU simulation versus MCMC across multiple problem types, with 13 visualization charts documenting convergence, distribution quality, and performance characteristics.
- **Optimization**: Identified and implemented JIT compilation improvements to THRML's core `_run_blocks` and `sample_blocks` functions, achieving a 10% simulator speedup (2.77s to 2.52s average on 9x9 Sudoku).
- **Learning Examples**: Practical experiments with Ising models, Gaussian chains, graph coloring, and constraint satisfaction problems.
- **Problem Solvers**: Working implementations of Sudoku and 8-Queens solvers using thermodynamic sampling.

---

## 🏗️ Architecture

```
thermodynamic-computing-research/
  examples-mac/
    P0_OPTIMIZATION_ANALYSIS.md         -- P0 JIT optimization results (10% speedup)
    P0_PLUS_OPTIMIZATION_ANALYSIS.md    -- Extended optimization analysis
    OPTIMIZATION_STRATEGY_UPDATE.md     -- Simulator vs algorithm optimization taxonomy
    THRML_OPTIMIZATION_TARGETS.md       -- Identified optimization targets
    benchmarks/
      benchmark_framework.py            -- Benchmark harness
      test_p0_optimization.py           -- P0 optimization test suite
      run_p0_benchmark.sh              -- Benchmark runner (P0)
      run_p0_plus_benchmark.sh         -- Benchmark runner (P0+)
    docs/                               -- Research documentation
    scripts/                            -- Utility scripts
  TSU_Benchmark_Results/
    results/
      00_demo_with_overlay.png          -- Performance demo overview
      01_node_distributions.png         -- Node distribution comparison
      02_convergence_comparison.png     -- TSU vs MCMC convergence curves
      02_magnetization_energy.png       -- Magnetization and energy analysis
      03_ks_test_results.png            -- Kolmogorov-Smirnov statistical tests
      03_sample_quality.png             -- Sample quality metrics
      04_coloring_[1-4].png             -- Graph coloring experiments
      04_parallel_efficiency.png        -- Parallel sampling efficiency
      05_code_complexity.png            -- Code complexity comparison
      integration_performance.png       -- Integration benchmark results
  THRML_Learning_Examples/
    results/
      coupling_comparison.png           -- Coupling strength analysis
      gaussian_chain_analysis.png       -- Gaussian chain model
      gaussian_grid_analysis.png        -- Gaussian grid model
      ising_chain_analysis.png          -- Ising chain model
      mixed_model_analysis.png          -- Mixed model analysis
  Thermoputer/
    README.md                           -- Detailed thermodynamic computing explainer
    sudoku.py                           -- Sudoku solver via thermodynamic sampling
    8queens.py                          -- 8-Queens solver via thermodynamic sampling
  ThermoKaiputer/                       -- Extended experiments
  thrml/                                -- Original THRML framework (reference)
```

---

## 📊 Key Results

### ⚡ P0 JIT Optimization

| Metric | Original | Optimized |
|--------|----------|-----------|
| Average time (9x9 Sudoku) | 2.774s | 2.522s |
| Speedup | -- | 1.09x (10%) |
| Optimization type | -- | JIT compilation via `@eqx.filter_jit` |

**Important distinction**: This optimization accelerates the THRML *simulator* (software running on CPU), not the TSU hardware itself. It reduces development iteration time. Algorithm-level optimizations (sparsification, early convergence) are the path to improving actual TSU hardware performance.

### 📈 Benchmark Experiments (5 Experiments, 13 Charts)

1. **Node Distribution Comparison** -- TSU vs MCMC sampling distribution fidelity
2. **Convergence Analysis** -- How quickly each method reaches equilibrium
3. **Sample Quality (KS Tests)** -- Statistical validation of sample distributions
4. **Graph Coloring** -- Constraint satisfaction problem benchmark
5. **Parallel Efficiency** -- Scaling characteristics of block-parallel sampling

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | THRML (Extropic) |
| Compute | JAX (CPU backend) |
| Language | Python |
| Visualization | Matplotlib |
| Platform | macOS (Apple Silicon / Intel) |

---

## 🧩 Problem Solvers

The `Thermoputer/` directory contains practical demonstrations of thermodynamic computing applied to classical combinatorial problems:

- **Sudoku Solver** -- Encodes Sudoku constraints as an energy landscape; the system "settles" into zero-conflict (solved) states through simulated annealing.
- **8-Queens Solver** -- Places 8 queens on a chessboard with zero mutual attacks by defining queen conflicts as energy, then sampling from the 92 known solutions.

Both solvers include detailed explanations of how each problem maps to a thermodynamic energy landscape.

---

## 👤 Author

**Huang Akai (Kai)**
Founder @ Universal FAW Labs | Creative Technologist | Ex-Ogilvy | 15+ years experience
