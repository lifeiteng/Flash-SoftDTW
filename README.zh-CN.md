# Flash-SoftDTW

**面向 PyTorch 的高速、低显存、精确 soft-DTW 实现——支持 CUDA 与 CPU、稠密与带状（banded）对齐、前向与反向传播。**

[English](README.md)

Flash-SoftDTW 用于计算成批代价矩阵（cost matrices）的软动态时间规整（soft-DTW）值及其精确梯度（即期望对齐矩阵）。它专为训练场景设计：原生支持变长 batch、Sakoe-Chiba 带（band）、规整惩罚（warp penalty）、fp16 / bf16 输入以及任意步幅（stride），且 kernel 全程最多只实例化一个 `[B, T, S]` 缓冲区。

- **速度快。** 采用倾斜条带波前（skewed-strip wavefront）kernel，将动态规划（DP）依赖链保留在寄存器与 warp shuffle 中。在 RTX PRO 6000 (Blackwell) 上处理稠密问题时，其**前向传播比逐行顺序执行的 CUDA 基线快 54–365×，前向+反向快 18–98×**（见下方表格）。
- **无长度限制。** 支持任意序列长度（已测试至 4096 × 4096）和任意带宽；没有“每列一个线程、最多 1024 列”的上限。
- **低显存开销。** 推理模式（`torch.no_grad()` 或输入未开启 `requires_grad`）仅需 `O(B · (T + S))` 临时空间，不保存得分矩阵。训练模式仅保存单个 `[B, T, S]` fp32 缓冲区；反向传播在线重算转移概率，而非提前存储。
- **带状对齐紧凑存储。** 传入 `[B, T, 2]` 的 band 和 `[B, T, W]` 的代价张量后，内存开销仅随带宽 `W` 而非 `S` 增长，且只计算带内单元。
- **精确且测试完备。** 包含约 600 个测试用例（CPU 加两块 GPU），与独立的 float64 参考实现对齐：涵盖数值、梯度、`torch.autograd.gradcheck`、CPU/CUDA 一致性、对比 `pyts` 的 band 构建、不变性（置换、padding、`gamma → 0` 收敛至硬 DTW）、half / bfloat16 以及非连续输入。
- **数值稳健。** 采用以 2 为底的 log-sum-exp 并以有限哨兵值代替 `inf`（避免 `inf - inf` 产生 NaN）；反向传播直接对前驱权重做归一化，而非使用在小 `gamma` 下会损失精度的 `R - cost`。

## 安装

环境要求：PyTorch（已在 2.13 + CUDA 12.9 与 2.15 CPU 上测试）、`ninja`、`setuptools >= 77`、PyTorch 构建所支持的 C++ 编译器；编译 GPU kernel 还需与 PyTorch 构建版本匹配的 CUDA Toolkit。

PyPI 包名为 `flash-softdtw-torch`（import 名为 `flash_softdtw`），与 PyPI 上的 `flash-softdtw` 包无关；两者 import 名相同，请勿安装在同一环境中。从源码编译安装：

```bash
git clone https://github.com/lifeiteng/Flash-SoftDTW.git
cd Flash-SoftDTW
pip install . --no-build-isolation
# pick target GPUs explicitly, e.g.:
TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0" pip install . --no-build-isolation
```

若未安装 CUDA Toolkit，则会编译 CPU 后端。也可以不安装、直接在源码检出目录中运行：扩展会在首次调用时自动 JIT 编译。

## 快速上手

```python
import torch
import flash_softdtw as fsd

x = torch.randn(32, 400, 80, device="cuda", requires_grad=True)   # [B, T, C]
y = torch.randn(32, 350, 80, device="cuda")                       # [B, S, C]
x_lens = torch.randint(200, 401, (32,), device="cuda")
y_lens = torch.randint(200, 351, (32,), device="cuda")

# 1) A loss module: distance + soft-DTW, optionally banded.
loss_fn = fsd.SoftDTWLoss(gamma=0.1, metric="sqeuclidean")
loss = loss_fn(x, y, x_lens, y_lens)
loss.backward()

# Soft-DTW divergence (Blondel et al., 2021): non-negative, 0 iff x == y.
div = fsd.SoftDTWLoss(gamma=0.1, normalize=True)(x, y, x_lens, y_lens)

# Sakoe-Chiba band: only cells within `window` steps of the diagonal.
banded = fsd.SoftDTWLoss(gamma=0.1, window=20)(x, y, x_lens, y_lens)

# 2) The raw operator on any cost matrix.
cost = fsd.pairwise_cost(x, y, metric="l1")        # [B, T, S]
value = fsd.soft_dtw(cost, x_lens, y_lens, gamma=0.1, warp_penalty=0.5)  # [B]
value.sum().backward()

# 3) Banded operator with compact storage.
band = fsd.sakoe_chiba_band(x_lens, y_lens, window=20)   # [B, T, 2] = [begin, end)
cost_b = fsd.pairwise_cost(x[:, : band.size(1)], y, "l1", band=band)  # [B, T, W]
value_b = fsd.soft_dtw(cost_b, x_lens, y_lens, band=band, gamma=0.1)
```

## 定义

对于每个 batch 元素及其中的单元 `(i, j)`（满足 `0 ≤ i < x_len`、`0 ≤ j < y_len`）：

```
R(-1, -1) = 0,   R(-1, j) = R(i, -1) = +inf,   R(cell outside the band) = +inf
R(i, j)   = C(i, j) + softmin_γ( R(i-1, j-1),  R(i-1, j) + w,  R(i, j-1) + w )
value     = R(x_len - 1, y_len - 1)
softmin_γ(a, b, c) = -γ · log(e^{-a/γ} + e^{-b/γ} + e^{-c/γ})
```

其中 `w` 为 `warp_penalty`（水平 / 垂直步的额外代价）。当 `γ → 0` 时退化为经典 DTW。`value` 对 `C` 的梯度即为期望对齐矩阵，返回布局与 `cost` 相同（稠密或紧凑带状），带外位置为零。如果终点单元落在带外或所存储张量范围之外，则值为 `+inf`，梯度为零。

## API

| 函数 | 说明 |
|---|---|
| `soft_dtw(cost, x_lens=None, y_lens=None, *, band=None, gamma=1.0, warp_penalty=0.0)` | 对 `[B, T, S]`（稠密）或 `[B, T, W]`（紧凑带状）代价张量计算可微的 soft-DTW 值 `[B]`。支持任意浮点 dtype；half / bf16 在内部以 fp32 累加。 |
| `SoftDTWLoss(gamma, metric, window, warp_penalty, normalize, reduction)` | `nn.Module`，接收特征序列 `x [B, T, C]`、`y [B, S, C]` 以及可选的序列长度。 |
| `pairwise_cost(x, y, metric, band=None)` | 支持 `"sqeuclidean"`、`"euclidean"`、`"l1"`、`"cosine"` 或自定义可调用对象；支持稠密或紧凑带状布局。`"sqeuclidean"` / `"cosine"` 基于矩阵乘法；`"l1"` 使用专门的广播实现（不走 `torch.cdist`），耗时随特征维度增长，特征维度较大时建议用 `"sqeuclidean"`。 |
| `sakoe_chiba_band(x_lens, y_lens, window)` | 生成沿（非方阵）对角线的 Sakoe-Chiba 带中每一行的 `[begin, end)` 列范围。 |
| `band_width`, `band_columns`, `gather_band` | 用于紧凑带状存储的辅助函数。 |

band 的列范围沿 `T` 维必须非递减（Sakoe-Chiba 带以及任何单调规整窗口均满足此条件）。支持一阶梯度。代价为 `+inf` 的单元视为障碍。内部数值以 `1 / gamma` 缩放后的 base-2 单位表示；在该单位下数值需保持在 `1e29` 以下（float32；float64 下为 `1e299`），仅在极端的 `gamma` 与代价值组合下才需注意。

## 基准测试

测试环境：RTX PRO 6000 Blackwell Max-Q (sm_120)，PyTorch 2.13 + CUDA 12.9，fp32，`gamma = 0.05`，`warp_penalty = 10`。使用 CUDA Event 计时，预热后取 10 次运行的中位数，期间 GPU 无其他负载。不含代价矩阵构建时间。Baseline 为早期的逐行顺序 CUDA soft-DTW 实现（不包含在本仓库中），在单次扫描中同时计算函数值与梯度，因此其前向耗时与前向+反向耗时相同。具体数值取决于测试机型。

**稠密场景，`T = S`**（单位：毫秒）：

| B | T = S | Flash fwd | Baseline fwd | Speedup | Flash fwd+bwd | Baseline fwd+bwd | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.093 | 5.28 | **56×** | 0.240 | 5.29 | **22×** |
| 1 | 512 | 0.371 | 85.7 | **231×** | 1.24 | 85.7 | **69×** |
| 1 | 1024 | 0.938 | 342 | **365×** | 3.50 | 342 | **98×** |
| 8 | 1024 | 1.01 | 349 | **345×** | 4.44 | 352 | **79×** |
| 64 | 512 | 0.383 | 96.3 | **252×** | 1.70 | 96.4 | **57×** |
| 64 | 1024 | 1.33 | 399 | **300×** | 5.14 | 400 | **78×** |
| 256 | 128 | 0.099 | 5.44 | **55×** | 0.301 | 5.47 | **18×** |
| 256 | 512 | 0.702 | 100 | **142×** | 3.55 | 100 | **28×** |
| 256 | 1024 | 2.41 | 402 | **167×** | 13.8 | 403 | **29×** |

**大 batch、窄 Sakoe-Chiba 带**（`B = 512`，`T ≤ 64`，`S ≤ 160`，window 为 4，变长序列，`gamma = 0.1`，`warp_penalty = 1`）：前向 0.079 ms vs 0.359 ms（**4.6×**），前向+反向 0.168 ms vs 0.369 ms（**2.2×**）。在此规模（约 0.2 M 个单元）下，耗时主要由 kernel 启动延迟主导；这类短 kernel 的耗时随 GPU 时钟状态波动（GPU 已升频时最多可快约 2×）。

**长序列**（无长度上限）：

| B | T = S | fwd | fwd+bwd | extra memory (fwd+bwd) |
|---:|---:|---:|---:|---:|
| 1 | 2048 | 2.9 ms | 12.0 ms | 32 MiB |
| 8 | 2048 | 3.8 ms | 17.4 ms | 256 MiB |
| 1 | 4096 | 10.1 ms | 50.8 ms | 128 MiB |
| 8 | 4096 | 14.0 ms | 66.3 ms | 1.0 GiB |
| 64 | 4096 | 15.2 ms | 72.2 ms | 8.0 GiB |

`python benchmarks/bench.py` 与 `python benchmarks/long_sequences.py` 可在你的 GPU 上复现上表中 Flash-SoftDTW 一侧的结果。

## 工作原理

每个问题的行被切分为高度为 `P` 的条带（strip），每行分配一个线程。在条带内部，线程 `k` 在时间步（tick）`t` 处理列 `j = jbase + t − k`，因此每个单元的三个 DP 前驱都由同一线程或线程 `k − 1` 在 1 或 2 个 tick 之前算得：

- 相邻线程的值通过 `__shfl_up_sync` 传递（每个 tick 仅一次 shuffle）；仅在 warp 边界处使用双缓冲共享内存（shared memory），仅在条带边界处使用全局内存中按问题分配的小型列缓冲区；
- 当 `P = 32` 时，每个 tick 完全无需同步屏障（barrier），且单个线程块（block）可打包四个独立问题——这是窄带与大 batch 场景下的快速路径；
- softmin 在无分支路径上以 2 为底（`ex2.approx` / `lg2.approx`）计算，并对不可达单元使用有限哨兵值；
- 在消费当前操作数的同时，预取接下来 8 个 tick 的操作数；
- 反向传播按逆序 tick 回放波前，根据保存的 `R` 重新计算归一化转移权重，并将期望对齐消息（左 / 上 / 对角）推送给各前驱单元。

## 测试

```bash
pip install pytest numpy pyts
python -m pytest -q
```

当检测到可用 GPU 时，会自动运行 CUDA 测试（覆盖所有可见 GPU，最多使用两张）。

## 引用

如果您在研究中使用了 Flash-SoftDTW，请引用本仓库及 soft-DTW 原论文：

```bibtex
@software{flash_softdtw,
  author = {Li, Feiteng},
  title  = {Flash-SoftDTW: Fast, memory-lean soft-DTW for PyTorch},
  url    = {https://github.com/lifeiteng/Flash-SoftDTW},
  year   = {2026}
}
@inproceedings{cuturi2017softdtw,
  title     = {Soft-DTW: a Differentiable Loss Function for Time-Series},
  author    = {Cuturi, Marco and Blondel, Mathieu},
  booktitle = {ICML},
  year      = {2017}
}
@inproceedings{blondel2021divergence,
  title     = {Differentiable Divergences Between Time Series},
  author    = {Blondel, Mathieu and Mensch, Arthur and Vert, Jean-Philippe},
  booktitle = {AISTATS},
  year      = {2021}
}
```

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE)。
