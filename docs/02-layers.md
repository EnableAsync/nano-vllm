# 阶段 2：基础组件层

## 目标

理解 Transformer 推理的基础构建块，重点掌握**张量并行如何拆分矩阵乘法**。

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `layers/activation.py` | 23 | SwiGLU 激活函数 |
| `layers/layernorm.py` | 68 | RMSNorm + 融合残差 |
| `layers/sampler.py` | 27 | Gumbel-max 采样 |
| `layers/rotary_embedding.py` | 85 | RoPE 旋转位置编码 |
| `layers/linear.py` | 214 | 张量并行线性层（核心） |
| `layers/embed_head.py` | 103 | 词表并行 Embedding + LM Head |

## 建议阅读顺序

```
activation → layernorm → sampler → rotary_embedding → linear → embed_head
简单独立      ↑残差连接     ↑采样技巧    ↑位置编码          ↑张量并行（重点）
```

---

## 第一部分：简单组件

### SwiGLU 激活 (`activation.py`)

**一句话**：输入一分为二，一半做门控一半做值，`SiLU(gate) * value`。

```
输入 x: [batch, seq, 2*hidden]
        ┌─────────┬─────────┐
        │  gate   │  value  │   chunk(2, -1)
        └────┬────┴────┬────┘
             │         │
         SiLU(gate)    │
             │         │
             └── × ────┘
                 │
        输出: [batch, seq, hidden]
```

**为什么用 SwiGLU？** 相比 ReLU/GELU，SwiGLU 在同等参数量下 loss 更低（PaLM 论文）。代价是 MLP 需要 3 个权重矩阵（gate、up、down）而非 2 个。

**关键代码**（2 行）：
```python
x, y = x.chunk(2, -1)       # 拆成 gate 和 value
return F.silu(x) * y         # SiLU 门控
```

### RMSNorm (`layernorm.py`)

**一句话**：`x / sqrt(mean(x²) + eps) * weight`，比 LayerNorm 省掉均值计算。

```
LayerNorm:  (x - mean) / sqrt(var + eps) * γ + β    ← 需要 mean 和 var
RMSNorm:     x / sqrt(mean(x²) + eps) * γ           ← 只需 mean(x²)
```

**两个前向路径**：

| 方法 | 用途 | 返回 |
|------|------|------|
| `rms_forward` | 第一层（无残差） | `norm_x` |
| `add_rms_forward` | 后续层（Pre-Norm） | `(norm_x, new_residual)` |

**融合残差**是重要优化：`residual + x → RMSNorm` 合成一个 kernel，减少一次显存读写。在 Transformer 中每层都要做，累积效果显著。

**Pre-Norm 数据流**：
```
               residual（跨层传递）
                  │
      ┌───────────┼───────────┐
      │     add_rms_forward   │
      │  x + residual → norm  │
      │           │           │
      │     new_residual   norm_x → 送入 Attention/MLP
      │           │
      └───────────┼───────────┘
                  │
            传给下一层
```

### Gumbel-max 采样 (`sampler.py`)

**一句话**：`argmax(softmax(logits/T) / Exp(1))` 等价于按概率采样。

**为什么不用 `torch.multinomial`？**
- multinomial 内部要构建 CDF（累积分布），有排序或累加
- Gumbel-max 只需 argmax，全程可并行，GPU 友好
- 配合 `torch.compile` 可以融合成单个 kernel

**数学等价性**：
```
传统采样:  softmax → CDF → 二分查找
Gumbel:   softmax(x) / Exp(1) → argmax    ← 证明等价于按概率采样
```

其中 `Exp(1)` 随机数 = `-log(Uniform)`，`clamp_min_(1e-10)` 防止除零。

---

## 第二部分：位置编码

### RoPE 旋转位置编码 (`rotary_embedding.py`)

**一句话**：把 Q/K 向量视为复数，乘以位置相关的旋转因子 `e^{iθ}`。

**核心思想**：
```
head_dim = 128 的向量 → 视为 64 个二维子空间
每个子空间施加旋转矩阵：

    ┌ cos θ  -sin θ ┐   ┌ x1 ┐   ┌ y1 ┐
    │                │ × │    │ = │    │
    └ sin θ   cos θ ┘   └ x2 ┘   └ y2 ┘

θ_i = pos / base^(2i/d)
```

**频率设计**的直觉：
- **低维（i 小）**：θ 变化快 → 捕捉相邻 token 的位置差异
- **高维（i 大）**：θ 变化慢 → 捕捉远距离的位置关系
- 类似傅里叶变换的多尺度分解

**为什么 Q 和 K 都要旋转？**

旋转后内积 `Q_m · K_n = f(m-n)`，只依赖相对位置差 `m-n`，不需要显式编码绝对位置。

**预计算缓存**：
```python
inv_freq = 1 / base^(2i/d)              # [d/2] 频率
freqs = positions × inv_freq             # [max_pos, d/2] 角度
cache = [cos(freqs), sin(freqs)]         # 预计算，推理时查表
```

---

## 第三部分：张量并行（重点）

### 核心概念

张量并行将一个大矩阵乘法 `Y = X @ W` 拆到多个 GPU 并行计算。

**两种切法**：

```
Column Parallel（按输出维度切）        Row Parallel（按输入维度切）

W = [W1 | W2]  按列切                 W = [W1]  按行切
                                          [W2]

GPU 0: Y1 = X @ W1                    GPU 0: Y0 = X0 @ W1
GPU 1: Y2 = X @ W2                    GPU 1: Y1 = X1 @ W2
                                              Y = Y0 + Y1 (all-reduce)
结果：Y = [Y1 | Y2] 拼接              结果：Y = sum (求和)
无需通信 ✓                             需要 all-reduce ✗
```

### Megatron-LM 配对方案

**关键洞察**：Column 和 Row 配对使用，整个子层只需 **1 次 all-reduce**。

```
MLP 层:
  输入 x ─→ [ColumnParallel: gate_up_proj] ─→ SwiGLU ─→ [RowParallel: down_proj] ─→ 输出
             每个 GPU 独立算部分输出             │          部分输入 → all-reduce 求和
             无需通信                            │          1 次 all-reduce
                                                 ↑
                                           Column 的输出
                                           直接作为 Row 的输入
                                           中间无需通信！

Attention 层:
  输入 x ─→ [QKVParallel] ─→ Attention ─→ [RowParallel: o_proj] ─→ 输出
             Column 切法       每个 GPU         all-reduce 求和
             Q/K/V 合并        算部分 heads      1 次 all-reduce
```

### 线性层类继承关系 (`linear.py`)

```
LinearBase                    基类：权重创建 + TP 信息
├── ReplicatedLinear          不切分，每个 GPU 完整副本
├── ColumnParallelLinear      按 output 维度切
│   ├── MergedColumnParallel  gate + up 合并为一次 GEMM
│   └── QKVParallelLinear     Q + K + V 合并（支持 GQA）
└── RowParallelLinear         按 input 维度切，forward 中 all-reduce
```

### weight_loader 机制

每个并行层都定义 `weight_loader`，告诉加载器如何从完整权重中提取本 GPU 的 shard：

```python
# ColumnParallel: tp_dim=0，沿 output 维度取
loaded_weight.narrow(0, rank * shard_size, shard_size)

# RowParallel: tp_dim=1，沿 input 维度取
loaded_weight.narrow(1, rank * shard_size, shard_size)

# MergedColumn: 先定位子矩阵(gate/up)，再取 shard
param.narrow(0, shard_offset, shard_size)
loaded_weight.chunk(tp_size, 0)[rank]

# QKV: 同理，但 Q/K/V 大小可能不同（GQA）
```

### RowParallelLinear 的 bias 技巧

```python
# 只有 rank 0 加 bias，避免 all-reduce 后 bias 被重复加 tp_size 次
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
```

### 词表并行 (`embed_head.py`)

**Embedding**：词表按行切分，每个 GPU 存 `vocab_size / tp` 行。

```
词表 = [0 ... 37983 | 37984 ... 75967 | 75968 ... 113951 | 113952 ... 151935]
        GPU 0          GPU 1             GPU 2              GPU 3

前向：
1. mask = (token_id ∈ 本 GPU 范围?)
2. 本地查表（范围外的 id 查到的结果会被 mask 置零）
3. all-reduce 求和 → 完整 embedding
```

**LM Head**：复用 Embedding 权重，但方向相反。

```
Embedding:  token_id → embedding 向量  (查表)
LM Head:    hidden → logits           (矩阵乘)
```

LM Head 用 `gather`（非 all-gather）收集到 rank 0，因为只有 rank 0 需要采样：

```
GPU 0: logits_0 (vocab 0~37983)     ──┐
GPU 1: logits_1 (vocab 37984~75967) ──┤ gather → rank 0 拼接
GPU 2: logits_2 (vocab 75968~113951)──┤          完整 logits
GPU 3: logits_3 (vocab 113952~151935)─┘          → 采样
```

Prefill 优化：只取每个序列最后一个 token 的 hidden state 做投影，减少计算量。

---

## 通信总结

| 组件 | 通信操作 | 次数/层 |
|------|----------|---------|
| Embedding | all-reduce | 1（仅首层） |
| QKV proj | 无 | 0 |
| Attention | 无（每 GPU 算自己的 heads） | 0 |
| o_proj (Row) | **all-reduce** | **1** |
| gate_up_proj | 无 | 0 |
| SwiGLU | 无 | 0 |
| down_proj (Row) | **all-reduce** | **1** |
| LM Head | gather | 1（仅末层） |

**每个 Transformer 层只需 2 次 all-reduce**（Attention 1 次 + MLP 1 次），这就是 Megatron-LM 张量并行的核心效率。

## 核心问题回答

**Q: 张量并行如何实现？**

1. **切分策略**：Column 按输出切，Row 按输入切，配对使用最小化通信
2. **权重加载**：`weight_loader` 机制让每个 GPU 从完整 checkpoint 中只取自己的 shard
3. **通信点**：只在 RowParallel 的 forward 中做 all-reduce（每层 2 次）
4. **词表特殊处理**：Embedding 用 mask + all-reduce，LM Head 用 gather 到 rank 0
