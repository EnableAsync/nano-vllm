# 阶段 3：模型结构与注意力机制

## 目标

理解 Qwen3 的模型组装方式和注意力机制实现，重点掌握 **Prefill 与 Decode 阶段 attention 的区别**。

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `layers/attention.py` | ~90 | KV cache 存储 (Triton) + Prefill/Decode 注意力分发 |
| `models/qwen3.py` | ~220 | Qwen3 模型定义（Attention + MLP + DecoderLayer + Model + CausalLM） |

## 建议阅读顺序

```
qwen3.py（整体结构） → attention.py（注意力细节）
先看骨架再看核心
```

---

## 第一部分：模型架构

### 架构树形图

```
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding     # token id → hidden
│   ├── layers: ModuleList[Qwen3DecoderLayer] × N
│   │   ├── input_layernorm: RMSNorm
│   │   ├── self_attn: Qwen3Attention
│   │   │   ├── qkv_proj: QKVParallelLinear       # Q/K/V 合并投影
│   │   │   ├── q_norm / k_norm: RMSNorm          # QK-Norm（无 bias 时）
│   │   │   ├── rotary_emb: RotaryEmbedding        # RoPE
│   │   │   ├── attn: Attention                    # Prefill/Decode 分发
│   │   │   └── o_proj: RowParallelLinear          # 输出投影
│   │   ├── post_attention_layernorm: RMSNorm
│   │   └── mlp: Qwen3MLP
│   │       ├── gate_up_proj: MergedColumnParallelLinear
│   │       ├── act_fn: SiluAndMul                 # SwiGLU
│   │       └── down_proj: RowParallelLinear
│   └── norm: RMSNorm                             # 最终层归一化
└── lm_head: ParallelLMHead                       # hidden → logits
```

### packed_modules_mapping：权重名映射

HuggingFace checkpoint 中 Q/K/V 和 gate/up 是独立权重，但推理时合并为一次 GEMM 更高效：

```
HF 原始名          →  本项目名         →  映射含义
q_proj             →  qkv_proj["q"]    →  QKVParallelLinear 的 Q 部分
k_proj             →  qkv_proj["k"]    →  QKVParallelLinear 的 K 部分
v_proj             →  qkv_proj["v"]    →  QKVParallelLinear 的 V 部分
gate_proj          →  gate_up_proj[0]  →  MergedColumnParallel 的第 0 段
up_proj            →  gate_up_proj[1]  →  MergedColumnParallel 的第 1 段
```

`weight_loader` 根据此映射，从 checkpoint 中提取各子矩阵并写入合并参数的正确位置。

### DecoderLayer 前向流程（Pre-Norm 残差）

```
输入: hidden_states, residual
         │
    ┌────┴────────────────────────────────────┐
    │  首层 (residual=None):                   │
    │    residual = hidden_states              │
    │    hidden = RMSNorm(hidden_states)       │
    │                                          │
    │  后续层:                                  │
    │    hidden, residual = RMSNorm(           │
    │        hidden_states + residual)  ← 融合  │
    └────┬────────────────────────────────────┘
         │ (normalized)
    Self-Attention
         │
    post_attention_layernorm(hidden + residual) → 更新 residual
         │ (normalized)
    MLP
         │
输出: hidden_states, residual → 传给下一层
```

关键点：`residual` 作为独立变量跨层传递，`RMSNorm` 内部将 `hidden + residual` 融合成一个 kernel，避免额外显存读写。

### GQA：Grouped-Query Attention

Qwen3 使用 GQA，Q heads 数量多于 KV heads：

```
例：Qwen3-8B（TP=1）
  num_attention_heads = 32     → Q heads = 32
  num_key_value_heads = 8      → KV heads = 8
  每 4 个 Q head 共享 1 组 KV  → group_size = 32/8 = 4

TP=2 切分后：
  每 GPU: Q heads = 16, KV heads = 4
  GQA 比例不变，Flash Attention 内部自动处理 repeat
```

优势：KV cache 减少为 MHA 的 `1/group_size`（这里 1/4），显存占用大幅降低。

---

## 第二部分：注意力机制

### KV Cache 存储：Triton Kernel

`store_kvcache_kernel` 是一个简单的 Triton kernel，将当前 token 的 K/V 向量写入 paged KV cache：

```
输入:
  key:   [N, num_kv_heads, head_dim]    # N = 所有 token 总数
  value: [N, num_kv_heads, head_dim]
  slot_mapping: [N]                      # 每个 token → cache 物理 slot

处理:
  每个 Triton program 处理 1 个 token:
    1. 读取 slot = slot_mapping[idx]
    2. 跳过 slot == -1（padding）
    3. 将 key[idx] → k_cache[slot]
    4. 将 value[idx] → v_cache[slot]

cache 布局:
  k_cache / v_cache: [num_blocks * block_size, num_kv_heads * head_dim]
  每行 = 一个 slot = 一个 token 的 K 或 V
```

为什么用 Triton 而非 PyTorch？避免 Python 循环，N 个 token 完全并行写入。

### Prefill vs Decode 对比

| | Prefill | Decode |
|---|---------|--------|
| **场景** | 处理整个 prompt | 逐 token 生成 |
| **Q 形状** | `[N_total, heads, dim]`（多 token） | `[batch, heads, dim]`（每序列 1 token） |
| **Flash 函数** | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| **输入 K/V** | 当前计算的 K/V（或 prefix cache） | KV cache 中的历史 K/V |
| **Context 关键字段** | `cu_seqlens_q/k`, `max_seqlen_q/k` | `context_lens`, `block_tables` |
| **KV Cache 操作** | 写入新 K/V | 写入新 K/V + 读取历史 K/V |
| **因果掩码** | 三角形（每个 token 只看前面） | 不需要掩码（只有 1 个 query） |

### Context 数据类字段说明

`Context` 是一个全局数据类，由 `ModelRunner` 在每次推理前设置，通过 `get_context()` 在模型内部读取：

| 字段 | 类型 | 含义 |
|------|------|------|
| `is_prefill` | `bool` | 当前是 Prefill 还是 Decode |
| `cu_seqlens_q` | `Tensor` | Q 的累积序列长度（Prefill 用），如 `[0, 5, 12, 20]` |
| `cu_seqlens_k` | `Tensor` | K 的累积序列长度（Prefill 用） |
| `max_seqlen_q` | `int` | 最长 Q 序列长度（Prefill 用） |
| `max_seqlen_k` | `int` | 最长 K 序列长度（Prefill 用） |
| `slot_mapping` | `Tensor` | 每个 token → KV cache 物理 slot 编号 |
| `context_lens` | `Tensor` | 每个序列的历史长度（Decode 用） |
| `block_tables` | `Tensor` | Paged attention 的 block 映射表 |

### 数值示例：3 条序列的 Prefill → Decode

**Prefill 阶段**：3 条 prompt 长度分别为 5, 7, 8

```
token 拼接：[t0 t1 t2 t3 t4 | t5 t6 t7 t8 t9 t10 t11 | t12 t13 t14 t15 t16 t17 t18 t19]
              seq 0 (len=5)     seq 1 (len=7)               seq 2 (len=8)

Context:
  is_prefill = True
  cu_seqlens_q = [0, 5, 12, 20]     # 累积长度
  cu_seqlens_k = [0, 5, 12, 20]     # Prefill 时 Q=K
  max_seqlen_q = 8
  max_seqlen_k = 8
  slot_mapping  = [s0, s1, ..., s19]  # 20 个 token 各自的 cache slot

Q/K/V shape: [20, num_heads, head_dim]  ← 所有序列拼接
flash_attn_varlen_func 根据 cu_seqlens 区分序列边界，各序列独立做因果 attention
```

**Decode 阶段**：每个序列生成 1 个新 token

```
token：[t_new_0, t_new_1, t_new_2]   # 3 条序列各 1 个新 token

Context:
  is_prefill = False
  context_lens = [5, 7, 8]           # 各序列历史长度
  block_tables = [[b0, b1], ...]     # paged cache block 映射
  slot_mapping  = [s20, s21, s22]    # 3 个新 token 的 slot

Q shape: [3, num_heads, head_dim]    ← batch=3, seqlen=1
flash_attn_with_kvcache 自动从 k_cache/v_cache 中读取历史 K/V
```

---

## 第三部分：完整注意力数据流

### 从 hidden_states 到 output

```
hidden_states: [N, hidden_size]
        │
   ┌────┴────┐
   │ qkv_proj │   QKVParallelLinear (一次 GEMM)
   └────┬────┘
        │ [N, q_size + 2 * kv_size]
        │
   split → Q [N, num_heads, head_dim]
           K [N, num_kv_heads, head_dim]
           V [N, num_kv_heads, head_dim]
        │
   ┌────┴────┐
   │ QK-Norm │   RMSNorm (可选，Qwen3 无 bias 时启用)
   └────┬────┘
        │
   ┌────┴────┐
   │  RoPE   │   旋转位置编码（只作用于 Q 和 K）
   └────┬────┘
        │
   ┌────┴─────────────────────────────────────┐
   │  store_kvcache                           │
   │  将 K/V 写入 paged cache                  │
   └────┬─────────────────────────────────────┘
        │
   ┌────┴─────────────────────────────────────┐
   │  Prefill?                                │
   │  ├─ Yes → flash_attn_varlen_func         │
   │  │        (变长拼接, cu_seqlens 分界)      │
   │  └─ No  → flash_attn_with_kvcache        │
   │           (batch 模式, 从 cache 读历史)    │
   └────┬─────────────────────────────────────┘
        │ o: [N, num_heads, head_dim]
        │
   ┌────┴────┐
   │ o_proj  │   RowParallelLinear (含 all-reduce)
   └────┬────┘
        │
   output: [N, hidden_size]
```

---

## 核心问题回答

**Q: Prefill 和 Decode 的 attention 有何不同？**

它们的数学公式相同（`softmax(QK^T / √d) V`），但在工程实现上有三个关键差异：

1. **输入形状不同**
   - Prefill：多条序列的所有 token 拼接成一个长序列 `[N_total, ...]`，用 `cu_seqlens` 标记边界
   - Decode：每条序列仅 1 个新 token，`[batch_size, ...]`

2. **K/V 来源不同**
   - Prefill：Q/K/V 全部来自当前计算（或 prefix cache），每个 token 只看自己之前的 token（因果掩码）
   - Decode：Q 来自当前新 token，K/V 从 paged KV cache 中读取全部历史

3. **Flash Attention 函数不同**
   - Prefill 用 `flash_attn_varlen_func`：支持变长序列拼接，通过 `cu_seqlens` 划分序列边界，一次 kernel 处理多条不等长序列
   - Decode 用 `flash_attn_with_kvcache`：专为 seqlen=1 优化，直接从 paged cache 读取 K/V，避免数据拷贝

**两者共享同一份 KV cache**：Prefill 写入 → Decode 追加写入并读取，这就是 `store_kvcache` 在两个阶段都会执行的原因。
