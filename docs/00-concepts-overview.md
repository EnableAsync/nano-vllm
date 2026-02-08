# Nano-vLLM 核心概念全景图

> 本文档梳理 nano-vllm 涉及的所有 AI Infra 核心概念，解释它们之间的关系，并用通俗比喻帮助理解。

---

## 一、概念地图

```
用户请求 "Hello"
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LLMEngine（饭店大堂经理）                                            │
│  ┌─────────────┐    ┌────────────────┐    ┌───────────────────────┐ │
│  │  Tokenizer   │───▶│   Scheduler    │───▶│    ModelRunner        │ │
│  │  文本↔token  │    │  (调度器)       │    │   (模型执行器)         │ │
│  └─────────────┘    │                │    │                       │ │
│                     │  Continuous    │    │  ┌─────────────────┐  │ │
│                     │  Batching      │    │  │  Qwen3Model     │  │ │
│                     │  (持续凑批)     │    │  │  ┌───────────┐  │  │ │
│                     │                │    │  │  │ Embedding  │  │  │ │
│                     │  ┌──────────┐  │    │  │  │ RMSNorm    │  │  │ │
│                     │  │ Block    │  │    │  │  │ Attention  │  │  │ │
│                     │  │ Manager  │  │    │  │  │  ├ RoPE    │  │  │ │
│                     │  │ (显存    │  │    │  │  │  ├ Flash   │  │  │ │
│                     │  │  管家)   │  │    │  │  │  └ KVCache │  │  │ │
│                     │  └──────────┘  │    │  │  │ MLP/SwiGLU │  │  │ │
│                     │                │    │  │  │ LM Head    │  │  │ │
│                     └────────────────┘    │  │  └───────────┘  │  │ │
│                                          │  └─────────────────┘  │ │
│                                          │  Sampler (采样器)      │ │
│                                          │  CUDA Graph / Compile  │ │
│                                          │  Tensor Parallelism    │ │
│                                          └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

概念分为三层：

| 层级 | 概念 | 关键词 |
|------|------|--------|
| **调度层** | Continuous Batching, Prefill/Decode, Scheduler, Preemption | 何时算、算哪些 |
| **显存层** | KV Cache, Paged Attention, Block Manager, Prefix Caching | 存在哪、怎么省 |
| **计算层** | Flash Attention, RoPE, RMSNorm, SwiGLU, Tensor Parallelism, CUDA Graph, torch.compile, Triton Kernel, Gumbel-max Sampling | 怎么算、算得快 |

---

## 二、概念详解与比喻

### 2.1 Prefill 与 Decode：两个截然不同的阶段

#### 是什么

LLM 推理分为两个阶段：

- **Prefill（预填充）**：把用户输入的整个 prompt 一次性送入模型，计算所有 token 的 KV 并缓存。输出第一个生成 token。这是**计算密集型**操作。
- **Decode（解码）**：每次只送入上一步生成的 1 个 token，利用缓存的 KV 做注意力计算，再生成下一个 token。重复直到结束。这是**访存密集型**操作。

#### 比喻：读书 vs 续写

**Prefill** 就像你拿到一本 200 页的书，需要从头到尾通读一遍，做好笔记（KV Cache）。这一步工作量大，但可以并行处理（一目十行）。

**Decode** 就像你读完书后开始续写故事。每写一个字，你都要翻看之前的笔记（KV Cache）找灵感。每次只写一个字，但每个字都要查阅大量笔记，瓶颈在于"翻笔记的速度"（显存带宽）而非"思考的速度"（计算能力）。

#### 在代码中

```
# scheduler.py: schedule() 决定当前做 prefill 还是 decode
if scheduled_seqs:
    return scheduled_seqs, True   # prefill
...
return scheduled_seqs, False      # decode

# model_runner.py: 两阶段准备不同的输入
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
```

**关键区别**：Prefill 送入整个 prompt（可能几百个 token），Decode 只送入 1 个 token。

---

### 2.2 KV Cache：避免重复计算的缓存

#### 是什么

Transformer 的注意力机制需要当前 token 和所有历史 token 做点积。如果每次 decode 都重新计算所有历史 token 的 K、V，复杂度是 O(n²)。

**KV Cache** 把每一层计算过的 Key 和 Value 存下来，decode 时直接读取，不重复计算。

#### 比喻：会议纪要

你在开一个长会，每个人发言时你需要参考之前所有人说过的话。

- **没有 KV Cache**：每个人发言时，你要把之前所有人的发言重新整理一遍。会开到第 100 人时，你要重新整理 99 份发言记录。
- **有 KV Cache**：你维护一份持续更新的会议纪要（KV Cache）。每个人发言后，你只需把这个人的要点追加到纪要里。下一个人发言时，直接查阅纪要即可。

#### 在代码中

```python
# model_runner.py: 一次性分配所有层的 KV Cache
# shape: [2(K+V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)

# attention.py: 每次前向时将新 K/V 写入缓存
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

#### 显存开销

KV Cache 是推理时最大的显存消费者。以 Qwen3-0.6B 为例：
- 每个 token 每层需要存 K 和 V 各 `num_kv_heads × head_dim` 个参数
- 总开销 = `2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size`

这就是为什么需要 Paged Attention 来高效管理这块显存。

---

### 2.3 Paged Attention / Block Manager：显存的"虚拟内存"

#### 是什么

传统方法为每个序列预分配一整块连续显存存放 KV Cache（按 max_seq_len 分配）。问题：

1. **碎片化**：序列实际长度远小于 max_seq_len，大量显存浪费
2. **无法共享**：相同前缀的序列无法复用 KV

**Paged Attention** 借鉴操作系统虚拟内存的思路：
- 将 KV Cache 切分为固定大小的 **Block**（类似内存页）
- 每个序列维护一个 **Block Table**（类似页表），记录自己用了哪些物理 Block
- Block 按需分配，用多少分多少

#### 比喻：酒店房间管理

**传统方式**（连续分配）：
> 来了一个旅行团说"我们最多 100 人"，你就得预留 100 间房。结果只来了 20 人，80 间房空着，别的客人也住不进来。

**Paged Attention**（分页分配）：
> 来了多少人就分多少房间，而且不需要连续房间。每个旅行团有一张"房间清单"（Block Table），记录团员住在哪些房间号。走了的客人房间立刻回收给新客人。

#### 在代码中

```python
# block_manager.py
class Block:
    block_id: int       # 房间号
    ref_count: int      # 几个旅行团共用这个房间（prefix caching 用）
    hash: int           # 房间内容的指纹（用于 prefix cache 匹配）

class BlockManager:
    blocks: list[Block]              # 所有房间
    free_block_ids: deque[int]       # 空房间列表
    hash_to_block_id: dict[int,int]  # 指纹→房间号（prefix cache 查找表）

# sequence.py
class Sequence:
    block_table: list[int]   # 这个序列的"房间清单"
```

---

### 2.4 Prefix Caching：相同前缀共享 KV

#### 是什么

很多请求有相同的 system prompt（如 "你是一个有帮助的助手..."）。这些相同前缀的 KV 计算结果完全一样，没必要重复算。

**Prefix Caching** 通过内容哈希识别相同的 Block，让多个序列共享同一块物理 KV Cache。

#### 比喻：共享教材

> 一个班 40 个学生上同一门课。教材的前 10 章（system prompt）内容完全一样。
>
> - **无 Prefix Cache**：每个学生复印一份完整教材，40 份。
> - **有 Prefix Cache**：前 10 章只印 1 份放在教室（共享 Block），40 个学生各自只印自己的笔记（不同的后续 token）。引用计数 = 40。

#### 在代码中

```python
# block_manager.py: allocate() 中尝试 prefix cache
h = self.compute_hash(token_ids, h)    # 计算哈希链
block_id = self.hash_to_block_id.get(h, -1)
if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
    # Cache 命中！复用已有 Block，跳过这些 token 的计算
    seq.num_cached_tokens += self.block_size
    block.ref_count += 1   # 引用计数 +1
```

哈希链设计精巧：`block1_hash = hash(block0_hash + block1_tokens)`。只有前缀完全相同的序列才会产生相同的哈希链。

---

### 2.5 Continuous Batching：持续凑批调度

#### 是什么

传统 batching（static batching）必须等一批请求全部完成才能处理下一批。问题：有的请求生成 10 个 token 就结束了，有的要生成 1000 个，短请求被迫等待长请求。

**Continuous Batching** 允许：
- 完成的请求随时离开
- 新请求随时加入
- 每一步都把当前所有活跃请求组成一个 batch 一起算

#### 比喻：食堂 vs 自助餐

**Static Batching**（食堂）：
> 10 个人坐一桌点菜，必须等所有人吃完才能收桌上新客人。你只吃了碗面 3 分钟就完事了，但隔壁在吃满汉全席，你得干等 2 小时。

**Continuous Batching**（自助餐传送带）：
> 传送带（GPU）持续运转。你吃完了把盘子撤走，新来的客人立刻坐到你位置。传送带始终满载运行，GPU 利用率最大化。

#### 在代码中

```python
# scheduler.py: 两个队列实现 Continuous Batching
waiting: deque[Sequence]  # 等待 prefill 的新请求
running: deque[Sequence]  # 正在 decode 的活跃请求

def schedule(self):
    # 优先处理 waiting 中的新请求（prefill）
    # 无新请求时，处理 running 中的活跃请求（decode）
    # 完成的请求从 running 移除，新请求随时加入 waiting
```

---

### 2.6 Preemption（抢占）：显存不够时怎么办

#### 是什么

当 KV Cache 显存不足以让所有 running 序列继续 decode 时，调度器需要**抢占**部分序列：
1. 释放被抢占序列的 KV Cache
2. 将其退回 waiting 队列头部
3. 下次有空间时重新 prefill（优先处理）

#### 比喻：高峰期餐厅让座

> 餐厅满座了，新来了一桌 VIP。经理请最后来的那桌客人暂时去休息区等候（preempt），空出座位给 VIP。等有空位了再请他们回来重新点菜（重新 prefill）。

#### 在代码中

```python
# scheduler.py
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)   # 释放 KV Cache
    self.waiting.appendleft(seq)          # 放回队列头部，下次优先处理
```

---

### 2.7 Flash Attention：高效注意力计算

#### 是什么

标准 Attention 的公式是 `softmax(Q @ K^T / √d) @ V`。朴素实现需要：
1. 计算 N×N 的注意力矩阵（显存 O(N²)）
2. 对矩阵做 softmax
3. 再与 V 相乘

**Flash Attention** 的核心优化：
- **分块计算（Tiling）**：不一次性算出完整 N×N 矩阵，而是分成小块在 SRAM（片上高速缓存）中计算
- **在线 Softmax**：边算边更新 softmax 的分母，不需要等所有分数算完
- **IO 感知**：减少 HBM（显存）的读写次数，大部分计算在 SRAM 中完成

结果：计算量不变，但显存从 O(N²) 降到 O(N)，速度提升 2-4 倍。

#### 比喻：开卷考试

**标准 Attention**：
> 老师让你把全班 100 个同学的成绩单全部打印出来铺在桌上（N×N 矩阵存入显存），然后找出每个人的相对排名。桌子不够大（显存不够）就完蛋了。

**Flash Attention**：
> 你一次只拿 10 个人的成绩单出来（分块），在小抄本（SRAM）上算出这 10 人的排名，记录下来，再拿下一批 10 人。最后用数学方法（在线 softmax）合并所有批次的结果。桌子再小也能做。

#### 在代码中

nano-vllm 直接调用 `flash_attn` 库，区分两种模式：

```python
# attention.py
if context.is_prefill:
    # Prefill: 变长序列，用 varlen 接口（多条 prompt 拼成一维，用 cu_seqlens 区分边界）
    o = flash_attn_varlen_func(q, k, v, ...)
else:
    # Decode: 每条序列仅 1 个新 token，从 paged KV Cache 读取历史 KV
    o = flash_attn_with_kvcache(q, k_cache, v_cache, block_table=..., ...)
```

---

### 2.8 RoPE（旋转位置编码）：让模型知道 token 的位置

#### 是什么

Transformer 本身不知道 token 的顺序（"我爱你" 和 "你爱我" 对它来说一样）。RoPE 通过**旋转矩阵**给 Q 和 K 编码位置信息：

- 将每个 head 的维度视为若干二维平面
- 每个平面按 token 位置旋转不同角度
- 低维转得快（捕捉近距离关系），高维转得慢（捕捉远距离关系）

核心性质：两个 token 的 Q·K 内积只取决于**相对位置**，而非绝对位置。

#### 比喻：时钟指针

> 想象 64 个时钟（head_dim=128, 即 64 个二维平面），每个时钟转速不同：
>
> - 第 1 个时钟：秒针，每个 token 转很大角度 → 敏感于相邻 token 的距离
> - 第 32 个时钟：分针，转得慢 → 敏感于中等距离
> - 第 64 个时钟：时针，转得很慢 → 敏感于远距离关系
>
> 两个 token 的"相似度"取决于所有时钟指针的相对角度差。距离越近的 token，快时钟的角度差越小。

#### 在代码中

```python
# rotary_embedding.py
# 频率基底：低维频率高（转得快），高维频率低（转得慢）
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
# 旋转公式
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

---

### 2.9 RMSNorm：轻量级归一化

#### 是什么

LayerNorm 需要计算均值和方差两个统计量。**RMSNorm** 省去均值计算，只用均方根（Root Mean Square）做归一化：

```
RMSNorm(x) = x / √(mean(x²) + ε) × weight
```

计算量更少，效果几乎一样。现代 LLM（LLaMA、Qwen）都用 RMSNorm。

#### 在代码中的融合优化

```python
# layernorm.py: 融合残差加法 + RMSNorm（add_rms_forward）
# 传统方式：两个 kernel → x = x + residual; x = rms_norm(x)
# 融合方式：一个 kernel 搞定 → 省一次显存读写
x = x.float().add_(residual.float())   # 融合的残差加法
residual = x.to(orig_dtype)            # 保存给下一层
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))    # RMSNorm
```

---

### 2.10 SwiGLU：带门控的激活函数

#### 是什么

传统 MLP 用 ReLU：`output = ReLU(x @ W1) @ W2`

SwiGLU 把中间层一分为二，一半做门控（gate），一半做值（value）：
```
SwiGLU(x) = SiLU(x @ W_gate) × (x @ W_up)
```
其中 `SiLU(x) = x × sigmoid(x)`（也叫 Swish）。

门控机制让网络可以学习"哪些信息该通过、哪些该屏蔽"，效果比 ReLU 更好。

#### 比喻：安检门

> ReLU 是一个简单的开关门：信号 > 0 就通过，≤ 0 就拦住。
>
> SwiGLU 是一个智能安检门：有两个通道同时处理你的信息。一个通道（gate）决定"放行多少"（0~1 的比例），另一个通道（up）准备"放行的内容"。最终 = 放行比例 × 放行内容。

#### 在代码中

```python
# activation.py
def forward(self, x):
    x, y = x.chunk(2, -1)    # 切成 gate 和 up 两半
    return F.silu(x) * y      # SiLU(gate) × up

# qwen3.py: gate_proj 和 up_proj 合并为一次 GEMM
self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2)
```

合并 gate 和 up 为一次矩阵乘（GEMM）是重要的性能优化——两次小 GEMM 换成一次大 GEMM，GPU 利用率更高。

---

### 2.11 GQA（Grouped Query Attention）：KV Head 的共享

#### 是什么

标准 Multi-Head Attention 中，每个 Q head 都有对应的 K、V head。当 head 数很多（如 32 个）时，KV Cache 占用显存巨大。

**GQA** 让多个 Q head 共享同一组 K、V head。例如 32 个 Q head、8 个 KV head → 每 4 个 Q head 共享 1 组 KV → KV Cache 缩小到 1/4。

#### 比喻：小组讨论

> 一个班 32 个学生（Q heads）讨论问题，每人都需要参考资料（K/V）。
>
> - **MHA**：每人一份参考资料 → 32 份。
> - **GQA**：4 人一组共享一份参考资料 → 8 份。省了 3/4 的纸张（显存），讨论质量几乎不受影响。

#### 在代码中

```python
# qwen3.py: Q heads 和 KV heads 数量可以不同
self.num_heads = num_heads // tp_size        # Q: 如 32/4=8 per GPU
self.num_kv_heads = num_kv_heads // tp_size  # KV: 如 8/4=2 per GPU
# flash_attn 内部自动处理 GQA 的 head 广播
```

---

### 2.12 Tensor Parallelism（张量并行）：多 GPU 切分计算

#### 是什么

单个 GPU 装不下大模型时，把模型权重**按维度切分**到多个 GPU：

- **Column Parallel**（列切分）：权重按输出维度切，每个 GPU 算输出的一部分，无需通信
- **Row Parallel**（行切分）：权重按输入维度切，每个 GPU 算部分结果，最后 all-reduce 求和

在 Transformer 中的经典搭配（Megatron-LM 方案）：
```
Attention:  qkv_proj (Column) → o_proj (Row) → 1次 all-reduce
MLP:        gate_up_proj (Column) → down_proj (Row) → 1次 all-reduce
```
每个子层只需 **1 次 all-reduce**，通信开销最小。

#### 比喻：分工合作做数学题

> 老师出了一道大矩阵乘法题，一个人算太慢。
>
> **Column Parallel**：4 个学生各算结果矩阵的 1/4 列，互不干扰，各算各的。
>
> **Row Parallel**：4 个学生各拿输入矩阵的 1/4 行，各自算出一份部分结果（和原始结果同尺寸但不完整），然后把 4 份结果加在一起（all-reduce）得到正确答案。

#### 在代码中

```python
# linear.py
class ColumnParallelLinear:    # 每 GPU 持有 [output/tp, input]
    def forward(self, x):
        return F.linear(x, self.weight)   # 无通信

class RowParallelLinear:       # 每 GPU 持有 [output, input/tp]
    def forward(self, x):
        y = F.linear(x, self.weight)
        dist.all_reduce(y)                # 唯一的通信点
        return y
```

多进程间的控制信号通过**共享内存（SharedMemory）+ Event**传递，比 NCCL 广播更轻量。

---

### 2.13 CUDA Graph：录制并回放 GPU 操作

#### 是什么

正常的 PyTorch 推理中，每次前向传播都要：
1. CPU 准备数据 → 2. CPU 发射 kernel → 3. GPU 执行 → 重复

Decode 阶段每次只算 1 个 token，计算量很小，**瓶颈在 CPU 发射 kernel 的开销**（kernel launch overhead）。

**CUDA Graph** 把一整次前向的所有 GPU 操作录制下来，之后直接在 GPU 上"一键回放"，跳过 CPU 调度的开销。

#### 比喻：乐谱 vs 即兴演奏

> **普通模式**（Eager）：指挥家（CPU）每演奏一个音符都要举一次棒子指挥乐团（GPU），乐团等指挥等得很烦。
>
> **CUDA Graph**：指挥家把整首曲子写成乐谱（录制 Graph），乐团拿到乐谱后自己演奏，不用等指挥。演奏速度取决于乐团能力，不受指挥反应速度限制。

#### 限制

- 输入形状必须固定（乐谱写死了音符数量）
- 所以只用于 decode（每次固定 batch_size 个 token），prefill 不用（长度不固定）
- 分桶策略：预捕获 bs=1,2,4,8,16,32,...，实际运行时向上取最近的桶

#### 在代码中

```python
# model_runner.py: 捕获 CUDA Graph
for bs in reversed(self.graph_bs):    # 从大到小捕获，共享 memory pool
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 录制
    self.graphs[bs] = graph

# 回放时：只需更新输入 tensor 的值，地址不变
graph_vars["input_ids"][:bs] = input_ids    # 更新输入值
graph.replay()                               # 一键回放
```

---

### 2.14 torch.compile：JIT 编译加速

#### 是什么

`torch.compile` 是 PyTorch 2.0 引入的 JIT 编译器。它分析 Python 代码，将多个小 kernel 融合（fuse）为少数大 kernel，减少 kernel launch 次数和显存读写。

在 nano-vllm 中，`@torch.compile` 用于以下计算密集型组件：
- `SiluAndMul`（SwiGLU 激活）
- `RMSNorm`（含融合残差加法版本）
- `RotaryEmbedding`（RoPE 旋转）
- `Sampler`（Gumbel-max 采样）

#### 比喻：快递打包

> 你有 10 个包裹（10 个小 kernel）要寄。
>
> - **不编译**：每个包裹单独寄，跑 10 趟快递站。
> - **torch.compile**：编译器帮你把 10 个包裹打包成 2 个大箱子（kernel fusion），只跑 2 趟。

---

### 2.15 Triton Kernel：自定义 GPU 计算

#### 是什么

Triton 是 OpenAI 开发的 GPU 编程语言，比 CUDA 更高级（不需要手动管线程），同时比 PyTorch 更底层（可以精确控制内存访问模式）。

在 nano-vllm 中，唯一的 Triton kernel 用于将 K/V 写入 Paged KV Cache：

```python
# attention.py: store_kvcache_kernel
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx = tl.program_id(0)                    # 第几个 token
    slot = tl.load(slot_mapping_ptr + idx)     # 该 token 的物理 slot
    if slot == -1: return                      # padding，跳过
    # 从 key/value 读取，写入 cache 对应 slot
    tl.store(k_cache_ptr + slot * D + ..., key)
```

#### 为什么需要

PyTorch 原生操作按连续内存索引。但 Paged KV Cache 中每个 token 的物理位置由 `slot_mapping` 决定（可能不连续），需要 scatter write。Triton kernel 可以用一行 `tl.store(addr + offset)` 完成这种非连续写入。

---

### 2.16 Gumbel-Max Sampling：高效采样

#### 是什么

从 logits 到 token id 需要"按概率采样"。传统方法：
1. softmax → 概率分布
2. 构建累积分布函数（CDF）
3. 生成均匀随机数，二分查找对应 token

**Gumbel-max 技巧**：数学上等价，但更高效：
```
token = argmax(probs / Exp(1))
```
其中 Exp(1) 是指数分布随机数。全程只需 element-wise 运算 + argmax，完美适合 GPU 并行。

#### 比喻：加噪音的选举

> 有 100 个候选人（token），每人有不同的支持率（概率）。
>
> - **传统方法**：画一个大饼图（CDF），扔飞镖，看落在哪个扇区。需要精确计算每个扇区边界。
> - **Gumbel-max**：给每个候选人的得票数加上一个随机噪音（噪音大小恰好让"得票最高者 = 按概率抽中者"），然后直接选票最多的。不需要画饼图，每个候选人独立计算。

---

## 三、概念之间的关系

### 3.1 一次推理请求的完整生命周期

```
用户输入 "Hello"
    │
    ▼
① Tokenizer: "Hello" → [15496]
    │
    ▼
② Scheduler.add(): 创建 Sequence, 加入 waiting 队列
    │
    ▼
③ Scheduler.schedule(): 从 waiting 取出, 进入 prefill
    │  ├── BlockManager.allocate(): 分配 KV Cache Block（尝试 Prefix Cache）
    │  └── 标记为 RUNNING, 移入 running 队列
    │
    ▼
④ ModelRunner.prepare_prefill(): 准备输入
    │  ├── input_ids: [15496]
    │  ├── positions: [0]
    │  ├── slot_mapping: [物理 slot 位置]
    │  └── cu_seqlens: [0, 1]（varlen 接口的序列边界）
    │
    ▼
⑤ Model.forward(): 逐层计算
    │  ├── Embedding:  token_id → 向量
    │  ├── For each layer:
    │  │   ├── RMSNorm(+ 残差融合)
    │  │   ├── QKV Proj (ColumnParallel, 一次 GEMM)
    │  │   ├── RoPE 旋转位置编码
    │  │   ├── store_kvcache (Triton kernel, 写入 KV Cache)
    │  │   ├── Flash Attention (flash_attn_varlen_func)
    │  │   ├── O Proj (RowParallel, all-reduce)
    │  │   ├── RMSNorm(+ 残差融合)
    │  │   └── MLP: gate_up(Column) → SwiGLU → down(Row, all-reduce)
    │  ├── Final RMSNorm
    │  └── LM Head: hidden → logits (gather to rank 0)
    │
    ▼
⑥ Sampler: logits → token_id（Gumbel-max, temperature 缩放）
    │
    ▼
⑦ Scheduler.postprocess(): 将 token 追加到序列
    │  ├── seq.append_token(token_id)
    │  ├── 检查: 是否 EOS? 是否达到 max_tokens?
    │  └── 未结束 → 继续 decode; 结束 → FINISHED, 释放 Block
    │
    ▼
⑧ 回到 ③, 但这次 schedule() 走 decode 路径:
    │  ├── 只送 last_token
    │  ├── 用 CUDA Graph 回放（跳过 CPU 开销）
    │  ├── flash_attn_with_kvcache 读取历史 KV
    │  └── BlockManager.may_append(): 可能分配新 Block
    │
    ▼
⑨ 循环 ⑧ 直到 FINISHED
    │
    ▼
⑩ Tokenizer.decode(): token_ids → "Hello! I'm a helpful assistant..."
```

### 3.2 概念关系图

```
                    ┌─────────────────────────┐
                    │     Continuous Batching  │
                    │    (调度多条序列的生命周期)  │
                    └──────┬──────────────────┘
                           │ 决定 prefill/decode
                    ┌──────▼──────┐
                    │  Scheduler  │◄──── Preemption（显存不足时抢占）
                    └──────┬──────┘
                           │ 分配/释放 Block
                    ┌──────▼──────────┐
                    │  Block Manager  │◄──── Prefix Caching（哈希链共享 Block）
                    │  (Paged Attn)   │
                    └──────┬──────────┘
                           │ 物理 slot 映射
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌───────────┐ ┌──────────────┐
     │  KV Cache  │ │ Flash Attn│ │ Triton Kernel│
     │  (显存池)   │ │ (高效计算) │ │ (KV写入)     │
     └────────────┘ └───────────┘ └──────────────┘
            ▲              ▲
            │              │
     ┌──────┴──────┐ ┌────┴─────┐
     │    GQA      │ │   RoPE   │
     │ (KV head   │ │ (位置编码) │
     │  压缩)      │ └──────────┘
     └─────────────┘

     ┌────────────────────────────────────────────┐
     │              执行优化                        │
     │  CUDA Graph ← Decode 阶段固定形状            │
     │  torch.compile ← 算子融合                    │
     │  Tensor Parallel ← 多 GPU 切分              │
     │  Pin Memory ← CPU→GPU 异步传输               │
     └────────────────────────────────────────────┘
```

### 3.3 核心矛盾与解决方案

| 矛盾 | 解决方案 | 对应概念 |
|------|---------|---------|
| KV Cache 占用显存太大 | 按需分页分配 | Paged Attention / Block Manager |
| 相同前缀重复计算 | 内容哈希共享 | Prefix Caching |
| 短请求等长请求 | 动态凑批 | Continuous Batching |
| Decode 计算量小但 CPU 开销大 | 录制回放 | CUDA Graph |
| 多个小 kernel 浪费 launch 开销 | 编译融合 | torch.compile |
| Attention 显存 O(N²) | 分块在 SRAM 计算 | Flash Attention |
| 模型太大单卡装不下 | 权重切分 | Tensor Parallelism |
| KV head 太多占显存 | Q head 共享 KV head | GQA |
| KV Cache 不足 | 抢占低优先级序列 | Preemption |

---

## 四、从代码文件到概念的映射

| 文件 | 核心概念 | 行数 |
|------|---------|------|
| `llm.py` | API 入口（LLMEngine 别名） | 7 |
| `config.py` | 全局配置 | 27 |
| `sampling_params.py` | 采样参数 | 13 |
| `engine/llm_engine.py` | 主循环、Continuous Batching 驱动 | 109 |
| `engine/scheduler.py` | 调度器、Preemption | 89 |
| `engine/block_manager.py` | Paged Attention、Prefix Caching | 142 |
| `engine/sequence.py` | 序列生命周期、Block Table | 92 |
| `engine/model_runner.py` | CUDA Graph、Tensor Parallel 通信、数据准备 | 319 |
| `layers/attention.py` | Flash Attention、Triton KV Cache 写入 | 95 |
| `layers/linear.py` | Tensor Parallelism（Column/Row Parallel） | 215 |
| `layers/activation.py` | SwiGLU | 24 |
| `layers/embed_head.py` | Vocab Parallel Embedding + LM Head | 104 |
| `layers/layernorm.py` | RMSNorm（含残差融合） | 69 |
| `layers/rotary_embedding.py` | RoPE | 86 |
| `layers/sampler.py` | Gumbel-Max Sampling | 28 |
| `models/qwen3.py` | 模型组装、GQA、权重映射 | 236 |
| `utils/context.py` | 全局 Attention 上下文传递 | 35 |
| `utils/loader.py` | 权重加载、packed modules 处理 | 44 |

**总计 ~1,735 行**（含注释），实现了工业级推理引擎的核心功能。

---

## 五、建议学习顺序

1. **先理解两个阶段**：Prefill vs Decode（2.1）— 这是所有优化的出发点
2. **再理解 KV Cache**（2.2）— 理解为什么 Decode 需要缓存，以及缓存有多大
3. **然后学 Paged Attention**（2.3）— 理解怎么高效管理 KV Cache 显存
4. **接着学 Continuous Batching**（2.5）— 理解怎么同时处理多个请求
5. **最后学执行优化**：Flash Attention（2.7）、CUDA Graph（2.13）、TP（2.12）
6. **模型结构部分**（RoPE、RMSNorm、SwiGLU、GQA）可穿插阅读

每个概念先读本文的比喻理解直觉，再去对应源文件看实现。
