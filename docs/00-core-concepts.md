# nano-vllm 核心概念：问题→解决方案级联

> 本文档以"根本矛盾"为起点，沿**问题→解决方案→新问题**的因果链，串联 LLM 推理的所有核心概念。
>
> **统一比喻**：GPU = 专业厨房，HBM（显存）= 储藏室，SRAM（片上缓存）= 操作台，每条请求 = 一张客单。个别概念用更贴切的独立比喻，会标注说明。

---

## 第一章：为什么 LLM 推理这么难

### 自回归的枷锁

LLM 生成文本的本质是**自回归**（autoregressive）：每次只能生成一个 token，且下一个 token 依赖前一个的结果。你无法跳过第 5 个 token 直接算第 6 个。

这就像一个厨师做寿司卷：必须从左到右一片一片放料，不能同时放——因为右边放什么取决于左边放了什么。

### 三大瓶颈

| 瓶颈 | 本质 | 厨房比喻 |
|------|------|---------|
| **计算** | 矩阵乘法量巨大 | 厨师切菜速度有限 |
| **显存** | KV Cache 随序列增长 | 储藏室被笔记塞满 |
| **延迟** | CPU↔GPU 来回传指令 | 厨师每道菜都等经理口令才动手 |

这三大瓶颈催生了后续所有优化。每一章解决一个瓶颈，同时引出新的问题。

---

## 第二章：Prefill 与 Decode——两种截然不同的工作模式

### 根本区别

| | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 处理什么 | 整条 prompt（可能数百 token） | 每次 1 个新 token |
| 瓶颈类型 | **计算密集**（大量矩阵乘法） | **访存密集**（反复读 KV Cache） |
| 厨房比喻 | 通读整张客单，为所有菜做备注 | 按备注一道一道出菜 |

Scheduler 的核心决策就是：当前这一步做 prefill 还是 decode？

```python
# scheduler.py:50-51 — 有 prefill 任务则直接返回
if scheduled_seqs:
    return scheduled_seqs, True   # prefill

# scheduler.py:72 — 否则走 decode
return scheduled_seqs, False
```

```python
# model_runner.py:262 — 两套数据准备路径
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
```

### 因果链

```
Prefill（一次性算完所有 prompt token）
    │ 产出
    ▼
KV Cache（缓存每层的 Key/Value）
    │ 被读取
    ▼
Decode（每次 1 token，查阅 KV Cache 生成下一个）
    │ 循环
    └──▶ Decode ──▶ Decode ──▶ ... ──▶ EOS
```

**引出的问题**：KV Cache 越来越大，显存怎么管？→ 第三章。

---

## 第三章：显存困局

### 3.1 KV Cache——避免 O(n²) 重复计算

**问题**：Attention 需要当前 token 与所有历史 token 做点积。若每次 decode 都重算所有历史的 K/V，复杂度 O(n²)。

**方案**：把每层计算过的 Key 和 Value 缓存下来（KV Cache），decode 时直接读取。

> **厨房比喻**：厨师的备菜笔记——每处理一道菜就记下要点，后续菜品只需翻看笔记而不是从头回忆。

显存开销公式：
```
KV Cache = 2(K+V) × num_layers × seq_len × num_kv_heads × head_dim × dtype_bytes
```

**新问题**：序列长度不确定。预分配 max_seq_len 的连续显存？太浪费。→ 3.2

---

### 3.2 Paged KV Cache——按需分配储物格

**问题**：预分配连续显存造成大量浪费——预留 100 格，实际只用 20 格，剩下 80 格空着也没法给别人用。

**方案**：借鉴 OS 虚拟内存，将 KV Cache 切成固定大小的 **Block**（每块 256 token），按需分配，维护 Block Table（页表）记录映射。

> **厨房比喻**：不再给每张客单预留一整排货架，而是按需分配储物格。客单多用几道菜就多拿几格，走了立刻回收。

```python
# block_manager.py:73-102 — allocate 核心逻辑
def allocate(self, seq):
    h = -1
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
        # ...（cache hit 时复用已有 block）
        seq.block_table.append(block_id)
```

**新问题**：很多请求有相同的 system prompt，每个都算一遍？→ 3.3

---

### 3.3 Prefix Caching——20 桌点同一前菜，只做一次

**问题**：100 个请求共享同一个 system prompt（如"你是一个助手…"），每个都重新计算 KV？

**方案**：对每个 block 计算内容哈希，形成**哈希链**（`hash_n = xxh64(hash_{n-1} + tokens_n)`）。相同前缀产生相同哈希链，可直接共享物理 block，引用计数管理生命周期。

> **厨房比喻**：20 桌都点了同一道前菜，厨师只做一次，20 桌共享同一盘（引用计数 = 20）。某桌走了，引用计数 -1，降到 0 才撤盘。

```python
# block_manager.py:42-52 — 哈希链计算
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**新问题**：KV head 数量太多，缓存仍然太大？→ 3.4

---

### 3.4 GQA（Grouped Query Attention）——4 个厨师共用一本笔记

**问题**：标准 Multi-Head Attention 中每个 Q head 配一套 K/V head。32 个 head = 32 份 KV Cache，显存压力巨大。

**方案**：多个 Q head 共享同一组 K/V head。例如 32 个 Q head、8 个 KV head → 每 4 个 Q 共享 1 组 KV → KV Cache 缩小到 1/4。

> **厨房比喻**：原来每个厨师记自己的笔记（32 本）。现在 4 个厨师共用一本笔记（8 本），省了 3/4 存储空间，做菜质量几乎不受影响。

```python
# qwen3.py — GQA 配置
self.num_heads = num_heads // tp_size          # Q: 32/4=8 per GPU
self.num_kv_heads = num_kv_heads // tp_size    # KV: 8/4=2 per GPU
# flash_attn 内部自动处理 GQA 的 head 广播
```

---

### 显存困局小结

```
重复计算 O(n²)
    │ KV Cache
    ▼
预分配浪费
    │ Paged KV Cache
    ▼
相同前缀重复算
    │ Prefix Caching（哈希链）
    ▼
KV head 太多
    │ GQA（共享 KV head）
    ▼
显存问题基本解决 ✓
```

---

## 第四章：调度困局

### 4.1 Continuous Batching——传送带寿司 vs 包桌

**问题**：Static Batching 必须等一批全部完成才能上新。一条请求 10 token 就完了，另一条要 1000 token，短请求被迫空等。

**方案**：完成的随时离开，新的随时加入。GPU 像传送带寿司一样持续满载运转。

> **厨房比喻**：
> - 包桌（Static Batching）：10 桌一起上菜，等最慢的那桌吃完才收台。
> - 传送带（Continuous Batching）：吃完一盘立刻撤走，新盘子立刻上来，传送带永远满载。

```python
# scheduler.py:27-72 — schedule() 核心
def schedule(self):
    # 1. 优先从 waiting 取序列做 prefill
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
           or not self.block_manager.can_allocate(seq):
            break
        self.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True      # prefill

    # 2. 无新请求时，running 中的序列做 decode
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq); break
        else:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    return scheduled_seqs, False         # decode
```

---

### 4.2 Prefill-first 策略——新客单优先

**设计选择**：Scheduler 总是优先处理 waiting 中的 prefill 请求。

**为什么**：Prefill 是计算密集的，能让 GPU 满载运转。Decode 是访存密集的，GPU 利用率较低。优先做 prefill，能更快地把新请求"暖起来"。

---

### 4.3 Preemption——储藏室满了，暂停最后来的客单

**问题**：所有 block 都被占用，新来的 decode token 无处写入。

**方案**：抢占最后加入的序列——释放它的 KV Cache，退回 waiting 队列头部。下次有空间时优先重新 prefill。

> **厨房比喻**：储藏室满了，经理请最后来的那桌暂时去休息区。等有空格了再请他们回来，从头备菜。

```python
# scheduler.py:74-78
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)    # 放回队列头部，下次优先
```

---

## 第五章：计算困局

### 5.1 Flash Attention——不铺开整张点菜单

**问题**：标准 Attention 需要计算 N×N 的注意力矩阵，显存 O(N²)。序列长 4096 → 矩阵有 1600 万个元素。

**方案**：分块计算（Tiling）。不一次性算出完整 N×N 矩阵，而是分成小块在 SRAM（操作台）上计算，用在线 Softmax 边算边更新。

> **厨房比喻**：不把所有客单铺满整个操作台（操作台放不下），而是一次取 10 张客单放到操作台上处理，处理完换下一批。用一个小本子记录中间结果（在线 Softmax），最后合并。

nano-vllm 区分两条路径：

```python
# attention.py:77-93
if context.is_prefill:
    # Prefill：变长序列拼接，用 varlen 接口（cu_seqlens 区分边界）
    o = flash_attn_varlen_func(q, k, v, ...)
else:
    # Decode：每序列仅 1 个新 token，从 paged KV cache 读取历史
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens,
                                block_table=context.block_tables, ...)
```

---

### 5.2 RoPE（旋转位置编码）——时钟指针

> **独立比喻（时钟指针）**：Transformer 本身不知道 token 的顺序。RoPE 用旋转矩阵编码位置。

想象 64 个时钟（head_dim=128，即 64 个二维平面），每个转速不同：

| 时钟 | 转速 | 敏感距离 |
|------|------|---------|
| 第 1 个 | 秒针（快） | 相邻 token |
| 第 32 个 | 分针（中） | 中等距离 |
| 第 64 个 | 时针（慢） | 远距离关系 |

核心性质：两个 token 的 Q·K 内积只取决于**相对位置**。

```python
# rotary_embedding.py — 频率公式
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))

# 旋转：低维转快（频率高），高维转慢（频率低）
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

---

### 5.3 RMSNorm + 残差融合——一趟完成两件事

**问题**：LayerNorm 需要算均值和方差两个统计量。现代 LLM 只需均方根就够了。

**方案**：RMSNorm 省去均值计算，且与残差加法融合为一个 kernel，省一次显存读写。

```
RMSNorm(x) = x / √(mean(x²) + ε) × weight
```

```python
# layernorm.py — 融合残差 + RMSNorm
x = x.float().add_(residual.float())     # 残差加法
residual = x.to(orig_dtype)              # 保存给下一层
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))      # RMSNorm
```

传统方式需要两个 kernel（先 add，再 norm），融合后一个 kernel 搞定。

---

### 5.4 SwiGLU——智能安检门

**问题**：ReLU 只是简单的"正通过、负拦截"。

**方案**：SwiGLU 将中间层一分为二：一半做门控（gate），一半做值（value）。

> **厨房比喻**：安检门有两个通道——一个判断"放行多少"（gate），一个准备"放行的内容"（value）。最终 = 放行比例 × 放行内容。

```python
# activation.py — 3 行核心
x, y = x.chunk(2, -1)     # 切成 gate 和 value 两半
return F.silu(x) * y       # SiLU(gate) × value
```

gate_proj 和 up_proj 合并为一次 GEMM（`MergedColumnParallelLinear`），两次小矩阵乘换成一次大矩阵乘，GPU 利用率更高。

---

## 第六章：执行困局

### 6.1 CUDA Graph——把整首曲子写成乐谱

**问题**：Decode 每次只算 1 个 token，计算量极小，但 CPU 发射每个 kernel 都有开销。大量时间浪费在 CPU→GPU 指令传递上。

**方案**：录制一整次前向的所有 GPU 操作（CUDA Graph），之后直接在 GPU 上"一键回放"，跳过 CPU 调度。

> **厨房比喻**：
> - 普通模式：经理（CPU）每道菜都要口头下达指令，厨师（GPU）等指令等得烦。
> - CUDA Graph：经理把整张客单写成操作手册（录制），厨师拿到手册自主执行（回放），不再等经理。

**限制**：输入形状必须固定 → 只用于 decode（每次固定 batch_size），prefill 不用（长度不固定）。分桶策略：预捕获 bs=1,2,4,8,16,32,...，运行时向上取最近桶。

```python
# model_runner.py:298-306 — 捕获（从大到小，共享 memory pool）
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    self.graphs[bs] = graph

# model_runner.py:250-257 — 回放（只更新值，地址不变）
graph_vars["input_ids"][:bs] = input_ids
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph.replay()
```

---

### 6.2 torch.compile——10 个包裹打包成 2 个大箱子

**问题**：很多计算由多个小 kernel 组成，每次 launch 都有开销。

**方案**：`torch.compile` 将多个小 kernel 融合为少数大 kernel（kernel fusion），减少 launch 次数和显存读写。

nano-vllm 中用 `@torch.compile` 装饰的组件：
- `SiluAndMul`（SwiGLU 激活）
- `RMSNorm`（含融合残差加法）
- `RotaryEmbedding`（RoPE 旋转）
- `Sampler`（Gumbel-max 采样）

> **厨房比喻**：把 10 个小包裹合并成 2 个大箱子发货，跑 2 趟而不是 10 趟。

---

### 6.3 Triton Kernel——非连续写入的专用工具

**问题**：Paged KV Cache 中每个 token 的物理位置由 slot_mapping 决定（可能不连续）。PyTorch 原生操作只能按连续内存索引。

**方案**：用 Triton 编写自定义 kernel，一行 `tl.store(addr + offset)` 完成 scatter write。

```python
# attention.py:12-32 — Triton kernel
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx = tl.program_id(0)                     # 第几个 token
    slot = tl.load(slot_mapping_ptr + idx)      # 物理 slot
    if slot == -1: return                       # padding
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
```

---

## 第七章：规模困局——Tensor Parallelism

### 问题

单卡显存装不下大模型。

### 方案：按维度切分权重到多 GPU

| 策略 | 切分方式 | 通信 | 厨房比喻 |
|------|---------|------|---------|
| Column Parallel | 按输出维度切 | **无** | 4 个厨师各切 1/4 的菜，互不干扰 |
| Row Parallel | 按输入维度切 | **all-reduce** | 4 个厨师各拿部分原料做菜，最后合在一起 |

Megatron-LM 经典搭配：每个子层只需 **1 次 all-reduce**。

```
Attention:  qkv_proj (Column) → o_proj (Row) → 1 次 all-reduce
MLP:        gate_up_proj (Column) → down_proj (Row) → 1 次 all-reduce
```

```python
# linear.py — 两种并行的 forward
class ColumnParallelLinear:
    def forward(self, x):
        return F.linear(x, self.weight)       # 无通信

class RowParallelLinear:
    def forward(self, x):
        y = F.linear(x, self.weight)
        dist.all_reduce(y)                    # 唯一通信点
        return y
```

### 通信机制

- **数据面**：NCCL all-reduce（大张量求和）
- **控制面**：SharedMemory + Event（调度指令广播，比 NCCL 轻量）

```python
# model_runner.py:98-106 — rank 0 向 worker 广播指令
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    self.shm.buf[0:4] = len(data).to_bytes(4, "little")
    self.shm.buf[4:len(data)+4] = data
    for event in self.event:
        event.set()                           # 通知所有 worker
```

---

## 第八章：采样——Gumbel-Max

> **独立比喻（加噪音的选举）**

**传统方式**：softmax → 构建 CDF → 扔飞镖看落在哪个扇区（需要排序或累加，串行瓶颈）。

**Gumbel-max 技巧**：给每个候选 token 的得票数加一个精心设计的随机噪音，然后直接选票最高的。数学上等价于按概率采样，但全程可并行。

```python
# sampler.py:18-27 — 3 行核心
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

公式本质：`argmax(softmax(logits/T) / Exp(1))`，其中 `Exp(1)` 是指数分布随机数。

---

## 第九章：全景关系图

### 9.1 问题→解决方案级联图（竖向因果链）

```
        LLM 推理的根本矛盾：自回归 + 模型巨大
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    显存困局      调度困局      计算困局
        │           │           │
        ▼           ▼           ▼
   KV Cache    Continuous   Flash Attention
        │       Batching         │
        ▼           │           ▼
  Paged KV Cache    ▼        RoPE / RMSNorm
        │       Preemption    / SwiGLU
        ▼                       │
  Prefix Caching                ▼
        │                   CUDA Graph
        ▼                   torch.compile
      GQA                   Triton Kernel
                                │
                                ▼
                        Tensor Parallelism
                        （模型太大单卡装不下）
```

### 9.2 数据流图（横向从输入到输出）

```
用户输入
  │
  ▼
Tokenizer ──▶ Scheduler.add() ──▶ waiting 队列
                                      │
                               schedule()
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
               Prefill                               Decode
           (prepare_prefill)                    (prepare_decode)
                    │                                    │
                    └────────────┬───────────────────────┘
                                 ▼
                          Model.forward()
                    ┌────────────────────────┐
                    │ Embedding              │
                    │ for layer in layers:   │
                    │   RMSNorm + 残差融合     │
                    │   QKV Proj (Column TP)  │
                    │   RoPE                 │
                    │   store_kvcache(Triton) │
                    │   Flash Attention      │
                    │   O Proj (Row TP)      │
                    │   RMSNorm + 残差融合     │
                    │   MLP (SwiGLU + TP)    │
                    │ Final RMSNorm          │
                    │ LM Head → logits       │
                    └────────────────────────┘
                                 │
                                 ▼
                    Sampler (Gumbel-max)
                                 │
                                 ▼
                    Scheduler.postprocess()
                    ├─ EOS / max_tokens → FINISHED
                    └─ 否则 → 继续 Decode 循环
```

### 9.3 核心矛盾与解决方案对照表

| 矛盾 | 方案 | 对应概念 | 章节 |
|------|------|---------|------|
| KV 重复计算 O(n²) | 缓存历史 K/V | KV Cache | 3.1 |
| 预分配显存浪费 | 分页按需分配 | Paged KV Cache | 3.2 |
| 相同前缀重复算 | 内容哈希共享 | Prefix Caching | 3.3 |
| KV head 太多 | Q head 共享 KV | GQA | 3.4 |
| 短请求等长请求 | 动态凑批 | Continuous Batching | 4.1 |
| KV Cache 不足 | 抢占低优先级 | Preemption | 4.3 |
| Attention 显存 O(N²) | 分块 SRAM 计算 | Flash Attention | 5.1 |
| Decode CPU 开销大 | 录制回放 | CUDA Graph | 6.1 |
| 小 kernel launch 多 | 编译融合 | torch.compile | 6.2 |
| 非连续 KV 写入 | 自定义 kernel | Triton | 6.3 |
| 单卡装不下 | 权重切分 | Tensor Parallelism | 7 |

---

## 第十章：代码文件↔概念映射表

| 文件 | 核心概念 | 行数 | 章节 |
|------|---------|------|------|
| `engine/scheduler.py` | Continuous Batching, Preemption | 89 | 4 |
| `engine/block_manager.py` | Paged KV Cache, Prefix Caching | 142 | 3.2-3.3 |
| `engine/sequence.py` | 序列生命周期, Block Table | 92 | 2-4 |
| `engine/model_runner.py` | CUDA Graph, TP 通信, 数据准备 | 319 | 2, 6, 7 |
| `layers/attention.py` | Flash Attention, Triton KV 写入 | 95 | 5.1, 6.3 |
| `layers/linear.py` | Tensor Parallelism (Column/Row) | 215 | 7 |
| `layers/activation.py` | SwiGLU | 24 | 5.4 |
| `layers/layernorm.py` | RMSNorm + 残差融合 | 69 | 5.3 |
| `layers/rotary_embedding.py` | RoPE | 86 | 5.2 |
| `layers/sampler.py` | Gumbel-Max Sampling | 28 | 8 |
| `layers/embed_head.py` | Vocab Parallel Embedding | 104 | 7 |
| `models/qwen3.py` | 模型组装, GQA | 236 | 3.4, 5, 7 |
| `utils/context.py` | 全局 Attention 上下文 | 35 | 2 |
| `utils/loader.py` | 权重加载 | 44 | — |
| `engine/llm_engine.py` | 主循环驱动 | 109 | 4 |
| `config.py` | 全局配置 | 27 | — |
| `sampling_params.py` | 采样参数 | 13 | 8 |

**总计 ~1,735 行**

---

## 第十一章：学习路径

### 路径 A：ML 新手

```
第一章（为什么难）
  → 第二章（Prefill/Decode 直觉）
    → 第三章 3.1（KV Cache 概念）
      → 第五章 5.2-5.4（RoPE/RMSNorm/SwiGLU 概念，跳过代码）
        → 第四章 4.1（Continuous Batching 概念）
          → 第九章（全景图，建立整体印象）
```

### 路径 B：有 ML 背景

```
第二章 → 第三章（完整） → 第四章 → 第五章 → 第六章 → 第七章
每章先读比喻，再对照代码片段，最后去源文件看完整实现。
推荐顺序：scheduler.py → block_manager.py → attention.py → model_runner.py
```

### 路径 C：熟悉 vLLM

直接看差异和简化：
1. **第十章映射表**：定位感兴趣的概念对应的文件
2. 重点关注 nano-vllm 的简化设计：
   - 单模型（Qwen3）→ 无 model registry
   - Gumbel-max 采样 → 无 CDF 构建
   - SharedMemory 控制面 → 无 Ray/ZMQ
   - 无 speculative decoding / pipeline parallelism
3. 全部代码 ~1,735 行，可从头到尾通读
