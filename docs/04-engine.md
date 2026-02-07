# 阶段 4：推理引擎

## 目标

理解 continuous batching 调度和 paged KV cache 管理的完整实现。

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `engine/sequence.py` | ~90 | 序列数据结构，贯穿请求生命周期 |
| `engine/block_manager.py` | ~140 | Paged KV Cache 分配/释放 + Prefix Caching |
| `engine/scheduler.py` | ~90 | Continuous Batching 调度 + 抢占 |
| `engine/llm_engine.py` | ~110 | 引擎主类，组装调度与执行 |

## 建议阅读顺序

```
sequence.py → block_manager.py → scheduler.py → llm_engine.py
数据结构 → 内存管理 → 调度策略 → 顶层组装
```

---

## 第一部分：Sequence 数据结构

### 生命周期

```
创建(WAITING) → 被调度 prefill(RUNNING) → decode 循环(RUNNING)
                                           │
                  ┌── 显存不足时 ──────────── 抢占(WAITING) → 重新 prefill
                  │
                  └── 遇到 eos / max_tokens → FINISHED → 释放 KV cache
```

### 关键属性

| 属性 | 含义 |
|------|------|
| `token_ids` | 完整 token 列表（prompt + 已生成） |
| `last_token` | 最新 token，decode 时只传此值给 worker（省带宽） |
| `num_prompt_tokens` | prompt 长度（固定） |
| `num_cached_tokens` | prefix cache 命中的 token 数（跳过计算） |
| `block_table` | 物理 block id 列表，用于 paged attention 定位 KV cache |

### 序列化优化

`__getstate__` / `__setstate__` 用于跨进程传输（tensor parallel）：
- Prefill 阶段：传完整 `token_ids`（worker 需要所有 token 做前向）
- Decode 阶段：只传 `last_token`（worker 只需最新 1 个 token）

---

## 第二部分：Paged KV Cache

### 核心思想

类似操作系统的虚拟内存分页：
- **物理 block**：固定大小（默认 256 token）的 KV cache 块
- **block_table**：每个序列持有一个逻辑→物理的映射表
- **按需分配**：序列增长时才分配新 block，避免预留最大长度的浪费

### Block 结构

```
Block {
    block_id:   int       # 物理编号
    ref_count:  int       # 引用计数（prefix cache 共享时 > 1）
    hash:       int       # 内容哈希（-1 = 未填满，不可复用）
    token_ids:  list[int] # 内容（用于验证 cache 命中）
}
```

### Prefix Caching

通过内容哈希链实现。相同前缀的序列共享 block，避免重复计算：

```
序列 A: "今天天气" → [block_0: hash_a0] [block_1: hash_a1] [block_2]
序列 B: "今天天气" → [block_0: hash_a0] [block_1: hash_a1] [block_3]
                     ↑ 共享（ref_count=2）  ↑ 共享            ↑ 独立

哈希链：
  hash_a0 = xxh64(token_ids_0)
  hash_a1 = xxh64(hash_a0 + token_ids_1)   ← 前缀哈希级联
```

只有**填满**的 block 才会被注册到 prefix cache（未填满的 block 内容不稳定）。

### 数值示例：分配与释放

```
假设 block_size=4，共 8 个物理 block

初始：free=[0,1,2,3,4,5,6,7]  used={}

── 序列 A 到达，prompt 长度 10（需要 3 个 block：4+4+2）──
allocate(A):
  block 0: token[0:4]  → hash=h0, 注册 prefix cache
  block 1: token[4:8]  → hash=h1, 注册 prefix cache
  block 2: token[8:10] → hash=-1（未填满，不注册）
  A.block_table = [0, 1, 2]
  A.num_cached_tokens = 0（首次，无 cache 命中）

状态：free=[3,4,5,6,7]  used={0,1,2}

── 序列 B 到达，prompt 前 8 个 token 与 A 相同 ──
allocate(B):
  block 0: hash 命中 → ref_count: 1→2，cache hit！
  block 1: hash 命中 → ref_count: 1→2，cache hit！
  block 3: 新分配
  B.block_table = [0, 1, 3]
  B.num_cached_tokens = 8（跳过前 2 个 block 的计算）

状态：free=[4,5,6,7]  used={0,1,2,3}

── 序列 A 完成，释放 ──
deallocate(A):
  block 2: ref_count 1→0 → 回收到 free
  block 1: ref_count 2→1 → 保留（B 还在用）
  block 0: ref_count 2→1 → 保留

状态：free=[4,5,6,7,2]  used={0,1,3}
```

### Decode 阶段的 Block 管理

`may_append` 处理三种情况：

| `num_tokens % block_size` | 含义 | 操作 |
|---|---|---|
| `== 1` | 新 token 跨入新 block | 分配新 block |
| `== 0` | 当前 block 刚好填满 | 计算哈希，注册 prefix cache |
| 其他 | block 未填满 | 无操作 |

---

## 第三部分：Continuous Batching

### 传统 Batching vs Continuous Batching

```
传统 Static Batching：
  batch = [seq_A, seq_B, seq_C]
  等所有序列都完成才处理下一批 → 短序列被长序列拖累

Continuous Batching：
  每一步都重新调度：
  step 1: [A, B, C] → A 完成
  step 2: [B, C, D] → D 插入空位    ← GPU 利用率更高
  step 3: [B, C, D] → B 完成
  ...
```

### 调度策略：Prefill 优先

```python
schedule():
    if waiting 队列非空:
        # Prefill 阶段：贪心地取尽可能多的 waiting 序列
        # 约束：① token 总数 ≤ max_num_batched_tokens
        #       ② 序列数 ≤ max_num_seqs
        #       ③ 有足够空闲 block
        return prefill_seqs, is_prefill=True
    else:
        # Decode 阶段：对所有 running 序列生成 1 个 token
        # 若 block 不足 → 抢占最后加入的序列
        return decode_seqs, is_prefill=False
```

**关键设计**：Prefill 和 Decode **不混合执行**。每次 schedule() 要么全做 prefill，要么全做 decode。这简化了 ModelRunner 的实现（不需要同时处理变长 prefill 和定长 decode）。

### 抢占机制

当 decode 阶段 KV cache 不足时：

```
running: [A, B, C, D]
                    ↑
D 需要新 block 但 free 为空
  → 抢占 D：释放 D 的所有 block → D 回到 waiting 头部
  → 下一轮 D 会重新 prefill（代价：重新计算，但避免死锁）

若抢占 D 后 C 仍不够：
  → 继续抢占 C
  → 最坏情况：抢占自身
```

抢占策略是 **recompute**（释放 → 重新 prefill），而非 swap（换到 CPU）。实现更简单，延迟略高。

### 数值示例：一轮完整调度

```
假设 max_num_seqs=3, max_num_batched_tokens=20

── 初始状态 ──
waiting: [A(len=8), B(len=6), C(len=10)]
running: []

── schedule() 第 1 次：Prefill ──
取 A(8)：8 ≤ 20 ✓ → allocate → batched_tokens=8
取 B(6)：8+6=14 ≤ 20 ✓ → allocate → batched_tokens=14
取 C(10)：14+10=24 > 20 ✗ → 停止
返回: [A, B], is_prefill=True

waiting: [C(len=10)]
running: [A, B]

── step(): model_runner 处理 14 个 token，生成 A/B 各 1 个新 token ──

── schedule() 第 2 次：Prefill ──
C(10) ≤ 20 ✓ → allocate
返回: [C], is_prefill=True

waiting: []
running: [A, B, C]

── schedule() 第 3 次起：Decode ──
返回: [A, B, C], is_prefill=False
（每次生成 3 个 token，直到某序列完成）

── A 完成（postprocess 检测到 eos）──
running: [B, C]

── 新请求 D 到达 ──
waiting: [D]

── schedule()：Prefill 优先 ──
返回: [D], is_prefill=True
（D prefill 后进入 running，下一步与 B、C 一起 decode）
```

---

## 第四部分：主循环

### step() — 引擎心跳

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()          # ① 调度
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 执行
    self.scheduler.postprocess(seqs, token_ids)           # ③ 更新
    outputs = [(seq.seq_id, seq.completion_token_ids)     # ④ 收集完成的
               for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

每次 `step()` 完成一个"调度→推理→后处理"循环。`generate()` 反复调用 `step()` 直到所有序列完成。

### num_tokens 的正负含义

| 值 | 含义 | 用途 |
|---|---|---|
| `> 0` | prefill，值 = 实际计算的 token 总数 | 计算 prefill 吞吐量 |
| `< 0` | decode，值 = -序列数 | 计算 decode 吞吐量（每序列 1 token） |

---

## 核心问题回答

**Q: Continuous Batching 相比 Static Batching 好在哪？**

Static Batching 必须等一批中最慢的序列完成才能处理下一批。短序列生成完后 GPU 空转。Continuous Batching 每一步都重新调度：序列完成即释放资源，新请求立刻插入，GPU 始终满载。

**Q: KV Cache 为什么要分页（paged）？**

如果为每个序列预分配最大长度的 KV cache，大量显存被浪费（大多数序列远没那么长）。分页后按需分配 block，显存利用率接近 100%。副产品是 prefix caching——相同前缀的序列共享 block，进一步节省显存和计算。

**Q: 抢占时为什么选择 recompute 而非 swap？**

Swap 需要把 KV cache 从 GPU 搬到 CPU 再搬回来，实现复杂且 PCIe 带宽是瓶颈。Recompute 直接释放 block，下次重新 prefill。对于大多数场景（抢占概率低），recompute 更简单高效。

**Q: Prefill 和 Decode 为什么不混合？**

两者的计算模式差异巨大：Prefill 是计算密集（处理长 prompt），Decode 是访存密集（每序列只算 1 token 但要读全部 KV cache）。混合执行需要 ModelRunner 同时处理变长输入和定长输入，增加复杂度。分开执行后 kernel 可以各自优化。
