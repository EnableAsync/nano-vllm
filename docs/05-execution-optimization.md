# 阶段 5：执行优化

## 目标

理解 `torch.compile`、CUDA Graph、Tensor Parallel 通信等推理执行层面的优化手段。

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `utils/context.py` | ~35 | 全局注意力上下文，避免 forward 参数层层传递 |
| `utils/loader.py` | ~44 | 权重加载，处理 TP 分片与合并权重映射 |
| `engine/model_runner.py` | ~319 | 模型前向执行，含 CUDA Graph 捕获/回放与 TP 通信 |

## 建议阅读顺序

```
context.py → loader.py → model_runner.py
全局上下文 → 权重加载 → 执行核心（初始化 → 数据准备 → CUDA Graph → TP 通信）
```

---

## 第一部分：全局注意力上下文 (`context.py`)

### 问题

Attention 计算需要大量元数据（cu_seqlens、slot_mapping、block_tables 等），如果通过 `forward(x, context)` 层层传递，每个模块都要改签名，侵入性太强。

### 解决方案

用模块级全局变量存储，三个函数管理生命周期：

```
set_context()   →   ModelRunner 准备好数据后调用
get_context()   →   Attention.forward() 中读取
reset_context() →   每次推理结束后清空
```

### 字段说明

| 字段 | Prefill | Decode | 含义 |
|------|---------|--------|------|
| `is_prefill` | True | False | 区分阶段，Attention 据此选择不同 kernel |
| `cu_seqlens_q` | ✓ | ✗ | query 累积长度前缀和（flash_attn_varlen 接口） |
| `cu_seqlens_k` | ✓ | ✗ | key 累积长度前缀和 |
| `max_seqlen_q/k` | ✓ | ✗ | batch 中最长序列长度（kernel 优化用） |
| `slot_mapping` | ✓ | ✓ | 每个 token 写入 KV cache 的物理 slot |
| `context_lens` | ✗ | ✓ | 每个序列的上下文长度 |
| `block_tables` | 有 cache 时 | ✓ | block table（物理 block id 列表） |

---

## 第二部分：权重加载 (`loader.py`)

### 两种加载路径

```
                    ┌─ 命中 packed_modules_mapping
weight_name ────────┤     → 替换名称，调用 param.weight_loader(param, weight, shard_id)
                    │       例：q_proj → qkv_proj 的 "q" 切片
                    │
                    └─ 未命中
                          → 直接 param.data.copy_(weight)
```

### packed_modules_mapping 示例

HuggingFace checkpoint 中 Q/K/V 是三个独立权重，但推理时合并为 `qkv_proj` 做一次 GEMM：

```python
# Qwen3ForCausalLM 定义的映射
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", "gate"),
    "up_proj":   ("gate_up_proj", "up"),
}
```

加载 `model.layers.0.self_attn.q_proj.weight` 时：
1. 匹配到 `"q_proj"` → 目标参数名替换为 `qkv_proj`
2. `shard_id="q"` → `weight_loader` 将权重写入 qkv_proj 的 Q 对应行区间

---

## 第三部分：ModelRunner 初始化流程

```
__init__
  │
  ├─ ① dist.init_process_group("nccl")    # 初始化多卡通信
  ├─ ② Qwen3ForCausalLM + load_model      # 构建模型 + 加载权重
  ├─ ③ warmup_model()                      # 最大输入跑一次前向
  │     └─ 目的 a: 触发 torch.compile JIT 编译
  │     └─ 目的 b: 记录峰值显存（供 KV cache 分配）
  ├─ ④ allocate_kv_cache()                 # 用剩余显存分配 KV cache
  ├─ ⑤ capture_cudagraph()                 # 捕获 decode 用的 CUDA Graph
  │
  └─ TP 通信初始化
      ├─ rank 0: 创建 SharedMemory
      └─ worker: 打开 SharedMemory → 进入 loop() 等待指令
```

### KV cache 分配公式

```
可用字节 = total × gpu_memory_utilization - used - peak + current

每 block 字节 = 2(K+V) × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes

num_blocks = 可用字节 // 每 block 字节
```

**数值示例**（Qwen3-0.6B, bf16, 单卡）：

```
假设 GPU 24GB, 利用率 0.9
  模型权重 ≈ 1.2GB, warmup 峰值 ≈ 2GB, 当前 ≈ 1.2GB

可用 = 24×0.9 - (24-free) - 2.0 + 1.2 ≈ 18GB（近似）

每 block = 2 × 28层 × 256 × 4heads × 128dim × 2bytes = 28×256×4×128×2×2 ≈ 29.36MB

num_blocks ≈ 18000 / 29.36 ≈ 613 个 block
每 block 256 token → 可缓存 ~157K token
```

---

## 第四部分：数据准备 — Prefill vs Decode

### Prefill (`prepare_prefill`)

处理多条变长序列的 prompt，拼接为一维输入：

```
序列 A: [t0, t1, t2, t3, t4]      len=5
序列 B: [t0, t1, t2]              len=3（其中前 2 个 token 命中 prefix cache）

input_ids  = [t0, t1, t2, t3, t4, t2]     ← B 只发送未缓存部分
positions  = [0, 1, 2, 3, 4, 2]
cu_seqlens_q = [0, 5, 6]                  ← A 贡献 5 个 query token，B 贡献 1 个
cu_seqlens_k = [0, 5, 8]                  ← B 的 key 长度是完整的 3（含缓存）
slot_mapping = [物理 slot 列表]            ← 每个新 token 写入 KV cache 的位置
```

**关键细节**：
- `cu_seqlens_q ≠ cu_seqlens_k` 时说明有 prefix cache 命中，需要 block_tables 读取缓存的 KV
- `pin_memory=True` + `.cuda(non_blocking=True)` 实现 CPU→GPU 异步传输

### Decode (`prepare_decode`)

每个序列只处理 1 个 token：

```
序列 A: 已有 100 个 token，最新 token=42
序列 B: 已有 50 个 token，最新 token=88

input_ids    = [42, 88]
positions    = [99, 49]
context_lens = [100, 50]                ← attention 需要读取的历史长度
slot_mapping = [对应 KV cache 写入位置]
block_tables = [[A 的 block ids], [B 的 block ids]]
```

---

## 第五部分：`torch.compile` — 算子融合

### 原理

`@torch.compile` 让 PyTorch 将 Python 层面的多个算子编译为更少的 CUDA kernel，减少 kernel launch 开销和中间 tensor 的显存读写。

### 5 个使用点

| 位置 | 融合效果 |
|------|---------|
| `SiluAndMul.forward` | `chunk + silu + mul` → 1 个 kernel |
| `RMSNorm.rms_forward` | `pow + mean + rsqrt + mul` → 1 个 kernel |
| `RMSNorm.add_rms_forward` | `add + pow + mean + rsqrt + mul` → 1 个 kernel（残差 + Norm 融合） |
| `RotaryEmbedding.forward` | `index + chunk + float + mul + sub + cat` → 1 个 kernel |
| `Sampler.forward` | `div + softmax + exponential + clamp + div + argmax` → 1 个 kernel |

### 为什么 warmup 必须在 CUDA Graph 之前？

`torch.compile` 在首次调用时 JIT 编译，编译过程涉及动态图跟踪（tracing）和代码生成，不是确定性的 CUDA 操作序列，**无法被 CUDA Graph 捕获**。必须先 warmup 触发编译，之后的调用走编译后的固定 kernel 路径，才能被 CUDA Graph 录制。

---

## 第六部分：CUDA Graph 捕获与回放（核心）

### 问题

Decode 阶段每个序列只计算 1 个 token，计算量小但 kernel 数量不变（embedding → 28层 transformer → norm → lm_head）。CPU 端逐个 launch kernel 的开销占比很高（可能 > 50%）。

### 解决方案

CUDA Graph 将整个前向的 kernel 调用链录制为一个 graph，replay 时 GPU 自行执行所有 kernel，跳过 CPU 调度：

```
普通模式：
  CPU: launch_kernel_1 → wait → launch_kernel_2 → wait → ... → launch_kernel_N
  GPU: ===kernel_1===          ===kernel_2===          ... ===kernel_N===
       ↑ 空闲等 CPU               ↑ 空闲等 CPU

CUDA Graph 模式：
  CPU: graph.replay() → 完成
  GPU: ===kernel_1=====kernel_2=====...=====kernel_N===
       ↑ 无间断执行
```

### 捕获流程

```python
capture_cudagraph():
    # 1. 预分配固定地址的 tensor（CUDA Graph 要求地址不变）
    input_ids = torch.zeros(max_bs, ...)
    outputs = torch.zeros(max_bs, ...)

    # 2. 分桶：graph_bs = [1, 2, 4, 8, 16, 32, ..., max_bs]
    #    实际 bs=5 时使用 bs=8 的 graph（向上取整）

    # 3. 逆序捕获（从大到小）
    for bs in reversed(graph_bs):
        warmup: outputs[:bs] = model(input_ids[:bs], ...)   # 必须先跑一次
        with torch.cuda.graph(graph, pool):
            capture: outputs[:bs] = model(input_ids[:bs], ...)  # 录制
        graphs[bs] = graph
```

### 三个关键设计

**① 分桶策略**

不可能为每个 batch size 都捕获一个 graph（太多）。预设几个桶，实际推理时向上取：

```
graph_bs = [1, 2, 4, 8, 16, 32, 48, ..., 512]

实际 bs=3  → 使用 bs=4 的 graph（多出的位置填零，不影响结果）
实际 bs=10 → 使用 bs=16 的 graph
实际 bs=16 → 正好匹配
```

**② 逆序捕获**

从最大 bs 开始捕获，因为大 bs 需要更多显存。第一个捕获的 graph 创建 memory pool，后续小 bs 的 graph 复用同一个 pool：

```python
if self.graph_pool is None:
    self.graph_pool = graph.pool()   # 第一个（最大）graph 创建 pool
# 后续 graph 通过 torch.cuda.graph(graph, self.graph_pool) 复用
```

**③ 固定地址 tensor**

CUDA Graph 录制的是 kernel 参数中的**指针地址**，不是值。所以捕获和回放必须使用同一块显存。回放时只需更新 tensor 的值：

```python
# 回放时：写入新数据到同一 tensor（地址不变）
graph_vars["input_ids"][:bs] = actual_input_ids
graph_vars["slot_mapping"].fill_(-1)           # -1 = 无效 slot
graph_vars["slot_mapping"][:bs] = actual_slots
graph.replay()                                 # GPU 重放录制的操作
logits = graph_vars["outputs"][:bs]            # 从同一 tensor 读取结果
```

### 数值示例

```
假设 max_num_seqs=64, max_model_len=4096, block_size=256

graph_bs = [1, 2, 4, 8, 16, 32, 48, 64]
max_num_blocks = ceil(4096 / 256) = 16

预分配 tensor：
  input_ids:   [64]
  positions:   [64]
  slot_mapping:[64]
  context_lens:[64]
  block_tables:[64, 16]
  outputs:     [64, hidden_size]

捕获顺序：64 → 48 → 32 → 16 → 8 → 4 → 2 → 1

运行时：
  20 条序列 → bs=20 → 选 graph_bs=32（下一个 ≥ 20 的桶）
  graph_vars["input_ids"][:20] = 实际数据
  graph_vars["input_ids"][20:32] = 0（无影响）
  graph.replay()
  结果 = graph_vars["outputs"][:20]
```

### 为什么 Prefill 不用 CUDA Graph？

Prefill 的输入长度每次都不同（不同 prompt 长度组合），无法预捕获所有可能的形状。且 Prefill 是计算密集型（长序列矩阵乘），kernel launch 开销占比小，CUDA Graph 收益低。

---

## 第七部分：Tensor Parallel 通信

### 架构

```
rank 0 (调度 + 推理)          rank 1 (推理)          rank 2 (推理)
  │                              │                      │
  ├─ SharedMemory (1MB) ────────►├─ read_shm() ─────────├─ read_shm()
  │   pickle([method, args])     │   event.wait()       │   event.wait()
  │                              │                      │
  ├─ event[0].set() ────────────►│                      │
  ├─ event[1].set() ─────────────────────────────────►  │
  │                              │                      │
  └─ 本地执行 method(*args)      └─ 本地执行             └─ 本地执行
                                     ↓                      ↓
                              NCCL all-reduce ◄─────────────┘
```

### 通信协议

| 步骤 | rank 0 | worker |
|------|--------|--------|
| 1 | `write_shm("run", seqs, is_prefill)` | `event.wait()` 阻塞 |
| 2 | `event.set()` 通知所有 worker | 收到信号，`read_shm()` 读取 |
| 3 | 本地执行 `run(seqs, is_prefill)` | 本地执行 `run(seqs, is_prefill)` |
| 4 | 模型前向中 all-reduce 同步 | 模型前向中 all-reduce 同步 |
| 5 | 执行采样，返回 token_ids | 不采样（只有 rank 0 采样） |

### 为什么用 SharedMemory 而不是 NCCL broadcast？

NCCL 为 GPU tensor 优化，传输少量 Python 对象（方法名、序列元数据）效率不高。SharedMemory + pickle 更轻量，且避免了 GPU 参与控制流通信。

---

## 核心问题回答

**Q: CUDA Graph 加速了什么？**

CUDA Graph 消除了 CPU 端逐个 launch kernel 的开销。Decode 阶段每个序列只算 1 个 token，单次前向涉及上百个小 kernel（embedding、每层的 attention + FFN、最终 norm + lm_head），每个 kernel 的 CPU launch 耗时 ~5-10μs，累计可达 1ms+，而 GPU 实际计算可能只需 1-2ms。CUDA Graph 将所有 kernel 打包为一次调用，CPU 端开销降为 ~10μs。

**Q: `torch.compile` 和 CUDA Graph 的关系？**

二者互补但层次不同：
- `torch.compile` 在**编译期**融合算子，减少 kernel 数量（如 5 个小算子 → 1 个融合 kernel）
- CUDA Graph 在**运行期**消除 kernel launch 开销（不管几个 kernel，一次 replay 全部执行）

两者叠加效果：`torch.compile` 先减少 kernel 数量，CUDA Graph 再消除剩余 kernel 的 launch 开销。

**Q: 为什么 CUDA Graph 要求固定地址 tensor？**

CUDA Graph 录制的是 kernel 的参数（包括 tensor 的 GPU 内存指针）。如果回放时 tensor 地址变了，kernel 会读写错误的内存。所以必须在捕获前预分配 tensor，回放时只改值不改地址。

**Q: 全局 Context 模式有什么代价？**

牺牲了一定的代码可追踪性——`Attention.forward()` 的输入不完全来自参数，还依赖全局状态。但对于推理引擎来说，这是合理的权衡：避免了对所有中间模块的 API 改动，代码更简洁。vLLM 也采用了类似的设计。

**Q: Tensor Parallel 中 worker 怎么知道该做什么？**

Rank 0 通过 SharedMemory 广播 `[method_name, *args]`（如 `["run", seqs, True]`），worker 反序列化后调用 `self.run(seqs, True)`。模型前向中的 all-reduce 通过 NCCL 自动同步各卡的中间结果。这样 worker 不需要自己的调度器，只需执行 rank 0 的指令。
