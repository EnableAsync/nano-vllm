# 阶段 1：入口与数据流

## 目标

理解用户如何使用 nano-vllm，数据如何从入口流向推理引擎。

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 2 | 包入口，暴露 `LLM` 和 `SamplingParams` |
| `llm.py` | 5 | `LLM` 类，`LLMEngine` 的别名 |
| `sampling_params.py` | 11 | 采样参数定义 |
| `config.py` | 26 | 模型与引擎配置 |
| `engine/llm_engine.py` | 93 | 推理引擎，核心调度循环 |

## 逐文件讲解

### `__init__.py` — 包入口

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

只暴露两个类，API 极简。对标 vLLM 的 `from vllm import LLM, SamplingParams`。

### `llm.py` — 用户 API

```python
class LLM(LLMEngine):
    pass
```

纯别名，保持与 vLLM 一致的用户接口，实际逻辑全在 `LLMEngine`。

### `sampling_params.py` — 采样参数

3 个参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `temperature` | 1.0 | 控制随机性，**禁止 greedy（≤1e-10）** |
| `max_tokens` | 64 | 最大生成 token 数 |
| `ignore_eos` | False | 忽略 EOS，benchmark 时强制生成到 max_tokens |

禁止 greedy sampling 是简化实现的取舍（避免处理 argmax 分支）。

### `config.py` — 模型配置

```python
@dataclass
class Config:
    model: str                          # 模型路径
    max_num_batched_tokens: int = 16384 # 一个 batch 最多处理的 token 数
    max_num_seqs: int = 512             # 同时处理的最大序列数
    max_model_len: int = 4096           # 最大序列长度
    gpu_memory_utilization: float = 0.9 # GPU 显存利用率
    tensor_parallel_size: int = 1       # 张量并行数
    enforce_eager: bool = False         # 禁用 CUDA graph/torch.compile
    hf_config: AutoConfig | None = None # HuggingFace 模型配置（自动加载）
    eos: int = -1                       # EOS token id（后续由 tokenizer 填充）
    kvcache_block_size: int = 256       # KV cache 块大小
    num_kvcache_blocks: int = -1        # KV cache 块数量（后续自动计算）
```

`__post_init__` 关键逻辑：
- 校验模型路径存在
- 从 HuggingFace 加载模型配置（层数、head 数等）
- `max_model_len` 取配置值与模型 `max_position_embeddings` 的较小值

### `engine/llm_engine.py` — 核心引擎

这是整个系统的心脏，分 3 部分理解。

#### `__init__`：初始化

```
Config → 启动 tensor parallel 子进程 → 创建 ModelRunner(rank=0)
       → 加载 tokenizer → 创建 Scheduler
```

- `tensor_parallel_size > 1` 时，为 rank 1~N 各启动一个 `mp.Process`
- rank 0 的 `ModelRunner` 在主进程运行
- 用 `mp.Event` 在进程间同步

#### `generate`：主循环

```python
# 1. 添加请求
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)   # tokenize → Sequence → scheduler.add

# 2. 循环推理直到全部完成
outputs = {}
while not self.is_finished():
    output, num_tokens = self.step()
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids

# 3. 解码输出
outputs = [tokenizer.decode(token_ids) for ...]
```

`num_tokens` 正负区分 prefill/decode：正数 = prefill token 总数，负数 = decode 序列数（每序列生成 1 token）。

#### `step`：单步推理（最关键的 3 行）

```python
seqs, is_prefill = self.scheduler.schedule()                  # 调度
token_ids = self.model_runner.call("run", seqs, is_prefill)   # 执行
self.scheduler.postprocess(seqs, token_ids)                   # 更新
```

这就是 vLLM 架构的精髓：**Schedule → Execute → Update**。

## 调用链路总结

```
用户: LLM(model).generate(prompts, params)
  │
  ├─ __init__: Config → ModelRunner × N → Scheduler
  │
  └─ generate:
       ├─ add_request: tokenize → Sequence → scheduler.add
       └─ while loop:
            └─ step:
                 ├─ scheduler.schedule()      → 选序列，决定 prefill/decode
                 ├─ model_runner.call("run")  → 模型前向推理
                 └─ scheduler.postprocess()   → 更新序列状态，标记完成
```

## 核心问题回答

**Q: LLM 类如何将用户请求转化为推理调用？**

1. 用户调用 `llm.generate(prompts, sampling_params)`
2. 每个 prompt 被 tokenize 后封装为 `Sequence`，加入 `Scheduler`
3. `generate` 进入 while 循环，反复调用 `step()`
4. 每次 `step()` 中，Scheduler 选出一批序列，ModelRunner 执行前向推理，再由 Scheduler 更新状态
5. 序列完成后收集 token ids，最终 decode 为文本返回
