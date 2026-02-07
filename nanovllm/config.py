import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str  # 模型路径（本地目录）
    max_num_batched_tokens: int = 16384  # 单次调度最大 token 总数（prefill 预算）
    max_num_seqs: int = 512  # 单 batch 最大序列数
    max_model_len: int = 4096  # 序列最大长度（包含 prompt + 生成）
    gpu_memory_utilization: float = 0.9  # GPU 显存使用比例，用于 KV cache 分配
    tensor_parallel_size: int = 1  # 张量并行数（GPU 数量）
    enforce_eager: bool = False  # 为 True 时禁用 CUDA Graph，仅用 eager 模式
    hf_config: AutoConfig | None = None  # 从 HuggingFace 加载的模型配置，__post_init__ 中自动填充
    eos: int = -1  # 结束符 token id，由 engine 从 tokenizer 获取后回填
    kvcache_block_size: int = 256  # KV cache 块大小（token 数）
    num_kvcache_blocks: int = -1  # KV cache 块数量，-1 表示自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)  # 模型路径必须是有效目录
        assert self.kvcache_block_size % 256 == 0  # 块大小必须是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8  # 最多支持 8 卡并行
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 加载 HF 模型配置（获取 hidden_size 等参数）
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # 不超过模型支持的最大位置编码
        assert self.max_num_batched_tokens >= self.max_model_len  # prefill 预算必须能容纳单条最长序列
