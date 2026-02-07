from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()   # 等待调度（尚未分配 KV cache）
    RUNNING = auto()   # 正在推理（已分配 KV cache）
    FINISHED = auto()  # 已完成（遇到 eos 或达到 max_tokens）


class Sequence:
    """单条推理序列，贯穿请求的整个生命周期。
    存储 token ids、block table（KV cache 映射）和采样参数。"""

    block_size = 256            # KV cache 块大小（token 数），与 BlockManager 一致
    counter = count()           # 全局自增 ID 生成器，保证 seq_id 唯一

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)        # 唯一序列 ID
        self.status = SequenceStatus.WAITING         # 初始状态：等待调度
        self.token_ids = copy(token_ids)             # 完整 token 列表（prompt + 已生成）
        self.last_token = token_ids[-1]              # 最新 token，decode 时只需发送此 token
        self.num_tokens = len(self.token_ids)        # 当前总 token 数
        self.num_prompt_tokens = len(token_ids)      # prompt 长度（固定不变）
        self.num_cached_tokens = 0                   # prefix cache 命中的 token 数
        self.block_table = []                        # 物理 block id 列表，映射到 KV cache
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens  # 支持 len(seq) 获取当前总 token 数

    def __getitem__(self, key):
        return self.token_ids[key]  # 支持 seq[i] 或 seq[a:b] 索引

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens  # 已生成的 token 数

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]  # 原始 prompt 部分

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]  # 生成部分

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size  # prefix cache 命中的完整 block 数

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size  # 向上取整的总 block 数

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size  # 最后一个 block 中的 token 数

    def block(self, i):
        """返回第 i 个 block 对应的 token ids 切片"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个生成的 token（decode 阶段每步调用一次）"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化：用于跨进程传输（pickle）。
        decode 阶段只传 last_token 而非完整列表，减少通信开销。"""
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """反序列化：worker 进程只需 block_table 等元数据即可执行推理"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]  # prefill 阶段：需要完整 token_ids
        else:
            self.last_token = state[-1]  # decode 阶段：只需 last_token
