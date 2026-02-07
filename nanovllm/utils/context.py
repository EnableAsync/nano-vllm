# 全局注意力上下文 (Attention Context)
# 将 attention 所需的元数据（序列长度、slot 映射、block table 等）存储在全局变量中，
# 避免在 model → layers → attention 的 forward 调用链中层层传递这些参数。
# 设计思路：set_context() 在 ModelRunner 准备数据时调用，
#           get_context() 在 Attention.forward() 中读取，
#           reset_context() 在每次推理完成后清空。

from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False                          # 当前是 prefill 还是 decode 阶段
    cu_seqlens_q: torch.Tensor | None = None          # query 累积序列长度（flash attention varlen 接口需要）
    cu_seqlens_k: torch.Tensor | None = None          # key 累积序列长度
    max_seqlen_q: int = 0                             # 当前 batch 中最长的 query 序列长度
    max_seqlen_k: int = 0                             # 当前 batch 中最长的 key 序列长度
    slot_mapping: torch.Tensor | None = None          # 每个 token 对应的 KV cache 物理 slot 位置
    context_lens: torch.Tensor | None = None          # decode 时每个序列的上下文长度（含历史 token）
    block_tables: torch.Tensor | None = None          # decode 时每个序列的 block table（物理 block id 列表）

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
