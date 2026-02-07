import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# Triton kernel：将当前 token 的 K/V 写入 paged KV cache 的对应 slot
# 每个 program 处理一个 token，通过 slot_mapping 找到该 token 在 cache 中的物理位置
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,           # num_heads * head_dim，一个 token 的 KV 总维度
):
    idx = tl.program_id(0)     # 第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)  # 该 token 在 cache 中的物理 slot 编号
    if slot == -1: return      # -1 表示 padding token，跳过
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """将 key/value 写入 paged KV cache。
    Args:
        key/value: [N, num_heads, head_dim] — N 为所有序列 token 总数（prefill）或 batch_size（decode）
        k_cache/v_cache: [num_blocks * block_size, num_heads * head_dim] — 物理 cache 池
        slot_mapping: [N] — 每个 token 映射到 cache 中的物理 slot 编号
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1        # 最内维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D    # cache 按 (slot, D) 排列
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """通用注意力层，封装 Prefill/Decode 两条路径和 KV cache 管理。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # KV cache 初始化为空 tensor，实际显存由 ModelRunner 在引擎启动时统一分配
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 将当前 token 的 K/V 写入 paged cache（首次推理 cache 为空时跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # --- Prefill 路径 ---
            # 使用 flash_attn_varlen_func：支持变长序列（多条 prompt 拼接）
            if context.block_tables is not None:    # prefix cache：K/V 从 cache 读取而非当前计算
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # --- Decode 路径 ---
            # 使用 flash_attn_with_kvcache：每个序列仅 1 个新 query token
            # q.unsqueeze(1) 将 [batch, heads, dim] → [batch, 1, heads, dim]（seqlen=1）
            # cache_seqlens 告诉 flash_attn 每个序列的历史长度
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
