# RoPE 旋转位置编码 (Rotary Position Embedding)
# 核心思想：用旋转矩阵对 Q/K 向量编码位置信息
# 优势：1) 天然编码相对位置（内积只依赖相对距离）
#       2) 无需额外参数（纯数学变换）
#       3) 可外推到训练时未见的序列长度
#
# 数学原理：
#   将 head_dim 维向量视为 head_dim/2 个二维子空间
#   每个子空间施加旋转：[cos θ, -sin θ; sin θ, cos θ] @ [x1, x2]
#   θ_i = pos / base^(2i/d)，不同维度旋转频率不同
#   低维转得快（捕捉局部位置），高维转得慢（捕捉远程位置）

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 将向量拆成前半 x1 和后半 x2（等价于拆成 head_dim/2 个二维对）
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 旋转公式：y1 = x1*cos - x2*sin
    y1 = x1 * cos - x2 * sin
    #           y2 = x2*cos + x1*sin
    y2 = x2 * cos + x1 * sin
    # 拼回原始维度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 频率基底：inv_freq_i = 1 / base^(2i/d)
        # 维度越高(i越大)，频率越低，旋转越慢
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 位置序列 [0, 1, 2, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[pos, i] = pos * inv_freq_i → 每个位置每个维度的旋转角度
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 缓存 [max_pos, 1, head_dim]，unsqueeze(1) 是为 num_heads 维度广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按 token 位置索引取出对应的 cos/sin
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 对 Q 和 K 施加相同旋转（V 不需要位置编码）
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    # 工厂函数，lru_cache 保证全局只创建一次
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
