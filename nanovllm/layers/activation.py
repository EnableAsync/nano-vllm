# SwiGLU 激活函数 (Swish-Gated Linear Unit)
# LLaMA / Qwen 等模型用 SwiGLU 替代传统 ReLU / GELU
# 核心思想：将输入沿最后一维一分为二，一半做门控(gate)，一半做值(value)
# 公式：SwiGLU(x, y) = SiLU(x) * y，其中 SiLU(x) = x * sigmoid(x)
# 相比 ReLU，SwiGLU 在同等参数量下表现更好（PaLM 论文验证）

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, 2 * hidden_size]
        # chunk(2, -1) 沿最后一维切成两半：gate 和 value
        x, y = x.chunk(2, -1)
        # SiLU(gate) * value — SiLU 即 swish 激活，自带平滑门控效果
        return F.silu(x) * y
