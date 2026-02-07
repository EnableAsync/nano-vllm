# RMSNorm (Root Mean Square Layer Normalization)
# 相比 LayerNorm，RMSNorm 去掉了均值中心化，只做缩放，计算更快
# 公式：RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
# LLaMA / Qwen 等现代 LLM 均采用 RMSNorm

import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # 转 float32 计算，避免 bf16/fp16 精度损失
        x = x.float()
        # 计算 RMS：每个位置各维度的平方均值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1/sqrt，原地乘完成归一化
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原精度，乘以可学习 weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 融合残差加法 + RMSNorm，减少一次 kernel launch 和显存读写
        # Transformer 中每个子层输出都要 residual + norm，融合后效率更高
        orig_dtype = x.dtype
        # 先做残差加法：x = x + residual（float32 精度）
        x = x.float().add_(residual.float())
        # 保存加法结果作为下一层的 residual（Pre-Norm 架构）
        residual = x.to(orig_dtype)
        # 再做 RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        # 返回归一化结果和更新后的 residual
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层：没有残差，直接做 RMSNorm
            return self.rms_forward(x)
        else:
            # 后续层：融合 residual + RMSNorm
            return self.add_rms_forward(x, residual)
