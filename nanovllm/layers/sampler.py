# 采样器：将 logits 转换为 token id
# 使用 Gumbel-max 技巧替代传统的 multinomial 采样
# 传统方式：softmax → 构建 CDF → 随机采样（慢，需要排序或累加）
# Gumbel-max：argmax(log(softmax(x)) - log(-log(U))) 等价于按概率采样
# 简化后等价于：argmax(softmax(x) / Exp(1))，即本文实现
# 优势：全程可并行，适合 GPU；且可用 torch.compile 优化

import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperature 缩放：T→0 趋近 greedy，T→∞ 趋近均匀分布
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max 采样核心：
        # 1. exponential_(1) 生成 Exp(1) 随机数（等价于 -log(U)，U~Uniform）
        # 2. clamp_min_ 防止除零
        # 3. probs / Exp(1) 后取 argmax 等价于按 probs 概率采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
