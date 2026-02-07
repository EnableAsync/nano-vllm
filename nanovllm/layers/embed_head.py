# 词表并行 Embedding + LM Head
#
# ━━━ 词表并行核心思想 ━━━
# 词表很大（如 151936），embedding 矩阵占大量显存
# 解法：将词表按行切分到多个 GPU，每个 GPU 只存 vocab_size/tp 行
#
# ━━━ Embedding 前向流程 ━━━
# 1. 每个 GPU 检查输入 token_id 是否在自己负责的范围内
# 2. 在范围内的正常查表，不在范围内的输出零向量
# 3. all-reduce 求和 → 拼出完整 embedding
#
# ━━━ LM Head 前向流程 ━━━
# 与 Embedding 共享权重（tied weights），但计算方向相反：
# Embedding: token_id → vector (查表)
# LM Head:   vector → logits (矩阵乘)
# 每个 GPU 只能算出自己那部分词表的 logits
# 用 gather 收集到 rank 0 拼成完整 logits（只有 rank 0 需要采样）

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# ━━━ 词表并行 Embedding ━━━
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每个 GPU 负责的词表片段大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本 GPU 负责的词表范围 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 只存本 GPU 那部分 embedding，shape: [vocab/tp, dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 从完整 embedding 矩阵中取本 GPU 负责的那段行
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # mask: 标记哪些 token 属于本 GPU 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将 id 转为本地索引（减去起始偏移），不属于的置 0（查表后会被 mask 掉）
            x = mask * (x - self.vocab_start_idx)
        # 本地查表
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 不属于本 GPU 的位置置零
            y = mask.unsqueeze(1) * y
            # all-reduce：所有 GPU 的结果求和 → 完整 embedding
            dist.all_reduce(y)
        return y


# ━━━ 并行 LM Head：复用 Embedding 权重做最终投影 ━━━
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        # 继承 VocabParallelEmbedding，复用权重和 weight_loader
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # prefill 阶段：只需要每个序列最后一个 token 的 logits
            # cu_seqlens_q[1:]-1 取每个序列的末尾位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 用 embedding 权重做线性投影：logits = x @ weight^T
        # 每个 GPU 只能算出自己负责的那部分词表的 logits
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # gather 到 rank 0：只有 rank 0 拼接完整 logits 做采样
            # 不用 all-gather 因为只有 rank 0 需要完整结果，节省通信量
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # rank 0 拼接所有分片的 logits → [batch, full_vocab_size]
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
