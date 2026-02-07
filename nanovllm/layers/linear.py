# 张量并行线性层 (Tensor Parallel Linear Layers)
#
# ━━━ 张量并行核心思想 ━━━
# 单个大矩阵乘法 Y = X @ W 拆到多个 GPU 上并行计算
# 两种拆法：
#   Column Parallel: 按列切 W → 每个 GPU 算输出的一部分列 → 拼接(或直接用)
#   Row Parallel:    按行切 W → 每个 GPU 算部分结果 → all-reduce 求和
#
# 在 Transformer MLP 中的典型搭配（Megatron-LM 方案）：
#   gate_up_proj: ColumnParallel（输出按列切，无需通信）
#   down_proj:    RowParallel（输入按行切，输出 all-reduce 一次）
# 整个 MLP 只需 1 次 all-reduce，通信量最小
#
# ━━━ weight_loader 机制 ━━━
# 加载预训练权重时，每个 GPU 只取自己那份 shard
# tp_dim=0: 按行切（output 维度）→ ColumnParallel
# tp_dim=1: 按列切（input 维度） → RowParallel

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ━━━ 基类：定义通用的权重创建和 TP 信息 ━━━
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # tp_dim: 权重加载时沿哪个维度切分（0=行/output, 1=列/input）
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 注意：这里 output_size 已经是切分后的大小（子类传入时已 divide）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # weight_loader: 告诉模型加载器如何从完整权重中提取本 GPU 的 shard
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ━━━ 不切分，每个 GPU 持有完整副本（用于小矩阵） ━━━
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 直接复制完整权重，不切分
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ━━━ 列并行：按 output 维度切分 ━━━
# W shape: [output, input] → 每个 GPU 持有 [output/tp, input]
# 前向：Y_i = X @ W_i^T → 每个 GPU 独立算出输出的一部分列，无需通信
# 典型用途：MLP 的 gate_proj / up_proj、Attention 的 QKV 投影
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # output_size / tp_size：每个 GPU 只负责一部分输出维度
        # tp_dim=0：加载权重时沿第 0 维(output)切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 从完整权重中取本 GPU 负责的那段 output 维度
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接矩阵乘，无需通信（后续 RowParallel 会做 all-reduce）
        return F.linear(x, self.weight, self.bias)


# ━━━ 合并列并行：gate_proj 和 up_proj 合并为一个大矩阵 ━━━
# MLP 中 gate 和 up 的输入相同，合并后只需一次 GEMM，效率更高
# 权重 shape: [gate_size + up_size, input] → 切分后每个 GPU 持有各自的 shard
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # output_sizes 例如 [gate_size, up_size]
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # loaded_shard_id: 0=gate, 1=up（或其他子矩阵的编号）
        # shard_offset: 该子矩阵在合并权重中的起始位置（已切分后）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 先定位到合并权重中该子矩阵的区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 再从完整权重中取本 GPU 的那份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# ━━━ QKV 并行：Q/K/V 三个投影合并为一个矩阵 ━━━
# 类似 MergedColumn，但 Q/K/V 大小可能不同（GQA 中 K/V head 数 < Q head 数）
# 权重 shape: [(num_q_heads + 2*num_kv_heads) * head_size, hidden_size]
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每个 GPU 分到的 head 数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出 = (Q heads + K heads + V heads) * head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        # Q/K/V 在合并权重中的布局：[Q | K | V]
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            # K 紧跟 Q 之后
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            # V 紧跟 K 之后
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 定位本 GPU 在合并权重中 Q/K/V 对应的区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取本 GPU 的 shard
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# ━━━ 行并行：按 input 维度切分 ━━━
# W shape: [output, input] → 每个 GPU 持有 [output, input/tp]
# 前向：Y_i = X_i @ W_i^T → 每个 GPU 算出部分结果 → all-reduce 求和得完整输出
# 典型用途：MLP 的 down_proj、Attention 的 o_proj
# 与 ColumnParallel 配对：Column 的输出直接作为 Row 的输入，中间无需通信
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # input_size / tp_size：每个 GPU 只接收一部分输入
        # tp_dim=1：加载权重时沿第 1 维(input)切分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 从完整权重中取本 GPU 负责的那段 input 维度
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只有 rank 0 加 bias，避免 all-reduce 后 bias 被重复加
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # all-reduce：所有 GPU 的部分结果求和 → 完整输出
            # 这是整个 MLP/Attention 中唯一的通信点
            dist.all_reduce(y)
        return y
