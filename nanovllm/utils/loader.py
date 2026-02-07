# 权重加载器 (Weight Loader)
# 从 safetensors 文件加载模型权重，处理两种情况：
# 1. 普通权重：直接 copy 到对应参数
# 2. 合并权重（packed_modules_mapping）：HF checkpoint 中 Q/K/V 是分开的，
#    但推理时合并为 qkv_proj 做一次 GEMM，需要按 shard_id 写入合并参数的对应切片

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认加载：直接复制权重到参数（无分片逻辑）"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 获取模型的权重名映射（如 q_proj → qkv_proj），无则为空
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历模型目录下所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否命中 packed_modules_mapping（需要分片加载）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 命中：k="q_proj" → v="qkv_proj", shard_id="q"
                        # 将权重名中的 k 替换为 v，找到合并后的参数
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 调用参数上的 weight_loader（由 QKVParallelLinear 等设置），
                        # 将权重写入合并参数的对应 shard 切片
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 未命中映射：直接加载到同名参数
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
