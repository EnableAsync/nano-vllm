from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 采样温度，越高随机性越大，越低越确定
    max_tokens: int = 64  # 单次生成的最大 token 数
    ignore_eos: bool = False  # 为 True 时忽略结束符，强制生成到 max_tokens

    def __post_init__(self):
        # temperature 必须 > 0，不支持 greedy（temperature=0）因为本项目只实现了随机采样
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
