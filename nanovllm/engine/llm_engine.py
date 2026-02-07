import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """推理引擎主类。组装 ModelRunner（模型执行）+ Scheduler（调度）+ Tokenizer。
    主循环：add_request → while step() → 收集结果。"""

    def __init__(self, model, **kwargs):
        # 从 kwargs 中筛选出属于 Config 的字段，忽略无关参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 启动张量并行的子进程（rank 1..N-1），rank 0 在主进程运行
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")  # 用 spawn 而非 fork，避免 CUDA 上下文问题
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 用于主进程与子进程之间的同步信号
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 的 ModelRunner 在主进程中创建，同时持有所有子进程的 event 用于同步
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 将 eos id 回填到 config，供 scheduler 判断结束
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # 注册退出钩子，确保子进程正常退出

    def exit(self):
        self.model_runner.call("exit")  # 通知所有 rank 退出
        del self.model_runner
        for p in self.ps:
            p.join()  # 等待子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  # 文本 prompt 先 tokenize
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 调度：从等待队列中选出本轮要处理的序列，并判断是 prefill 还是 decode
        seqs, is_prefill = self.scheduler.schedule()
        # 执行：将序列送入模型，获取每个序列的下一个 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：将生成的 token 追加到序列，并检查是否结束（eos / max_tokens）
        self.scheduler.postprocess(seqs, token_ids)
        # 收集已完成的序列输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 统计 token 数：prefill 时为正（总输入 token 数），decode 时为负（取负序列数，即每个序列生成 1 个 token）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 如果传入单个 SamplingParams，复制为与 prompts 等长的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 将所有请求加入调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 主循环：反复调用 step() 直到所有序列生成完毕
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                # num_tokens > 0 表示 prefill 阶段，< 0 表示 decode 阶段
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids  # 按 seq_id 收集结果
                if use_tqdm:
                    pbar.update(1)
        # 按 seq_id 排序输出，保证与输入 prompts 顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
