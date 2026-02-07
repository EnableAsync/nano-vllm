from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """Continuous Batching 调度器。
    管理 waiting（等待 prefill）和 running（正在 decode）两个队列，
    每次 schedule() 决定执行 prefill 还是 decode，并处理显存不足时的抢占。"""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                # 单 batch 最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单次调度最大 token 数（prefill 预算）
        self.eos = config.eos                                  # 结束符 token id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()   # 等待 prefill 的序列队列
        self.running: deque[Sequence] = deque()   # 正在 decode 的序列队列

    def is_finished(self):
        return not self.waiting and not self.running  # 两个队列都空时全部完成

    def add(self, seq: Sequence):
        self.waiting.append(seq)  # 新请求进入等待队列

    def schedule(self) -> tuple[list[Sequence], bool]:
        """核心调度逻辑：prefill 优先策略。
        1. 优先尝试从 waiting 中取序列做 prefill
        2. 若无 waiting 序列，对 running 中的序列做 decode
        返回 (被调度的序列列表, 是否是 prefill)"""

        # === Prefill 阶段 ===
        # 尝试从 waiting 队列取出尽可能多的序列做 prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查两个约束：token 预算 和 KV cache 容量
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)           # 分配 KV cache block（含 prefix cache 尝试）
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 实际需计算的 token 数（减去 cache 命中）
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True  # 有 prefill 任务则直接返回

        # === Decode 阶段 ===
        # 对 running 队列中的序列做 decode（每个序列生成 1 个 token）
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否有足够的 block 追加新 token
            while not self.block_manager.can_append(seq):
                # KV cache 不足：抢占最后加入的序列以释放 block
                if self.running:
                    self.preempt(self.running.pop())   # 抢占其他序列
                else:
                    self.preempt(seq)                  # 无其他序列可抢占，抢占自身
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)     # 更新 block 状态（可能分配新 block）
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 将调度的序列放回 running 队列头部，保持原有顺序
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占：释放序列的 KV cache，将其退回 waiting 队列头部（下次优先 prefill）"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """后处理：将生成的 token 追加到序列，检查是否完成。
        完成条件：遇到 eos（且未设置 ignore_eos）或 达到 max_tokens。"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)     # 释放已完成序列的 KV cache
                self.running.remove(seq)               # 从 running 队列移除
