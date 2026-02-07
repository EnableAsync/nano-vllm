# 模型前向执行器 (Model Runner)
# 核心职责：
# 1. 初始化模型、分配 KV cache
# 2. 准备 prefill / decode 的输入数据（input_ids, positions, attention 元数据）
# 3. 捕获 CUDA Graph 加速 decode 阶段的前向计算
# 4. Tensor Parallel 多卡通信（共享内存 + Event 同步）

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event          # rank=0 持有 list[Event]（通知各 worker），worker 持有单个 Event

        # --- 初始化流程 ---
        # ① 初始化 NCCL 进程组（多卡 all-reduce 通信）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # ② 构建模型并加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # ③ warmup：用最大输入跑一次前向，触发 torch.compile 编译 + 测量峰值显存
        self.warmup_model()
        # ④ 根据剩余显存分配 KV cache
        self.allocate_kv_cache()
        # ⑤ 捕获 CUDA Graph（enforce_eager=True 时跳过）
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # --- Tensor Parallel 通信初始化 ---
        # 使用共享内存（SharedMemory）在进程间传递方法名和参数，
        # 用 Event 做同步信号（比 NCCL 广播更轻量，适合传少量控制信息）
        if self.world_size > 1:
            if rank == 0:
                # rank 0 创建共享内存（1MB），用于向 worker 广播调度指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()      # 等待所有 worker 就绪
            else:
                dist.barrier()
                # worker 打开已有的共享内存
                self.shm = SharedMemory(name="nanovllm")
                self.loop()         # worker 进入消息循环，等待 rank 0 指令

    def exit(self):
        """清理资源：关闭共享内存、释放 CUDA Graph、销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()   # 只有创建者 unlink
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker 消息循环：反复从共享内存读取指令并执行，直到收到 exit"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Worker 从共享内存读取一条指令（阻塞等待 Event 信号）"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()           # 阻塞直到 rank 0 写入完成
        # 前 4 字节存数据长度，后续为 pickle 序列化的 [method_name, *args]
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()          # 重置信号，准备下一次
        return method_name, args

    def write_shm(self, method_name, *args):
        """Rank 0 向共享内存写入指令并通知所有 worker"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")    # 写入数据长度
        self.shm.buf[4:n+4] = data                      # 写入序列化数据
        for event in self.event:                         # 通知每个 worker
            event.set()

    def call(self, method_name, *args):
        """调用指定方法。rank 0 会先通过共享内存广播给 worker，再本地执行"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """预热模型：用最大输入跑一次前向。
        目的：① 触发 torch.compile 编译（首次调用会 JIT 编译）
              ② 测量峰值显存，供 allocate_kv_cache 计算可用空间"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造最大规模输入：num_seqs 条 max_model_len 长度的序列
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存计算并分配 KV cache。
        公式：num_blocks = (总显存 × 利用率 - 已用 - 峰值 + 当前) / 每 block 字节数
        峰值是 warmup 时记录的，当前是 warmup 后的（差值 = 临时分配已释放的部分）"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size   # TP 分片后每卡的 KV head 数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每个 block 的字节数 = 2(K+V) × 层数 × block_size × num_kv_heads × head_dim × dtype 字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一次性分配所有 KV cache：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将 KV cache 的每层切片绑定到对应 Attention 模块
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """将各序列的 block_table 对齐为相同长度，转为 GPU tensor"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备 prefill 阶段的输入。
        Prefill 处理多条变长序列，使用 flash_attn_varlen 接口，
        需要 cu_seqlens（累积序列长度）来区分 batch 中的各序列。"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]          # 累积 query 长度前缀和，varlen 接口需要
        cu_seqlens_k = [0]          # 累积 key 长度前缀和
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []           # 每个新 token 写入 KV cache 的物理位置
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # prefix cache 命中时只需处理未缓存的 token
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens   # 实际计算的 query 长度
            seqlen_k = seqlen                            # key 长度始终是完整序列（含 cache）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 时没有 block_table
                continue
            # 计算 slot_mapping：将新 token 映射到 KV cache 的物理 slot
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 有 prefix cache 命中时 k > q，需要 block_tables 读取缓存的 KV
            block_tables = self.prepare_block_tables(seqs)
        # pin_memory + non_blocking：CPU→GPU 异步传输，与计算重叠
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 decode 阶段的输入。
        Decode 阶段每个序列只处理 1 个 token（最新生成的），
        但需要 attention 到全部历史 KV cache（通过 block_tables 定位）。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)            # 只取最新 token
            positions.append(len(seq) - 1)              # 位置 = 序列总长 - 1
            context_lens.append(len(seq))               # attention 需要的上下文长度
            # 新 token 写入最后一个 block 的下一个 slot
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集采样温度参数（仅 rank 0 需要，因为只有 rank 0 做采样）"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行模型前向。Decode 阶段优先使用 CUDA Graph 回放（跳过 Python 开销和 kernel launch）。
        Prefill 不用 CUDA Graph 因为输入长度不固定，无法提前捕获。"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Prefill / eager / 超大 batch：直接前向
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode：CUDA Graph 回放
            bs = input_ids.size(0)
            context = get_context()
            # 向上找到最近的预捕获 batch size（分桶策略：1,2,4,8,16,32,...）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 将实际数据写入 graph 捕获时使用的 tensor（地址不变，值更新）
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)         # -1 表示无效 slot，多余位置不写入 KV cache
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()           # 清零后写入实际值
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()                               # 回放捕获的 CUDA 操作序列
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """完整推理流程：数据准备 → 模型前向 → 采样"""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获 CUDA Graph，用于加速 decode 阶段的前向计算。

        CUDA Graph 原理：将一系列 CUDA kernel 调用录制为一个 graph，
        之后 replay 时跳过 CPU 端的 kernel launch 开销，直接在 GPU 上重放。

        分桶策略：预捕获 [1,2,4,8,16,32,...,max_bs] 几种 batch size 的 graph，
        实际推理时向上取最近的桶。浪费少量计算但避免为每个 bs 捕获一个 graph。

        逆序捕获（从大到小）：大 bs 的 graph 分配更多显存，先捕获可确保小 bs 复用同一 pool。
        pool 共享：所有 graph 共用同一个 memory pool（graph_pool），避免重复分配。"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配最大尺寸的输入 tensor（CUDA Graph 要求捕获和回放使用相同地址的 tensor）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 分桶：小 bs 细粒度（1,2,4,8），大 bs 按 16 步进
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 逆序捕获：从最大 bs 开始，确保 pool 分配足够大
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup（必须，让 PyTorch 分配中间 buffer）
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # 捕获：录制所有 CUDA 操作
            if self.graph_pool is None:
                self.graph_pool = graph.pool()   # 第一个 graph（最大 bs）创建 pool，后续复用
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存所有 graph 共享的输入/输出 tensor 引用，run_model 通过修改这些 tensor 的值来传入新数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
