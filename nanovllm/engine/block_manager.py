from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """物理 KV cache 块。每个 block 存储 block_size 个 token 的 K/V。"""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0       # 引用计数：有多少序列共享此 block（用于 prefix caching）
        self.hash = -1           # 内容哈希，-1 表示未填满（不可被 prefix cache 复用）
        self.token_ids = []      # 此 block 对应的 token ids（用于验证 cache 命中）

    def update(self, hash: int, token_ids: list[int]):
        """block 填满后更新哈希和内容，使其可被 prefix cache 查找"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重新分配时重置状态：引用计数置 1，清除哈希和内容"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Paged KV Cache 管理器。
    核心思想：将 KV cache 切分为固定大小的 block，按需分配/释放，
    并通过内容哈希实现 prefix caching（相同 prompt 前缀共享 block）。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有物理 block
        self.hash_to_block_id: dict[int, int] = dict()   # 内容哈希 → block id，prefix cache 查找表
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲 block 队列
        self.used_block_ids: set[int] = set()  # 已占用 block 集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算 block 内容哈希。hash = xxh64(prefix_hash + token_ids)。
        prefix 参数是前一个 block 的哈希，形成哈希链：
        block0_hash = hash(token_ids_0)
        block1_hash = hash(block0_hash + token_ids_1)
        这样相同前缀的序列会产生相同的哈希链，实现 prefix caching。"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # 将前缀哈希编码为 8 字节
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个物理 block：从 free 移到 used，引用计数置 1"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放一个物理 block：从 used 移到 free"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够空闲 block 容纳整个序列（最坏情况：无 cache 命中）"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为新序列分配所有 block，同时尝试 prefix cache 复用。
        逐 block 计算哈希，若命中 cache 则共享已有 block（引用计数+1），
        一旦 cache miss 则后续所有 block 都新分配。"""
        assert not seq.block_table  # 序列尚未分配
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有填满的 block 才计算哈希（最后一个 block 可能未填满，hash=-1）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 哈希未命中或内容不匹配，标记为 miss
            if cache_miss:
                # cache miss：分配新的空闲 block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # cache hit：复用已有 block，跳过这些 token 的计算
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1  # 已被其他序列使用，增加引用计数
                else:
                    block = self._allocate_block(block_id)  # 在 free 中但哈希还在，重新激活
            if h != -1:
                block.update(h, token_ids)  # 填满的 block 注册到 prefix cache
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有 block。逆序释放使最后（最不可能被共享的）block 先回收。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)  # 无人引用则回收
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查 decode 阶段能否追加一个 token。
        仅当新 token 跨入新 block（num_tokens % block_size == 1）时需要 1 个空闲 block。"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 阶段追加 token 后更新 block 状态。三种情况：
        1. 跨入新 block（余 1）：分配新 block 并追加到 block_table
        2. 填满当前 block（余 0）：计算哈希并注册到 prefix cache
        3. 其他：当前 block 未填满，无需操作"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 新 token 刚好跨入新 block，需要分配新 block
            assert last_block.hash != -1  # 上一个 block 应已填满
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 当前 block 刚好填满，计算哈希注册到 prefix cache
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1  # block 未填满，无操作
