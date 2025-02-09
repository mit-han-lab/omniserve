# original file: https://github.com/vllm-project/vllm/blob/main/vllm/worker/cache_engine.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

"""CacheEngine class for managing the KV cache."""

from math import prod
from typing import Dict, List, Tuple

import torch

# TODO (kentang-mit@): cache_ops is not used now
# from vllm._C import cache_ops
from qserve.config import CacheConfig, ModelConfig, ParallelConfig
from qserve.logger import init_logger
from qserve.utils.utils import STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_cache_config: Dict,  # INT4_ENABLED: Whether to use int4 for kv_cache, ZEROS_ENABLED: Whether to use zero point for kv_cache
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        self.elements_per_block = prod(self.get_key_block_shape())
        # k and v (*2), 2 bytes per unit (*2)
        self.num_bytes_per_block = self.elements_per_block // (
            2 if kv_cache_config["INT4_ENABLED"] else 1
        ) + self.block_size * self.num_heads * (
            4 if kv_cache_config["ZEROS_ENABLED"] else 2
        )
        if kv_cache_config["INT4_ENABLED"]:
            assert kv_cache_config[
                "ZEROS_ENABLED"
            ], "INT4 KV Cache must be used with Zero Points."
            print("[INFO] USE INT4 w/ ZERO_POINTS for KV CACHE")
        else:
            assert kv_cache_config[
                "ZEROS_ENABLED"
            ], "INT8 KV Cache must be used with Zero Points."
            print("[INFO] USE INT8 for KV CACHE")
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int]:
        return (self.num_heads, self.block_size, self.head_size)

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.block_size,
            self.head_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(
                    self.num_gpu_blocks,
                    self.num_bytes_per_block,
                ),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, self.num_bytes_per_block),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    # def allocate_gpu_cache(self) -> torch.Tensor:
    #     key_block_shape = self.get_key_block_shape()
    #     value_block_shape = self.get_value_block_shape()
    #     gpu_cache: torch.Tensor = torch.empty(
    #         size=(2, self.num_layers, self.num_gpu_blocks, *key_block_shape),
    #         dtype=self.dtype,
    #         device="cuda",
    #     )
    #     return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = True
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        return
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        return
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_bit: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        return cache_bit * total // 8


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
