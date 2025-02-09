# original file: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from qserve.config import (
    CacheConfig,
    DeviceConfig,
    IFBConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)

_STR_DTYPE_TO_TORCH_DTYPE = {
    "int8": torch.int8,
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    kv_cache_dtype: str = "int8"
    seed: int = 0
    max_model_len: Optional[int] = None
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 64
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.75
    max_num_batched_tokens: int = 262144
    max_num_seqs: int = 256
    max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    device: str = "cuda"
    ifb_mode: bool = False
    benchmarking: bool = False
    precision: str = "w4a8kv4"
    # int4_kv: bool = False
    # kv_zp: bool = True
    quant_path: Optional[str] = None
    group_size: int = -1
    run_vlm: bool = False
    omit_prompt: bool = False
    img_per_seq: int = 0
    img_files: str = None
    img_rotation: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            "--model",
            type=str,
            default="facebook/opt-125m",
            help="name or path of the huggingface model to use",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=EngineArgs.tokenizer,
            help="name or path of the huggingface tokenizer to use",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="the specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--code-revision",
            type=str,
            default=None,
            help="the specific revision to use for the model code on "
            "Hugging Face Hub. It can be a branch name, a tag name, or a "
            "commit id. If unspecified, will use the default version.",
        )
        parser.add_argument(
            "--tokenizer-revision",
            type=str,
            default=None,
            help="the specific tokenizer version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help='tokenizer mode. "auto" will use the fast '
            'tokenizer if available, and "slow" will '
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="trust remote code from huggingface",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=EngineArgs.download_dir,
            help="directory to download and load the weights, "
            "default to the default cache dir of "
            "huggingface",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=EngineArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=EngineArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="data type for model weights and activations. "
            'The "auto" option will use FP16 precision '
            "for FP32 and FP16 models, and BF16 precision "
            "for BF16 models.",
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            choices=["int8", "fp16", "fp8_e5m2"],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            "data type. Note FP8 is not supported when cuda version is "
            "lower than 11.8.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="model context length. If unspecified, "
            "will be automatically derived from the model.",
        )
        # Parallel arguments
        parser.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=EngineArgs.pipeline_parallel_size,
            help="number of pipeline stages",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="number of tensor parallel replicas",
        )
        parser.add_argument(
            "--max-parallel-loading-workers",
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help="load model sequentially in multiple batches, "
            "to avoid RAM OOM when using tensor "
            "parallel and large models",
        )
        # KV cache arguments
        parser.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            choices=[64],
            help="token block size",
        )
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument(
            "--seed", type=int, default=EngineArgs.seed, help="random seed"
        )
        parser.add_argument(
            "--swap-space",
            type=int,
            default=EngineArgs.swap_space,
            help="CPU swap space size (GiB) per GPU",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="the fraction of GPU memory to be used for "
            "the model executor, which can range from 0 to 1."
            "If unspecified, will use the default value of 0.9.",
        )
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="maximum number of batched tokens per " "iteration",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="maximum number of sequences per iteration",
        )
        parser.add_argument(
            "--max-paddings",
            type=int,
            default=EngineArgs.max_paddings,
            help="maximum number of paddings in a batch",
        )
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="disable logging statistics",
        )
        # Quantization settings.
        parser.add_argument(
            "--quantization",
            "-q",
            type=str,
            choices=["awq", "gptq", "squeezellm", None],
            default=EngineArgs.quantization,
            help="Method used to quantize the weights. If "
            "None, we first check the `quantization_config` "
            "attribute in the model config file. If that is "
            "None, we assume the model weights are not "
            "quantized and use `dtype` to determine the data "
            "type of the weights.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Always use eager-mode PyTorch. If False, "
            "will use eager mode and CUDA graph in hybrid "
            "for maximal performance and flexibility.",
        )
        parser.add_argument(
            "--max-context-len-to-capture",
            type=int,
            default=EngineArgs.max_context_len_to_capture,
            help="maximum context length covered by CUDA "
            "graphs. When a sequence has context length "
            "larger than this, we fall back to eager mode.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=EngineArgs.disable_custom_all_reduce,
            help="See ParallelConfig",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=EngineArgs.device,
            choices=["cuda"],
            help=(
                "Device type for vLLM execution. "
                "Currently, only CUDA-compatible devices are supported."
            ),
        )
        parser.add_argument(
            "--ifb-mode",
            action="store_true",
            help="Enable In-flight Batching mode.",
        )
        parser.add_argument(
            "--benchmarking",
            action="store_true",
            help="Enable Profiling mode.",
        )
        parser.add_argument(
            "--precision",
            type=str,
            default="w4a8kv4",
            help="Model precision. Select from [w4a8kv4, w4a8kv8, w8a8kv4, w8a8kv8]. If kv precision is not specified, it will be the same as the activation.",
        )
        # parser.add_argument(
        #     "--int4-kv",
        #     action="store_true",
        #     help="Use 4-bit quantization for key-value cache",
        # )
        # parser.add_argument(
        #     "--kv-zp",
        #     action="store_true",
        #     help="Use zero-point quantization for key-value cache",
        # )
        parser.add_argument(
            "--quant-path",
            type=str,
            default=None,
            help="Path to the quantized checkpoint",
        )
        parser.add_argument(
            "--group-size",
            type=int,
            default=-1,
            help="Group size for weight quantization, -1 means per-channel",
        )
        parser.add_argument(
            "--run-vlm",
            action="store_true",
            help="Run Visual Language Models (VILA)",
        )
        parser.add_argument(
            "--omit-prompt",
            action="store_true",
            help="Whether to omit the prompt in the final output",
        )
        parser.add_argument(
            "--img-per-seq",
            type=int,
            default=0,
            help="Number of images per sequence",
        )
        parser.add_argument(
            "--img-files",
            type=str,
            default=None,
            help="Path to the image files, split by comma.",
        )
        parser.add_argument(
            "--img-rotation",
            action="store_true",
            help="Enable image rotation. When rotation (hadamard) is used in LLM quantization, the image embedding should also be rotated.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[
        ModelConfig,
        CacheConfig,
        ParallelConfig,
        SchedulerConfig,
        DeviceConfig,
        IFBConfig,
        bool,  # benchmarking_mode
        str,  # precision
        bool,  # int4_kv
        bool,  # kv_zp
        str,  # quant_path
        int,  # group_size
        bool,  # run_vlm
        bool, # omit_prompt
        int,  # img_per_seq
        str,  # img_files
        bool, # img_rotation
    ]:
        assert self.precision in [
            "w4a8",
            "w4a8kv4",
            "w4a8kv8",
            "w8a8",
            "w8a8kv4",
            "w8a8kv8",
            "w16a16kv8",
            "w16a16kv4"
        ], f"Invalid precision {self.precision} specified. Please choose from w4a8, w4a8kv4, w4a8kv8, w8a8, w8a8kv4, w8a8kv8, w16a16kv8, w16a16kv4."

        if "kv4" in self.precision:
            self.kv_cache_bits = 4
            self.int4_kv = True
        else:
            self.kv_cache_bits = 8
            self.int4_kv = False
        precision = self.precision
        self.kv_zp = True

        kv_zp = self.kv_zp
        int4_kv = self.int4_kv

        device_config = DeviceConfig(self.device)
        model_config = ModelConfig(
            self.model,
            self.tokenizer,
            self.tokenizer_mode,
            self.trust_remote_code,
            self.download_dir,
            self.load_format,
            self.dtype,
            self.seed,
            self.revision,
            self.code_revision,
            self.tokenizer_revision,
            self.max_model_len,
            self.quantization,
            self.enforce_eager,
            self.max_context_len_to_capture,
            self.run_vlm,
        )
        self.kv_cache_bits = _get_dtype_size(
            _STR_DTYPE_TO_TORCH_DTYPE[self.kv_cache_dtype]
        )
        if self.int4_kv:
            # The storage of kv cache is still in int8 format, so we do not change kv_cache_dtype here
            self.kv_cache_bits = 4

        cache_config = CacheConfig(
            self.block_size,
            self.gpu_memory_utilization,
            self.swap_space,
            self.kv_cache_dtype,
            self.kv_cache_bits,
            model_config.get_sliding_window(),
        )
        parallel_config = ParallelConfig(
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.max_parallel_loading_workers,
            self.disable_custom_all_reduce,
        )
        scheduler_config = SchedulerConfig(
            self.max_num_batched_tokens,
            self.max_num_seqs,
            model_config.max_model_len,
            self.max_paddings,
        )
        ifb_config = IFBConfig(self.ifb_mode)
        benchmarking_mode = self.benchmarking

        quant_path = self.quant_path
        group_size = self.group_size
        run_vlm = self.run_vlm
        omit_prompt = self.omit_prompt
        img_per_seq = self.img_per_seq
        img_files = self.img_files
        img_rotation = self.img_rotation
        return (
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            device_config,
            ifb_config,
            benchmarking_mode,
            precision,
            int4_kv,
            kv_zp,
            quant_path,
            group_size,
            run_vlm,
            omit_prompt,
            img_per_seq,
            img_files,
            img_rotation,
        )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--disable-log-requests",
            action="store_true",
            help="disable logging requests",
        )
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=None,
            help="max number of prompt characters or prompt "
            "ID numbers being printed in log. "
            "Default: unlimited.",
        )
        return parser


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
