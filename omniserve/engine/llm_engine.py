# original file: https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }
import os
import copy
import time
from typing import Any, Iterable, List, Optional, Union

import torch

from omniserve.config import (
    CacheConfig,
    DeviceConfig,
    IFBConfig,
    ModelConfig,
    ParallelConfig,
    ProfilingConfig,
    SchedulerConfig,
)
from omniserve.core.scheduler import Scheduler, SchedulerOutputs
from omniserve.engine.arg_utils import EngineArgs
from omniserve.logger import init_logger
from omniserve.sampling_params import SamplingParams
from omniserve.sequence import (
    SamplerOutput,
    Sequence,
    SequenceGroup,
    SequenceGroupOutput,
    SequenceStatus,
)
from omniserve.utils.tokenizer import get_tokenizer  # TokenizerGroup
from omniserve.utils.utils import (
    Counter,
    get_distributed_init_method,
    get_ip,
    get_open_port,
)

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        ifb_config: IFBConfig,  # in-flight batching
        benchmarking_mode: bool,  # In benchmarking mode, we simplify the post-processing. (e.g., no sequence dequeuing)
        precision: str,
        int4_kv: bool,
        kv_zp: bool,
        quant_path: Optional[str],
        group_size: int,
        omit_prompt: bool,
        log_stats: bool,
        profiling_mode: bool = False,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"disable_custom_all_reduce={parallel_config.disable_custom_all_reduce}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"kv_cache_dtype={cache_config.cache_dtype}, "
            f"device_config={device_config.device}, "
            f"ifb_config={ifb_config.ifb_mode}, "
            f"ctx_attn_mode={model_config.sp_attn_config.sparse_context_enabled()}, "
            f"dec_attn_mode={model_config.sp_attn_config.sparse_decode_enabled()}, "
            f"attn_static_sparsity={model_config.sp_attn_config.get_static_sparsity()}, "
            f"seed={model_config.seed})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.log_stats = log_stats
        self.profiling_mode = profiling_mode
        self.benchmarking_mode = benchmarking_mode
        self.ifb_mode = ifb_config.ifb_mode
        self.precision = precision
        self.kv_cache_config = {"INT4_ENABLED": int4_kv, "ZEROS_ENABLED": kv_zp}
        self.init_num_blocks = (
            None  # Depends on the input & generation length, only used in non-IFB mode
        )
        self.quant_path = quant_path
        self.group_size = group_size
        self.omit_prompt = omit_prompt
        self._verify_args()

        self._init_tokenizer()
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        self._init_workers()

        # Profile the memory usage and initialize the cache.
        self._init_cache()
        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, ifb_config)

        if self.ifb_mode:
            print("Running with ifb mode")
        else:
            self.block_table_initialized = False
            print("Running without ifb mode")

        if self.benchmarking_mode:
            print("Running with benchmarking mode")

    def get_tokenizer_for_seq(self, sequence: Sequence):
        return self.tokenizer

    def _init_workers(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from omniserve.worker.worker import Worker

        # assert (
        #     self.parallel_config.world_size == 1
        # ), "Ray is required if parallel_config.world_size > 1."

        self.workers: List[Worker] = []
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
            precision=self.precision,
            kv_cache_config=self.kv_cache_config,
        )
        self.workers.append(self.driver_worker)
        # self._run_workers("init_model")
        # self._run_workers("load_model")

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        init_kwargs = dict(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision,
        )
        init_kwargs.update(tokenizer_init_kwargs)
        # self.tokenizer: TokenizerGroup = TokenizerGroup(
        #     self.model_config.tokenizer, **init_kwargs
        # )
        self.tokenizer = get_tokenizer(self.model_config.tokenizer, **init_kwargs)

    def _verify_args(self) -> None:
        # TODO(kentang-mit@: add back
        pass

        # self.model_config.verify_with_parallel_config(self.parallel_config)
        # self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameters.
        """
        # Initialize the cache.
        self._run_workers(
            "init_cache_engine",
            cache_config=self.cache_config,
            quant_path=self.quant_path,
            group_size=self.group_size,
        )

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        # placement_group = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(
            *engine_configs,
            # placement_group,
            log_stats=not engine_args.disable_log_stats,
        )
        return engine

    def encode_request(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)
        return prompt_token_ids

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        prefix_pos: Optional[int] = None,
        profiling_config: Optional[ProfilingConfig] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """

        if arrival_time is None:
            arrival_time = time.monotonic()

        if profiling_config is None:
            prompt_token_ids = self.encode_request(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
            )
            prompt_len = len(prompt_token_ids)
            print(f"prompt_len: {prompt_len}")
        else:
            # profiling mode
            prompt_len = profiling_config.prompt_len
            generation_len = profiling_config.generation_len
            prompt_token_ids = (
                torch.randint(self.model_config.get_vocab_size(), (prompt_len,))
                .numpy()
                .tolist()
            )
            sampling_params = copy.deepcopy(sampling_params)
            sampling_params.max_tokens = generation_len

        # TODO: (Shang) Add decoding simulation feature
        decoding_sim_length=int(os.environ.get("DECODING_SIMULATION_LENGTH", 0))
        simulation_start_idx = len(prompt_token_ids) - decoding_sim_length
        input_prompt_token_ids = prompt_token_ids[:simulation_start_idx]
        sim_prompt_token_ids = prompt_token_ids[simulation_start_idx:]
        sampling_params.update_decoding_sim_token_ids(sim_prompt_token_ids)
        # print(sampling_params)
        # print(self.workers[0].model_runner.model.sampler.sampling_params)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, input_prompt_token_ids, block_size, prompt_token_ids)

        # Check whether the input specifies prefix
        prefix = (
            self.scheduler.prefix_pool.add_or_get_prefix(
                input_prompt_token_ids[:prefix_pos],
                0,
            )
            if prefix_pos is not None
            else None
        )

        # Defensive copy of SamplingParams, which are used by the sampler
        sampling_params = copy.deepcopy(sampling_params)

        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id, [seq], sampling_params, arrival_time, prefix
        )

        # Add the sequence group to the scheduler.
        if prompt_len < self.model_config.max_model_len:
            self.scheduler.add_seq_group(seq_group)
            return True
        else:
            return False

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _process_sequence_group_outputs(
        self, seq_group: SequenceGroup, outputs: SequenceGroupOutput
    ) -> None:
        token_id = outputs
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        seqs[0].append_token_id(token_id, {token_id: 0})
        self._check_stop(seqs[0], seq_group.sampling_params)

    def _process_model_outputs(
        self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs
    ) -> List:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # Create the outputs.
        request_outputs: List = []
        num_finished = 0
        for seq_group in scheduled_seq_groups:
            seqs = seq_group.get_seqs()
            assert len(seqs) == 1
            if seq_group.is_finished():
                # free kv cache
                self.scheduler.free_seq(seqs[0])
                request_outputs.append(
                    {
                        "id": seq_group.request_id,
                        "text": seqs[0].output_text,
                        "finished": True,
                    }
                )
                num_finished += 1
            else:
                request_outputs.append(
                    {
                        "id": seq_group.request_id,
                        "tokens": seqs[0].get_token_ids(),
                        "finished": False,
                    }
                )

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        return request_outputs

    def _process_model_outputs_benchmark(
        self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs
    ) -> List:
        # Update the scheduled sequence groups with the model outputs. Only for benchmarking
        # Major difference: this function does not return output token ids
        # and it does not check stop, either.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            assert len(seqs) == 1
            seqs[0].append_token_id(outputs, {outputs: 0})

        return scheduled_seq_groups

    def update_init_num_blocks(self, init_num_blocks: int) -> None:
        self.init_num_blocks = init_num_blocks

    def step(self) -> List:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the workers to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id), prompt, sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """

        # Note: difference between ifb_mode and no_ifb_mode:
        # 1. scheduler is executed in each iteration for ifb, but only executed once for no_ifb
        # (i.e. benchmarking mode)
        # 2. no_ifb basically does not check stop and return output tokens in the end of model
        # execution.
        if self.ifb_mode:
            (
                self.seq_group_metadata_list,
                self.scheduler_outputs,
            ) = self.scheduler.schedule()
            # return seq_group_metadata_list, scheduler_outputs
            if not self.scheduler_outputs.is_empty():
                # Execute the model.
                all_outputs = self._run_workers(
                    "execute_model",
                    seq_group_metadata_list=self.seq_group_metadata_list,
                    ifb_mode=self.ifb_mode,
                )
                output = all_outputs[0].cpu().numpy().tolist()
                # return seq_group_metadata_list, scheduler_outputs, *all_outputs
                # # Only the driver worker returns the sampling results.
                # # TODO (kentang-mit@): check whether this is correct
                # output = all_outputs[0]
            else:
                output = []
            out = self._process_model_outputs(output, self.scheduler_outputs)
        else:
            # TODO (shang): Without ifb mode, implement how to decode
            # Execute the model.
            if not self.block_table_initialized:
                # initialize the block tables
                self.scheduler.update_init_num_blocks(self.init_num_blocks)
                (
                    self.seq_group_metadata_list,
                    self.scheduler_outputs,
                ) = self.scheduler.schedule()
                self._run_workers(
                    "init_block_tables",
                    seq_group_metadata_list=self.seq_group_metadata_list,
                    sliding_window=None,
                )
                self.block_table_initialized = True
            else:
                self.seq_group_metadata_list[0].is_prompt = False
            all_outputs = self._run_workers(
                "execute_model",
                seq_group_metadata_list=self.seq_group_metadata_list,
                ifb_mode=self.ifb_mode,
            )
            output = all_outputs[0].cpu().numpy().tolist()
            if self.benchmarking_mode:
                out = self._process_model_outputs_benchmark(
                    output, self.scheduler_outputs
                )
            else:
                out = self._process_model_outputs(output, self.scheduler_outputs)
        return out

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        if (
            seq.get_last_token_id() in sampling_params.stop_token_ids
            and not self.profiling_mode
        ):
            stop_str = self.get_tokenizer_for_seq(seq).convert_ids_to_tokens(
                seq.get_last_token_id()
            )
            self._finalize_sequence(seq)
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            self._finalize_sequence(seq)
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            self._finalize_sequence(seq)
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if (
            (not sampling_params.ignore_eos)
            and seq.get_last_token_id() == self.get_tokenizer_for_seq(seq).eos_token_id
            and not self.profiling_mode
        ):
            self._finalize_sequence(seq)
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _finalize_sequence(self, seq: Sequence) -> None:
        if not self.profiling_mode:
            if self.omit_prompt:
                token_ids = seq.data.get_token_ids()
                def remove_sublist(lst, sublist):
                    n = len(sublist)
                    i = 0
                    while i <= len(lst) - n:
                        if lst[i:i+n] == sublist:
                            del lst[i:i+n]
                        else:
                            i += 1
                    return lst
                token_ids = remove_sublist(token_ids, seq.full_prompt_token_ids)
                seq.output_text = self.get_tokenizer_for_seq(seq).decode(token_ids)
            else:
                seq.output_text = self.get_tokenizer_for_seq(seq).decode(
                    [token for token in seq.data.get_token_ids()]
                )

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        return all_outputs
