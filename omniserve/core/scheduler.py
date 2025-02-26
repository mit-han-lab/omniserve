# original file: https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
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
import enum
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union

from omniserve.config import CacheConfig, IFBConfig, SchedulerConfig
from omniserve.core.block_manager import AllocStatus, BlockSpaceManager
from omniserve.core.policy import PolicyFactory
from omniserve.logger import init_logger
from omniserve.prefix import PrefixPool
from omniserve.sequence import (
    Sequence,
    SequenceData,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceStatus,
)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """

    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:
    def __init__(
        self,
        scheduled_seq_groups: Iterable[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        retrieval_blocks_to_swap_in: Dict[int, int],
        streaming_blocks_to_swap_in: Dict[int, int],
        retrieval_blocks_to_swap_out: Dict[int, int],
        streaming_blocks_to_swap_out: Dict[int, int],
        retrieval_blocks_to_copy: Dict[int, List[int]],
        streaming_blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.retrieval_blocks_to_swap_in = retrieval_blocks_to_swap_in
        self.streaming_blocks_to_swap_in = streaming_blocks_to_swap_in
        self.retrieval_blocks_to_swap_out = retrieval_blocks_to_swap_out
        self.streaming_blocks_to_swap_out = streaming_blocks_to_swap_out
        self.retrieval_blocks_to_copy = retrieval_blocks_to_copy
        self.streaming_blocks_to_copy = streaming_blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (retrieval_blocks_to_swap_in and retrieval_blocks_to_swap_out)
        assert not (streaming_blocks_to_swap_in and streaming_blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (
            not self.scheduled_seq_groups
            and not self.retrieval_blocks_to_swap_in
            and not self.streaming_blocks_to_swap_in
            and not self.retrieval_blocks_to_swap_out
            and not self.streaming_blocks_to_swap_out
            and not self.retrieval_blocks_to_copy
            and not self.streaming_blocks_to_copy
        )


class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        ifb_config: IFBConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.ifb_mode = ifb_config.ifb_mode
        self.init_num_blocks = (
            None  # Depends on the input & generation length, only used in non-IFB mode
        )

        self.prompt_limit = min(
            self.scheduler_config.max_model_len,
            self.scheduler_config.max_num_batched_tokens,
        )

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_retrieval_gpu_blocks=self.cache_config.num_retrieval_gpu_blocks,
            num_retrieval_cpu_blocks=self.cache_config.num_retrieval_cpu_blocks,
            num_streaming_gpu_blocks=self.cache_config.num_streaming_gpu_blocks,
            num_streaming_cpu_blocks=self.cache_config.num_streaming_cpu_blocks,
            sp_attn_config=self.cache_config.sp_attn_config,
        )

        # Create the prefix pool to cache the prefixes.
        self.prefix_pool = PrefixPool(self.cache_config.block_size)

        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        self.swapped: Deque[SequenceGroup] = deque()

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def update_init_num_blocks(self, init_num_blocks: int) -> None:
        self.init_num_blocks = init_num_blocks

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        retrieval_blocks_to_swap_in: Dict[int, int] = {}
        streaming_blocks_to_swap_in: Dict[int, int] = {}
        retrieval_blocks_to_swap_out: Dict[int, int] = {}
        streaming_blocks_to_swap_out: Dict[int, int] = {}
        retrieval_blocks_to_copy: Dict[int, List[int]] = {}
        streaming_blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(
                seq_group.get_max_num_running_seqs() for seq_group in self.running
            )
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            leftover_waiting_sequences = deque()
            while self.waiting:
                seq_group = self.waiting[0]
                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt " "sequence."
                )
                num_prompt_tokens = waiting_seqs[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}"
                    )
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(
                    seq_group, self.ifb_mode, self.init_num_blocks
                )
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds the capacity of block_manager"
                    )
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = sum(
                    new_seq_lens
                )  # len(new_seq_lens) * max(new_seq_lens)
                if num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                # num_paddings = num_batched_tokens - sum(new_seq_lens)
                # if num_paddings > self.scheduler_config.max_paddings:
                #     break
                seq_lens = new_seq_lens

                self.waiting.popleft()
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            self.waiting.extendleft(leftover_waiting_sequences)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens) if seq_lens else 0,
                    retrieval_blocks_to_swap_in=retrieval_blocks_to_swap_in,
                    streaming_blocks_to_swap_in=streaming_blocks_to_swap_in,
                    retrieval_blocks_to_swap_out=retrieval_blocks_to_swap_out,
                    streaming_blocks_to_swap_out=streaming_blocks_to_swap_out,
                    retrieval_blocks_to_copy=retrieval_blocks_to_copy,
                    streaming_blocks_to_copy=streaming_blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: Deque[SequenceGroup] = deque()
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.popleft()
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop()
                    self._preempt(victim_seq_group, retrieval_blocks_to_swap_out=retrieval_blocks_to_swap_out, streaming_blocks_to_swap_out=streaming_blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, retrieval_blocks_to_swap_out=retrieval_blocks_to_swap_out, streaming_blocks_to_swap_out=streaming_blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, retrieval_blocks_to_copy=retrieval_blocks_to_copy, streaming_blocks_to_copy=streaming_blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(
                seq_group.get_max_num_running_seqs() for seq_group in self.running
            )

            leftover_swapped = deque()

            while self.swapped:
                seq_group = self.swapped[0]

                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                self.swapped.popleft()
                self._swap_in(seq_group, retrieval_blocks_to_swap_in=retrieval_blocks_to_swap_in, streaming_blocks_to_swap_in=streaming_blocks_to_swap_in)
                self._append_slot(seq_group, retrieval_blocks_to_copy=retrieval_blocks_to_copy, streaming_blocks_to_copy=streaming_blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

            self.swapped.extendleft(leftover_swapped)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            retrieval_blocks_to_swap_in=retrieval_blocks_to_swap_in,
            streaming_blocks_to_swap_in=streaming_blocks_to_swap_in,
            retrieval_blocks_to_swap_out=retrieval_blocks_to_swap_out,
            streaming_blocks_to_swap_out=streaming_blocks_to_swap_out,
            retrieval_blocks_to_copy=retrieval_blocks_to_copy,
            streaming_blocks_to_copy=streaming_blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            retrieval_block_tables: Dict[int, List[int]] = {}
            streaming_block_tables: Dict[int, Optional[List[int]]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                retrieval_block_tables[seq_id] = self.block_manager.get_retrieval_block_table(seq)
                streaming_block_tables[seq_id] = self.block_manager.get_streaming_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                retrieval_block_tables=retrieval_block_tables,
                streaming_block_tables=streaming_block_tables,
                prefix=seq_group.prefix,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def prepare_input(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Simplized version of schedule(), no need to manage kv blocks
        scheduler_outputs = self._schedule()
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=None,
                prefix=seq_group.prefix,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = deque(
            seq_group for seq_group in self.running if not seq_group.is_finished()
        )

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group, self.ifb_mode, self.init_num_blocks)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        retrieval_blocks_to_copy: Dict[int, List[int]],
        streaming_blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            retrieval_ret, streaming_ret = self.block_manager.append_slot(seq)
            if retrieval_ret is not None:
                src_block, dst_block = retrieval_ret
                if src_block in retrieval_blocks_to_copy:
                    retrieval_blocks_to_copy[src_block].append(dst_block)
                else:
                    retrieval_blocks_to_copy[src_block] = [dst_block]
            if streaming_ret is not None:
                src_block, dst_block = streaming_ret
                if src_block in streaming_blocks_to_copy:
                    streaming_blocks_to_copy[src_block].append(dst_block)
                else:
                    streaming_blocks_to_copy[src_block] = [dst_block]
            
            # if ret is not None:
            #     src_block, dst_block = ret
            #     if src_block in blocks_to_copy:
            #         blocks_to_copy[src_block].append(dst_block)
            #     else:
            #         blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        retrieval_blocks_to_swap_out: Dict[int, int],
        streaming_blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, retrieval_blocks_to_swap_out, streaming_blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.appendleft(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        retrieval_blocks_to_swap_out: Dict[int, int],
        streaming_blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, retrieval_blocks_to_swap_out, streaming_blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        retrieval_blocks_to_swap_in: Dict[int, int],
        streaming_blocks_to_swap_in: Dict[int, int],
    ) -> None:
        retrieval_mapping, streaming_mapping = self.block_manager.swap_in(seq_group)
        retrieval_blocks_to_swap_in.update(retrieval_mapping)
        streaming_blocks_to_swap_in.update(streaming_mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        retrieval_blocks_to_swap_out: Dict[int, int],
        streaming_blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error."
            )
        retrieval_mapping, streaming_mapping = self.block_manager.swap_out(seq_group)
        retrieval_blocks_to_swap_out.update(retrieval_mapping)
        streaming_blocks_to_swap_out.update(streaming_mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
