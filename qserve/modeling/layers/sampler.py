# original file: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sampler.py
# Modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
"""A layer that samples the next tokens from the model's outputs."""


import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata


def prepare_logits_processor(
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
    min_tokens_to_keep: int = 1,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    # Removed for the newest version of VILA
    # if repetition_penalty > 1.0:
    #     processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(
            TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)
        )
    return processor_list


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs."""

    def __init__(self, sampling_params: SamplingParams) -> None:
        super().__init__()
        self.sampling_params = sampling_params
        self.logits_processor = prepare_logits_processor(
            self.sampling_params.temperature,
            self.sampling_params.repetition_penalty,
            self.sampling_params.top_p,
            self.sampling_params.top_k,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        output_ids = input_ids.clone()
        if self.logits_processor:
            if self.sampling_params.repetition_penalty > 1.0:
                # tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                tmp_output_ids = output_ids
            else:
                tmp_output_ids = None
            # if input_metadata.is_prompt:
            if False:
                last_token_logits = self.logits_processor(
                    tmp_output_ids, logits[input_metadata.cu_seqlens[1:] - 1, :]
                )
            else:
                last_token_logits = self.logits_processor(tmp_output_ids, logits)
        else:
            # if input_metadata.is_prompt:
            if False:
                last_token_logits = logits[input_metadata.cu_seqlens[1:] - 1, :]
            else:
                last_token_logits = logits
        if (
            self.sampling_params.temperature < 1e-5 or self.sampling_params.top_p < 1e-8 or self.sampling_params.top_k == 1 # greedy
        ):
            token = torch.argmax(last_token_logits, dim=-1)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).view(-1)
        return token
