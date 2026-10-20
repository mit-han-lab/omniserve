# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
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

# Inspired by the following papers:
# @article{jiang2024mixtral,
#   title={Mixtral of experts},
#   author={Jiang, Albert Q and Sablayrolles, Alexandre and Roux, Antoine and Mensch, Arthur and Savary, Blanche and Bamford, Chris and Chaplot, Devendra Singh and Casas, Diego de las and Hanna, Emma Bou and Bressand, Florian and others},
#   journal={arXiv preprint arXiv:2401.04088},
#   year={2024}
# }
# MoE release will come in the future.
from typing import Dict, List, Optional

# import omniserve_backend.fused_attention as fused_attention
import omniserve_backend.fused_attention_fine_grained_dense as fused_attention # TODO: Fix this

# import gc
import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from omniserve_backend import fused_kernels
from torch import nn
from transformers import MixtralConfig

import omniserve.utils.constants
from omniserve.utils.rope_utils import get_rope_scaling, get_rope_theta
from omniserve.modeling.layers.activation import SiluAndMulQuant
from omniserve.modeling.layers.layernorm import RMSNorm, RMSNormGeneral
from omniserve.modeling.layers.quantized_linear import (
    MoEW4A8OF16LinearDynamicInputScale,
    W4A8OF16LinearDynamicInputScale,
)
from omniserve.modeling.layers.sampler import Sampler
from omniserve.sampling_params import SamplingParams
from omniserve.utils.input_metadata import InputMetadata
from omniserve.utils.quant_config import QServeQuantConfig
from omniserve.utils.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from omniserve.modeling.layers.ctx_attn.ctx_attn_func import attention_wrapper
# import moe_helpers
from torch.cuda import nvtx
max_seq_len = omniserve.utils.constants.max_seq_len


class MixtralAttention(nn.Module):
    def __init__(
        self, args, layer_idx: int, kv_cache_config: Optional[Dict] = None
    ) -> None:
        super().__init__()

        hidden_size = args.hidden_size
        num_heads = args.num_attention_heads
        num_kv_heads = args.num_key_value_heads
        rope_theta = get_rope_theta(args)
        rope_scaling = get_rope_scaling(args)
        max_position_embeddings = args.max_position_embeddings

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size

        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        num_kv_heads_replicas = max(1, tp_size // self.total_num_kv_heads)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_int8 = True

        if kv_cache_config is None:
            self.kv_cache_config = {"INT4_ENABLED": False, "ZEROS_ENABLED": False}
            print("[Warning] kv cache config is not provided, using default config KV8")
        else:
            self.kv_cache_config = kv_cache_config

        self.qkv_proj = W4A8OF16LinearDynamicInputScale(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads * num_kv_heads_replicas)
            * self.head_dim,
            bias=args.attention_bias,
            group_size=128,
        )

        self.o_proj = W4A8OF16LinearDynamicInputScale(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=args.attention_bias,
            group_size=128,
        )

        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

    def forward(
        self,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor,
        input_metadata: InputMetadata,
        qkv_proj_act_buffer: torch.Tensor,
        out_down_proj_act_buffer: torch.Tensor,
    ):
        # INT8 in, FP16 out for this module
        # print(self.layer_idx, "begin", hidden_states.isnan().sum(), input_scale.shape)
        self.qkv_proj(
            quantized_hidden_states_buffer,
            quantized_scale_buffer,
            quantized_sum_buffer,
            qkv_proj_act_buffer,
        )
        # qkv = qkv.half()
        if input_metadata.is_prompt:
            # Note: the conversion of kv_scale_orig_quant is currently important
            # by default, self.kv_scale_orig_quant will have the same dtype as the model.
            # but the kernel requires float.
            size_per_retrieval_token = self.num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            size_per_streaming_token = self.num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            fused_attention.apply_bias_rope_update_kv_cache(
                qkv_proj_act_buffer,
                input_metadata.retrieval_context_lens,
                input_metadata.streaming_context_lens,
                input_metadata.padding_offsets,  # size [batch_size, max_seq_len]
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                self.retrieval_head_flags,
                self.head_rank_table,
                self.num_heads,
                self.num_kv_heads,
                input_metadata.max_seq_len,
                64,  # tokens_per_block
                size_per_retrieval_token, 
                size_per_streaming_token,
                self.sink_size, self.local_size,
                self.sink_blocks, self.local_blocks,
                self.num_retrieval_kv_heads,
                self.num_streaming_kv_heads,
                self.head_dim,
                self.rope_theta,
                self.max_position_embeddings,
                True,  # neox style
                self.kv_cache_config["INT4_ENABLED"],  # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
            )

            # FIXME: currently qkv share same scale, plan to use seperate scales
            q, k, v = qkv_proj_act_buffer.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            # k_cache, v_cache = kv_cache
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
            # with nvtx.range("context_attention"):
            attn_output = attention_wrapper(
                q, k, v,
                cu_seqlens_q=input_metadata.cu_seqlens,
                cu_seqlens_k=input_metadata.cu_seqlens,
                max_seqlen_q=input_metadata.max_seq_len,
                max_seqlen_k=input_metadata.max_seq_len,
                dropout_p=0.0, causal=True,
                head_mask_type=self.head_mask_type,
                streaming_info=self.streaming_info,
            )
            attn_output = attn_output.reshape(q.size(0), -1)
        else:
            q, k, v = qkv_proj_act_buffer.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            # k_cache, v_cache = kv_cache
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
            # Shang's important fix, but it might cause problem in the end of a block...
            lengths_per_sample = input_metadata.retrieval_context_lens  # + 1
            alibi_slopes = None
            memory_max_len = self.kv_max_seq_len
            tokens_per_block = 64
            
            size_per_retrieval_token = self.num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            size_per_streaming_token = self.num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            
            timestep = input_metadata.max_seq_len
            rotary_embedding_dim = self.head_dim
            rotary_base = self.rope_theta
            neox_rotary_style = True
            # shape is (#tokens, #heads, head_dim)

            attn_output = fused_attention.single_query_attention(
                q,
                k,
                v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                self.retrieval_head_flags,
                self.head_rank_table,
                lengths_per_sample,
                alibi_slopes,
                memory_max_len,
                tokens_per_block,
                size_per_retrieval_token, 
                size_per_streaming_token,
                self.sink_size, self.local_size,
                self.sink_blocks, self.local_blocks,
                self.num_retrieval_kv_heads,
                self.num_streaming_kv_heads,
                timestep,
                rotary_embedding_dim,
                rotary_base,
                neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],  # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
            )
            attn_output = attn_output.reshape(q.size(0), -1)

        # FP16 in, INT8 out
        fused_kernels.invoke_quant_fuse_sum(
            quantized_hidden_states_buffer,
            attn_output,
            quantized_sum_buffer,
            quantized_scale_buffer,
        )
        # INT8 in, FP16 out
        self.o_proj(
            quantized_hidden_states_buffer,
            quantized_scale_buffer,
            quantized_sum_buffer,
            out_down_proj_act_buffer,
        )


class MixtralSparseMoeBlockOurs(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.w1_w3 = MoEW4A8OF16LinearDynamicInputScale(
            w_bit=4,
            num_experts=self.num_experts,
            in_features=self.hidden_dim,
            out_features=self.ffn_dim * 2,
            bias=False,
            group_size=128,
        )
        self.w2 = MoEW4A8OF16LinearDynamicInputScale(
            w_bit=4,
            num_experts=self.num_experts,
            in_features=self.ffn_dim,
            out_features=self.hidden_dim,
            bias=False,
            group_size=128,
        )
        self.act_fn = SiluAndMulQuant()

        # Jitter parameters
        # self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states).float()
        ##################### step1: topk softmax kernel #####################

        # permuted loc to unpermuted loc
        raise NotImplementedError("MOE is yet to be supported")

        (
            routing_weights,
            expert_indices,
            permuted_loc_to_unpermuted_loc,
        ) = moe_helpers.moe_topk_gating_softmax(router_logits, self.top_k)
        problem_sizes = torch.bincount(expert_indices, minlength=self.num_experts).int()
        problem_sizes = F.pad(problem_sizes, pad=(1, 0), value=0)
        q_hidden_states = torch.empty(
            num_tokens, hidden_dim, dtype=torch.int8, device=hidden_states.device
        )
        q_input_ssums = torch.empty(
            num_tokens,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        q_input_scales = torch.empty(
            num_tokens,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        # quantize layernorm outputs to INT8
        fused_kernels.invoke_quant_fuse_sum(
            q_hidden_states,
            hidden_states,
            q_input_ssums,
            q_input_scales,
        )
        # print("topk softmax", routing_map)
        ##################### step2: expand inputs #####################
        (
            q_moe_input_feats,
            moe_input_ssums,
            moe_input_scales,
            unpermuted_loc_to_permuted_loc,
        ) = moe_helpers.moe_expand_inputs_and_scales(
            q_hidden_states,
            q_input_ssums,
            q_input_scales,
            permuted_loc_to_unpermuted_loc,
        )

        # print("expand input", q_moe_input_feats)
        # ##################### step3: MoE-GEMM #####################
        # ###### 3.1 self.w1, self.w3 ######
        # # gate_proj
        q_w1_w3_out = self.w1_w3(
            q_moe_input_feats,
            moe_input_scales,
            moe_input_ssums,
            problem_sizes,
        )

        # gate_up_out = F.silu(q_w1_out) * q_w3_out
        q_gate_up_out = torch.empty(
            num_tokens * self.top_k,
            self.ffn_dim,
            device=hidden_states.device,
            dtype=torch.int8,
        )
        moe_input_ssums = torch.empty(
            num_tokens * self.top_k,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        moe_input_scales = torch.empty(
            num_tokens * self.top_k,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        self.act_fn(q_w1_w3_out, q_gate_up_out, moe_input_scales, moe_input_ssums)

        ###### 3.2 self.w2 ######
        q_w2_out = self.w2(
            q_gate_up_out, moe_input_scales, moe_input_ssums, problem_sizes
        )
        # # print("w2 output", q_w2_out)
        # ##################### step4: Reorder #####################
        output_hidden_states = moe_helpers.moe_finalize_routing(
            q_w2_out,
            routing_weights,
            unpermuted_loc_to_permuted_loc,
            num_tokens,
        )
        # print("router", output_hidden_states)
        return output_hidden_states, routing_weights


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = True

        self.self_attn = MixtralAttention(
            config, layer_idx=layer_idx, kv_cache_config=kv_cache_config
        )

        self.block_sparse_moe = MixtralSparseMoeBlockOurs(config)

        self.input_layernorm = RMSNormGeneral(
            config.hidden_size, eps=config.rms_norm_eps, use_per_token_quant=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        qkv_proj_act_buffer: torch.Tensor,
        out_down_proj_act_buffer: torch.Tensor,
        gate_up_proj_act_buffer: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_mlp_act_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor,
    ) -> torch.Tensor:
        # with nvtx.range("layer_fwd"):
        residual = hidden_states

        self.input_layernorm(
            hidden_states,
            quantized_hidden_states_buffer,
            quantized_scale_buffer,
            quantized_sum_buffer,
        )

        # Self Attention
        self.self_attn(
            quantized_hidden_states_buffer=quantized_hidden_states_buffer,
            input_metadata=input_metadata,
            quantized_scale_buffer=quantized_scale_buffer,
            quantized_sum_buffer=quantized_sum_buffer,
            qkv_proj_act_buffer=qkv_proj_act_buffer,
            out_down_proj_act_buffer=out_down_proj_act_buffer,
        )

        hidden_states = residual + out_down_proj_act_buffer

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MixtralModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_kv_cache: bool = True,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            # self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                (
                    MixtralDecoderLayer(config, i, kv_cache_config)
                    if quant_kv_cache
                    else None
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # self._attn_implementation = config._attn_implementation
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)
            batched_seq_len = hidden_states.shape[0]

            # TODO: We may need to change the buffer name.
            act_buffer = torch.empty(
                (
                    batched_seq_len
                    * max(
                        self.q_size + 2 * self.kv_size,
                        2 * self.config.intermediate_size,
                    )
                ),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            qkv_proj_act_buffer = act_buffer[
                : batched_seq_len * (self.q_size + 2 * self.kv_size)
            ].view(batched_seq_len, self.q_size + 2 * self.kv_size)
            out_down_proj_act_buffer = act_buffer[
                : batched_seq_len * self.config.hidden_size
            ].view(batched_seq_len, self.config.hidden_size)
            gate_up_proj_act_buffer = act_buffer[
                : batched_seq_len * 2 * self.config.intermediate_size
            ].view(batched_seq_len, 2 * self.config.intermediate_size)

            quantized_act_buffer = torch.empty(
                (
                    batched_seq_len
                    * max(self.config.hidden_size, self.config.intermediate_size)
                ),
                device=hidden_states.device,
                dtype=torch.int8,
            )
            quantized_hidden_states_buffer = quantized_act_buffer[
                : batched_seq_len * self.config.hidden_size
            ].view(batched_seq_len, self.config.hidden_size)
            quantized_mlp_act_buffer = quantized_act_buffer[
                : batched_seq_len * self.config.intermediate_size
            ].view(batched_seq_len, self.config.intermediate_size)

            quantized_scale_buffer = torch.empty(
                (batched_seq_len), device=hidden_states.device, dtype=torch.float16
            )
            quantized_sum_buffer = torch.empty(
                (batched_seq_len), device=hidden_states.device, dtype=torch.float16
            )

            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    hidden_states,
                    input_metadata,
                    qkv_proj_act_buffer,
                    out_down_proj_act_buffer,
                    gate_up_proj_act_buffer,
                    quantized_hidden_states_buffer,
                    quantized_mlp_act_buffer,
                    quantized_scale_buffer,
                    quantized_sum_buffer,
                )
            hidden_states = self.norm(hidden_states)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: MixtralConfig,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(
            config, quant_kv_cache, kv_cache_config=kv_cache_config
        )
        vocab_size = config.vocab_size

        # NOTE: The LM head is not quantized.
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "w2"]
        self.sampler = Sampler(sampling_params)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        if quant_path is not None:
            self.load_weights(quant_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, input_metadata)
        if input_metadata.is_prompt:
            output = self.lm_head(
                hidden_states[input_metadata.cu_seqlens[1:] - 1, :]
            )  # only compute last logits
        else:
            output = self.lm_head(hidden_states)
        return output  # .float()

    def sample(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        return self.sampler(input_ids, logits, input_metadata)

    def sample(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        return self.sampler(input_ids, logits, input_metadata)

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        if self.quant_config is None:
            col_weight_suffixes = ["weight"]
            row_weight_suffixes = ["weight"]
        else:
            col_weight_suffixes = self.quant_config.get_col_parallel_tensor_names()
            row_weight_suffixes = self.quant_config.get_row_parallel_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # TODO fix the tp parallelism
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        tp_size = 1
        tp_rank = 0

        q_proj_shard_size = self.config.hidden_size // tp_size
        num_kv_heads_replicas = max(1, tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1, self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * num_kv_heads_per_gpu
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]

        w1_proj_shard_size = self.config.intermediate_size // tp_size
        w3_proj_shard_size = self.config.intermediate_size // tp_size
        w2_proj_shard_size = self.config.hidden_size // tp_size

        moe_weight_specs_w1_w3 = []
        moe_weight_specs_w2 = []
        for expert_idx in range(self.num_experts):
            moe_weight_specs_w1_w3.append(
                (f"experts.{expert_idx}.w1", expert_idx, w1_proj_shard_size, 0)
            )
            moe_weight_specs_w1_w3.append(
                (
                    f"experts.{expert_idx}.w3",
                    expert_idx,
                    w3_proj_shard_size,
                    w1_proj_shard_size,
                )
            )
            moe_weight_specs_w2.append(
                (f"experts.{expert_idx}.w2", expert_idx, w2_proj_shard_size, 0)
            )

        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
                pass
                # continue
            if "norm" in name:
                continue

            packed_dim = None
            is_transposed = False
            if self.quant_config is not None:
                packed_dim = self.quant_config.get_packed_dim(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                # print(weight_name)
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if packed_dim is not None:
                    shard_dim = 0 if not is_transposed else 1
                    if packed_dim == shard_dim:
                        shard_size //= self.quant_config.pack_factor
                        offset //= self.quant_config.pack_factor

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[
                    shard_size * shard_id : shard_size * (shard_id + 1)
                ]
                if "s2_scales" in name or "s2_zeros" in name:
                    param_slice = param.data[:, offset : offset + shard_size]
                else:
                    param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for weight_name, expert_idx, shard_size, offset in moe_weight_specs_w1_w3:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "w1_w3")]
                if is_transposed:
                    param = param.T

                if "s2_scales" in name or "s2_zeros" in name:
                    param_slice = param.data[
                        expert_idx, :, offset : offset + shard_size
                    ]
                else:
                    param_slice = param.data[expert_idx, offset : offset + shard_size]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            is_w2_weight = False
            for weight_name, expert_idx, shard_size, offset in moe_weight_specs_w2:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "w2")]
                if is_transposed:
                    param = param.T

                if "s2_scales" in name or "s2_zeros" in name:
                    param_slice = param.data[
                        expert_idx, :, offset : offset + shard_size
                    ]
                else:
                    param_slice = param.data[expert_idx, offset : offset + shard_size]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_w2_weight = True
                break
            if is_w2_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )
