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
# @article{touvron2023llama,
#   title={Llama 2: Open foundation and fine-tuned chat models},
#   author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
#   journal={arXiv preprint arXiv:2307.09288},
#   year={2023}
# }

# @article{touvron2023llama,
#   title={Llama: Open and efficient foundation language models},
#   author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and others},
#   journal={arXiv preprint arXiv:2302.13971},
#   year={2023}
# }


from typing import Dict, List, Optional

# import omniserve_backend.fused_attention as fused_attention

# import gc
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from omniserve_backend import fused_kernels
import torch
from torch import nn
from transformers import LlamaConfig

import omniserve.utils.constants
from omniserve.utils.rope_utils import get_rope_scaling, get_rope_theta
from omniserve.modeling.layers.activation import SiluAndMul
from omniserve.modeling.layers.layernorm import RMSNorm, RMSNormGeneral
from omniserve.modeling.layers.sampler import Sampler
from omniserve.modeling.layers.ctx_update_kv import ApplyBiasRopeUpdateKVCacheWrapper, PagedMinMaxPoolWrapper
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
from omniserve.modeling.layers.decoding_attention import DecodingAttentionWrapper
from torch.cuda import nvtx
import os
from omniserve.config import ModelConfig

max_seq_len = omniserve.utils.constants.max_seq_len

# from omniserve.modeling.models.transformers_utils import LlamaRMSNorm, per_token_activation_quantization

class LlamaMLP(nn.Module):
    def __init__(self, args, model_config) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_int8 = True
        self.model_config = model_config

        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False,
        )

        self.act_fn = SiluAndMul()

    def forward(self, x):
        # INT8 in, FP16 out
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x 


class LlamaAttention(nn.Module):
    def __init__(
        self,
        args,
        model_config: ModelConfig,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        num_heads = args.num_attention_heads
        num_kv_heads = args.num_key_value_heads
        rope_theta = get_rope_theta(args)
        rope_scaling = get_rope_scaling(args)
        max_position_embeddings = args.max_position_embeddings

        self.layer_idx = layer_idx
        self.sparse_kv_cache_enabled = model_config.sp_attn_config.sparse_kv_cache_enabled()

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
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.use_int8 = True

        if kv_cache_config is None:
            self.kv_cache_config = {"INT4_ENABLED": False, "ZEROS_ENABLED": False}
            print("[Warning] kv cache config is not provided, using default config KV8")
        else:
            self.kv_cache_config = kv_cache_config

        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False
        self.qkv_proj = nn.Linear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads * num_kv_heads_replicas)
            * self.head_dim,
            bias=attention_bias,
        )

        self.o_proj = nn.Linear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )
        self.kv_scale_quant_orig = nn.Parameter(torch.ones(2))
        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

        # self.invoke_quant = self.invoke_quant_wo_act_sum


        self.tokens_per_block = 64                                                                          # TODO: This should be a parameter
        self.kv_quant_granularity = model_config.kv_quant_granularity
        self.sub_chunk_per_block = model_config.sp_attn_config.get_dec_sub_chunk_per_block()
        self.sparse_decode_mode = model_config.sp_attn_config.get_sparse_decode_mode()
        self.dynamic_sparse_token_budget = model_config.sp_attn_config.get_dec_dynamic_sparse_token_budget()
        self.sub_chunk_size = self.tokens_per_block // self.sub_chunk_per_block
        self.selector_update_interval = model_config.sp_attn_config.get_dec_selector_update_interval()
        self.multiblock_switch = model_config.multiblock_switch

        self.alibi_slopes = None                                                            
        self.rotary_embedding_dim = self.head_dim
        self.neox_rotary_style = True

        self.apply_bias_rope_update_kv_cache_wrapper = ApplyBiasRopeUpdateKVCacheWrapper(
            self.layer_idx, 
            self.num_heads, self.num_kv_heads, self.tokens_per_block, 
            self.head_dim, self.rope_theta, self.rope_scaling,
            self.max_position_embeddings, self.neox_rotary_style,
            self.kv_quant_granularity, self.kv_cache_config, self.use_int8,
        )
        self.paged_min_max_pool_wrapper = PagedMinMaxPoolWrapper(
            self.layer_idx, self.tokens_per_block, self.sub_chunk_per_block,
            self.head_dim, self.kv_cache_config, self.use_int8
        )
        self.decoding_attention_wrapper = DecodingAttentionWrapper(
            self.layer_idx, self.sparse_kv_cache_enabled,
            self.head_dim, self.alibi_slopes, self.kv_max_seq_len, self.tokens_per_block,
            self.rotary_embedding_dim, self.rope_theta, self.rope_scaling,
            self.neox_rotary_style, self.kv_quant_granularity, self.kv_cache_config, self.use_int8,
            self.sparse_decode_mode, self.sub_chunk_size, self.dynamic_sparse_token_budget,
            self.multiblock_switch, self.selector_update_interval
        )


    # def invoke_quant_wo_act_sum(self, activation_buffer, attn_output):
    #     fused_kernels.invoke_quant(
    #         activation_buffer.quantized_hidden_states_buffer,
    #         attn_output,
    #         activation_buffer.quantized_scale_buffer,
    #     )

    def forward(
        self,
        hidden_states,
        input_metadata: InputMetadata,
    ):
        qkv = self.qkv_proj(hidden_states)

        # qkv = qkv.half()
        if input_metadata.is_prompt:
            if not hasattr(self, "cached_dynamic_sparse_page_idx"):
                self.cached_dynamic_sparse_page_idx = None
            if self.cached_dynamic_sparse_page_idx is not None:
                self.cached_dynamic_sparse_page_idx = None   # reset the chosen dynamic_sparse pages in context stage

            # Note: the conversion of kv_scale_orig_quant is currently important
            # by default, self.kv_scale_orig_quant will have the same dtype as the model.
            # but the kernel requires float.
            size_per_retrieval_token = self.num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            size_per_streaming_token = self.num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

            kv_scale_quant_orig = self.kv_scale_quant_orig.float()
            kv_scale_orig_quant = 1 / kv_scale_quant_orig

            self.apply_bias_rope_update_kv_cache_wrapper(
                qkv, input_metadata, 
                self.retrieval_head_flags, self.head_rank_table,
                self.sink_size, self.local_size, self.sink_blocks, self.local_blocks,
                self.num_retrieval_kv_heads, self.num_streaming_kv_heads,
                kv_scale_orig_quant
            )

            # FIXME: currently qkv share same scale, plan to use seperate scales
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
            
            if self.sparse_decode_mode != 0:
                k = k.contiguous()
                self.paged_min_max_pool_wrapper(
                    k, input_metadata, self.pooling_heads_idx, self.num_retrieval_kv_heads
                )

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
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
            # alibi_slopes = None
            # memory_max_len = self.kv_max_seq_len
            # tokens_per_block = 64
            
            # size_per_retrieval_token = self.num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
            # size_per_streaming_token = self.num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

            attn_output, dynamic_sparse_page_idx_to_cache = self.decoding_attention_wrapper(
                q, k, v,
                input_metadata,
                self.retrieval_head_flags, self.head_rank_table,
                self.sink_size, self.local_size, self.sink_blocks, self.local_blocks,
                self.num_retrieval_kv_heads, self.num_streaming_kv_heads,
                self.cached_dynamic_sparse_page_idx,
                self.kv_scale_quant_orig
            )
            self.cached_dynamic_sparse_page_idx = dynamic_sparse_page_idx_to_cache

            attn_output = attn_output.reshape(q.size(0), -1)
            
        # # FP16 in, INT8 out
        # self.invoke_quant(activation_buffer, attn_output)
        # # INT8 in, FP16 out
        # print(f"attn_output.shape: {attn_output.shape}")
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        model_config: ModelConfig,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = True    # NOTE: INT8 kv cache
        # Requires transformers > 4.32.0
        rope_theta = get_rope_theta(config)
        rope_scaling = get_rope_scaling(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            config,
            model_config,
            layer_idx=layer_idx,
            kv_cache_config=kv_cache_config
        )
        self.mlp = LlamaMLP(
            config,
            model_config
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_quant=False,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_quant=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # with nvtx.range("layer_fwd"):
        # FP16 in FP16 out
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, input_metadata)

        # print(f"hidden_states.shape: {hidden_states.shape}; residual.shape: {residual.shape}")
        # print(f"residual[0,:5]: {residual[0,:5]}")
        # print(f"hidden_states[0,:5]: {hidden_states[0,:5]}") 
        
        # print(f"hidden_states.dtype: {hidden_states.dtype}; residual.dtype: {residual.dtype}")
        # print(f"hidden_states.device: {hidden_states.device}; residual.device: {residual.device}")
        
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        model_config: ModelConfig,
        quant_kv_cache: bool = True,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.model_config = model_config
        vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                (
                    LlamaDecoderLayer(config, model_config, i, kv_cache_config)
                    if quant_kv_cache
                    else None
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    hidden_states,
                    input_metadata
                )
            seq_len = hidden_states.size(0)
            for start_idx in range(0, seq_len, self.model_config.chunk_prefill_size):
                end_idx = min(seq_len, start_idx + self.model_config.chunk_prefill_size)
                hidden_states[start_idx: end_idx, :] = self.norm(hidden_states[start_idx: end_idx, :])

        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        model_config: ModelConfig,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(
            config,
            model_config,
            quant_kv_cache,
            kv_cache_config=kv_cache_config
        )
        self.model_config = model_config
        vocab_size = config.vocab_size
        # NOTE: The LM head is not quantized.
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "down_proj"]
        self.sampler = Sampler(sampling_params)

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
        sampling_params: SamplingParams,
    ):
        # pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
        # sampled_token_idx = self.sampler(input_ids, logits, input_metadata)
        # pred_token_idx = pred_token_idx.reshape(sampled_token_idx.shape)
        # # print(sampled_token_idx, pred_token_idx)

        # return pred_token_idx
        return self.sampler(input_ids, logits, input_metadata, sampling_params)


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
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
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
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                if "s2_scales" in name or "s2_zeros" in name:
                    shard_size = param.shape[1] // 2
                    param_slice = param.data[
                        :, stride_id * shard_size : (stride_id + 1) * shard_size
                    ]
                else:
                    shard_size = param.shape[0] // 2
                    param_slice = param.data[
                        shard_size * stride_id : shard_size * (stride_id + 1)
                    ]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
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
