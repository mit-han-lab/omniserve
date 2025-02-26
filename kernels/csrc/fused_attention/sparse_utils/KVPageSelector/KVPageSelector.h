#pragma once

#include "../../common/cudaFp8Utils.h"
#include "../../common/gptKernels.h"
#include "../../common/kvCacheUtils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                  \
    do                                                                                                    \
    {                                                                                                     \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess)                                                                       \
        {                                                                                                 \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template <typename T>
struct Multihead_attention_page_selector_params_base
{

    // The output buffer. Dimensions B x D.
    T *out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The indirections to use for cache when beam sampling.
    const int *cache_indir = nullptr;

    // scales
    const float *query_weight_output_scale = nullptr;
    const float *attention_qk_scale = nullptr;
    const float *attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    // TODO: change name max_seq_len
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // Controls MHA/MQA/GQA
    int num_kv_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // Rotary position embedding type
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kROPE_GPT_NEOX;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    float rotary_embedding_base = 0.0f;
    RotaryScalingType rotary_embedding_scale_type = RotaryScalingType::kNONE;
    float rotary_embedding_scale = 1.0f;
    int rotary_embedding_max_positions = 0;
    // The current timestep. TODO Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // If relative position embedding is used
    const T *relative_attention_bias = nullptr;
    int relative_attention_bias_stride = 0;
    int max_distance = 0;

    // The slope per head of linear position bias to attention score (H).
    const T *linear_bias_slopes = nullptr;

    const T *ia3_key_weights = nullptr;
    const T *ia3_value_weights = nullptr;
    const int *ia3_tasks = nullptr;

    const float *qkv_scale_quant_orig = nullptr;
    const float *attention_out_scale_orig_quant = nullptr;

    // half *k_scale_orig_quant = nullptr;
    // half *v_scale_orig_quant = nullptr;
    half **k_scale_quant_orig = nullptr;
    half **v_scale_quant_orig = nullptr;

    bool int8_kv_cache = false;
    bool fp8_kv_cache = false;

    bool int4_kv_cache = false;
    bool kv_cache_with_zeros = false;

    // Multi-block setups
    bool multi_block_mode = false;

    // Number of streaming processors on the device.
    // Tune block size to maximum occupancy.
    int multi_processor_count = 1;

    mutable int timesteps_per_block = -1;
    mutable int timesteps_per_block_logic = -1;
    mutable int seq_len_tile = -1;

    mutable int max_seq_len_tile = -1;
    // The partial output buffer. Dimensions max_seq_len_tile x B x D. (for each timestep only seq_len_tile x B x D is
    // needed)
    T *partial_out = nullptr;
    // ThreadBlock sum. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float *partial_sum = nullptr;// add by JXGuo: should be B x H x max_seq_len_tile
    // ThreadBlock max. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float *partial_max = nullptr;// add by JXGuo: should be B x H x max_seq_len_tile
    // threadblock counter to identify the complete of partial attention computations
    int *block_counter = nullptr;

    const int *memory_length_per_sample = nullptr;

    int smem_preload_switch = 2048;
    int multiblock_switch = 2048; // (This is fake. Will be updated by fused_attention.cpp)
};

template <typename T>
struct Multihead_attention_page_selector_params;

// self-attention params
template <typename T>
struct Multihead_attention_page_selector_params : public Multihead_attention_page_selector_params_base<T>
{

    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool *finished = nullptr;

    // required in case of masked attention with different length
    const int *length_per_sample = nullptr;

    // input lengths to identify the paddings (i.e. input seq < padding < new generated seq).
    const int *input_lengths = nullptr;


    const int *retrieval_head_flags_ptr = nullptr;

    const int *head_rank_table_ptr = nullptr;

    int num_retrieval_kv_heads = 0;
    
    int num_streaming_kv_heads = 0;

    int streaming_sink_token_num = 0;
    
    int streaming_local_token_num = 0;

    int tokens_per_block = 0;

    const int *dynamic_sparse_page_idxes_ptr = nullptr;

    int num_dynamic_sparse_pages = 0;

    bool do_dynamic_sparse = false;

};
template <class T>
using Masked_multihead_attention_page_selector_params = Multihead_attention_page_selector_params<T>;
////////////////////////////////////////////////////////////////////////////////////////////////////

#define DECLARE_MMHA_NORMAL_AND_PAGED(T)                                                                \
    void masked_multihead_attention_page_selector(const Masked_multihead_attention_page_selector_params<T> &params,                 \
                                    const KVBlockArray<false> &retrieval_kv_buffer,                  \
                                    const KVBlockArray<true> &streaming_kv_buffer); \
    // void masked_multihead_attention_page_selector(const Masked_multihead_attention_page_selector_params<T> &params,                 \
    //                                 const KVLinearBuffer &kv_cache_buffer, const cudaStream_t &stream); 
DECLARE_MMHA_NORMAL_AND_PAGED(float);
DECLARE_MMHA_NORMAL_AND_PAGED(uint16_t);
#undef DECLARE_MMHA_NORMAL_AND_PAGED

////////////////////////////////////////////////////////////////////////////////////////////////////
