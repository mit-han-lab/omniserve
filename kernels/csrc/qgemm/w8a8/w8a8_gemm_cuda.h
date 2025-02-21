// Implemented by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
// @article{yang2025lserve,
//   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
//   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
//   year={2025}
// }

#include <torch/extension.h>

void w8a8_gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _wscales, torch::Tensor _ascales, torch::Tensor _out_feats);

