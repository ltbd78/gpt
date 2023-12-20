Homemade GPT trained on Tiny Shakespeare dataset using NVIDIA 2070 Super (Desktop) and NVIDIA 3070 (Laptop).

Follows theoretical concepts in Karpathy's tutorial but diverges in code style and implementation.

Custom MultiheadAttention implementation performs as fast as torch's nn.MultiheadAttention.

Custom MultiheadAttention diverges from Karpathy in that it allows possibility of cross-attention.

Careful variable naming (closely following torch's naming style) and plenty of in-line documentation allows for easy understanding of repo.

Added training and Deployment on GCP.

Next steps:
- nn.Parallel for multi GPU scaling
- training on more datasets

Notes
- `GPT.ipynb` trains torch model locally
- `deployment.ipynb` deploys model on gcp endpoint
- `pipeline.ipynb` trains and deploys using kfp on gcp (will req an image file generated from deployment.ipynb)
- `kfp tutorial.ipynb` shows simple kfp examples


Recommended to have pytorch v2 or higher to experiment with `is_causal` parameter in `nn.MultiheadAttention.foward`

References:

nanoGPT - https://github.com/karpathy/nanoGPT

1706 - Attention is All You Need - https://arxiv.org/abs/1706.03762

1806 - GPT-1 - https://openai.com/research/language-unsupervised

1902 - GPT-2 - https://openai.com/research/better-language-models

2005 - GPT-3 - https://arxiv.org/pdf/2005.14165.pdf