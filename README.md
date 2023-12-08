Homemade character-level GPT trained for 1-2 minutes on Tiny Shakespeare dataset using NVIDIA 2070 Super (Desktop) and NVIDIA 3070 (Laptop).

Follows theoretical concepts in Karpathy's tutorial but diverges in code style and implementation.

Custom MultiheadAttention implementation performs as fast as torch's nn.MultiheadAttention.

Custom MultiheadAttention diverges from Karpathy in that it allows possibility of cross-attention.

Careful variable naming (closely following torch's naming style) and plenty of in-line documentation allows for easy understanding of repo.

Was not able to scale due to limited time, but clean code foundation has been laid.

Next steps (will complete most before Dec 10):
- add model save, load, resume training
- training on word-level embeddings instead of character-level embeddings (see packages: tiktoken, sentencepiece)
- making the model larger (increasing num_layers, embed_dim, num_heads, sequence_dim, batch_size, train_steps)
- training longer (overnight)
- nn.Parallel for multi GPU scaling
- training on more datasets

To Use:

Run GPT.ipynb from beginning to end.

Recommended to have pytorch v2 or higher to experiment with `is_causal` parameter in `nn.MultiheadAttention.foward`

References:

nanoGPT - https://github.com/karpathy/nanoGPT

1706 - Attention is All You Need - https://arxiv.org/abs/1706.03762

1806 - GPT-1 - https://openai.com/research/language-unsupervised

1902 - GPT-2 - https://openai.com/research/better-language-models

2005 - GPT-3 - https://arxiv.org/pdf/2005.14165.pdf