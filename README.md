Homemade character-level GPT trained for 1-2 minutes on Tiny Shakespeare dataset using NVIDIA 2070 Super (Desktop) and NVIDIA 3070 (Laptop).

Follows theoretical concepts in Karpathy's tutorial but diverges in code style and implementation.

Custom MultiheadAttention implementation performs as fast as torch's nn.MultiheadAttention.

Custom MultiheadAttention diverges from Karpathy in that it allows possibility of cross-attention.

Careful variable naming (closely following torch's naming style) and plenty of in-line documentation allows for easy understanding of repo.

Was not able to scale due to limited time, but clean code foundation has been laid.

Next steps:
- add model save, load, resume training
- training on word-level embeddings instead of character-level embeddings (see packages: tiktoken, sentencepiece)
- making the model larger (increasing num_layers, embed_dim, num_heads, sequence_dim, batch_size, train_steps)
- training longer (overnight)
- nn.Parallel for multi GPU scaling
- training on more datasets