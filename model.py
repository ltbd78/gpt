from attention import *

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4 * out_features), # TODO: 4x according to paper
            nn.ReLU(),
            nn.Linear(4 * in_features, out_features), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, sequence_dim, embed_dim, num_heads, dropout=0.2): # L, E, H
        super().__init__()
        # self.register_buffer("attn_mask", torch.triu(torch.full((sequence_dim, sequence_dim), float('-inf')), diagonal=1)) # flavor 1 - pytorch
        self.register_buffer("attn_mask", torch.tril(torch.ones(sequence_dim, sequence_dim)) == 0) # flavor 2 - karpathy

        self.ln1 = nn.LayerNorm(embed_dim) # https://arxiv.org/pdf/2002.04745.pdf
        self.mha = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)

    def forward(self, x):
        x = self.ln1(x)
        attn_output, attn_output_weights = self.mha(x, x, x, need_weights=False, attn_mask=self.attn_mask, is_causal=True) # self attend
        x = x + attn_output
        x = x + self.ff(self.ln2(x)) # think on data
        return x

class GPT(nn.Module):
    def __init__(self, vocab_dim, sequence_dim, embed_dim, num_heads):
        super().__init__()
        self.sequence_dim = sequence_dim
        
        self.token_embedding = nn.Embedding(vocab_dim, embed_dim)
        self.position_embedding = nn.Embedding(sequence_dim, embed_dim)
        self.blocks = nn.Sequential(
            SelfAttentionBlock(sequence_dim, embed_dim, num_heads),
            SelfAttentionBlock(sequence_dim, embed_dim, num_heads),
            SelfAttentionBlock(sequence_dim, embed_dim, num_heads),
            nn.LayerNorm(embed_dim),
        )
        self.linear = nn.Linear(embed_dim, vocab_dim)

    def forward(self, x, y=None):
        # N is batch, L is length of time series, E is embedding dim
        N, L = x.shape
        token_embeddings = self.token_embedding(x) # (N, L, E)
        position_embeddings = self.position_embedding(torch.arange(L).to(x.device)) # (L, E) # T <= sequence_dim
        x = token_embeddings + position_embeddings # (N, L, E) +  (-, L, E) -> (N, L, E)
        x = self.blocks(x)
        logits = self.linear(x) # pred
        if y is None:
            loss = None
        else:
            N, L, E = logits.shape
            logits = logits.view(N*L, E)
            y = y.view(N*L)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, x, n_tokens):
        for i in range(n_tokens):
            x_cropped = x[:, -self.sequence_dim:] # crop s.t. it's <= sequence_dim
            logits, _ = self(x_cropped) # (N, L, E)
            logits = logits[:, -1, :] # (N, E)
            probs = F.softmax(logits, dim=-1) # (N, E)
            y_pred = torch.multinomial(probs, num_samples=1) # (N, 1)
            x = torch.cat((x, y_pred), dim=1) # (N, L) + (N, 1) = (N, L + 1)
        return x