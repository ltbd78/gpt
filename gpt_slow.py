import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout=0.0, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        if not self.batch_first: # transpose it to batch first
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            # TODO: when doing identity, (x^T)^T loss is higher than x loss for both this and nn.MultiheadAttention (shouldn't it be the same?)
        assert k.shape == v.shape
        N, L, E = q.shape
        N, S, E = k.shape
        q = self.query(q)  # (N, L, E)
        k = self.key(k) # (N, S, E)
        v = self.value(v) # (N, S, E)
        attn_output_weights = (q @ k.transpose(-1, -2)) / (k.size(-1) ** (-1/2)) # (N, L, E) @ (N, E, S) = (N, L, S)
        if is_causal: # attn_mask only supports 2D atm
            if attn_mask.dtype == torch.bool:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask, float('-inf'))
            elif attn_mask.dtype == torch.float32:
                attn_output_weights = attn_output_weights + attn_mask
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = attn_output_weights @ v # (N, L, S) @ (N, S, E) = (N, L, E)
        return attn_output

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim, dropout=dropout, batch_first=batch_first) for _ in range(num_heads)])
        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, q, k, v, need_weights=False, attn_mask=None, is_causal=False):
        if need_weights:
            raise NotImplementedError
        else:
            attn_output = torch.cat([h(q, k, v, attn_mask=attn_mask, is_causal=is_causal) for h in self.heads], dim=-1)
            attn_output = self.projection(attn_output)
            return attn_output, None # TODO: attn_output_weights