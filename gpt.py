import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, q, k, v, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
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
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, L, E//H)
        k = k.view(N, S, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, S, E//H)
        v = v.view(N, S, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, S, E//H)

        if need_weights:
            # https://github.com/pytorch/pytorch/blob/3cbe7a53a9a1cea2ef2a042f1ab6f7758f7e4d74/torch/csrc/api/include/torch/nn/functional/activation.h#L884
            attn_output_weights = (q @ k.transpose(-1, -2)) / (k.size(-1) ** (-1/2)) # (N, H, L, E/H) @ (N, H, E/H, S) = (N, H, L, S)
            if is_causal: # attn_mask only supports 2D atm
                if attn_mask.dtype == torch.bool:
                    attn_output_weights = attn_output_weights.masked_fill(attn_mask, float('-inf'))
                elif attn_mask.dtype == torch.float32:
                    attn_output_weights = attn_output_weights + attn_mask
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = self.dropout1(attn_output_weights) # TODO: (N, L, S) in nn vs (N, L, S)
            attn_output = attn_output_weights @ v # (N, H, L, S) @ (N, H, S, E//H) = (N, H, L, E//H)
        else: # faster
            attn_output_weights = None
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(N, L, E) # (N, L, E) # TODO: check contiguous
        attn_output = self.projection(attn_output)

        if average_attn_weights:
            attn_output_weights.sum(dim=1) / self.num_heads

        return attn_output, attn_output_weights