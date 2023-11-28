import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiheadAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=True, dropout=0.0, device=None, dtype=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.mask = mask
        self.device = device # TODO: remove but torch.tril needs to be in buffer
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, E = q.size() # q, k, v should be same size
        q = self.query(q)  # (B, T, E)
        k = self.key(q) # (B, T, E)
        v = self.value(v) # (B, T, E)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, E//H)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, E//H)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, E//H)
        att = (q @ k.transpose(-1, -2)) / (k.size(-1) ** (-1/2)) # (B, H, T, E/H) @ (B, H, E/H, T) = (B, H, T, T)
        if self.mask:
            mask = torch.tril(torch.ones(T, T)).to(self.device)  # ?? .view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout1(att)
        y = att @ v # (B, H, T, T) @ (B, H, T, E//H) = (B, H, T, E//H)
        y = y.transpose(1, 2).contiguous().view(B, T, E) # TODO: check contiguous
        y = self.projection(y)
        y = self.dropout2(y)
        return y