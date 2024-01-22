import warnings

from attention import *


# TODO: add type hints to modules
class SelfMultiheadAttentionBlock(nn.Module):
    """
    Similar to:
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) # https://arxiv.org/pdf/2002.04745.pdf
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) # can prefix nn. to use torch's; if errors, make sure torch is v2.1.1
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False): # (N, L, E)
        ln_x = self.ln1(x) # saves as separate var ln_x since x is used for resid later
        attn_output, attn_output_weights = self.self_attn(ln_x, ln_x, ln_x, need_weights=False, attn_mask=src_mask, is_causal=is_causal)
        x = x + self.dropout(attn_output) # resid + self-attention
        x = x + self.mlp(self.ln2(x)) # resid + think on data
        return x # (N, L, E)


class GPT(nn.Module):
    def __init__(self, vocab_dim, sequence_dim, embed_dim, num_heads, num_layers, dropout=0.0, device='cpu'):
        super().__init__()
        self.sequence_dim = sequence_dim # also used for generate.py
        
        # self.register_buffer("attn_mask", torch.triu(torch.full((sequence_dim, sequence_dim), float('-inf')), diagonal=1)) # flavor 1 - pytorch
        self.register_buffer("attn_mask", torch.tril(torch.ones(sequence_dim, sequence_dim)) == 0) # flavor 2 - karpathy

        self.token_embedding = nn.Embedding(vocab_dim, embed_dim)
        self.position_embedding = nn.Embedding(sequence_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # if using personal model
        self.smab = SelfMultiheadAttentionBlock(embed_dim, num_heads, dim_feedforward=4*embed_dim, dropout=dropout) # 4x according to paper
        
        # # if using PyTorch's model
        # self.smab = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4*embed_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        
        # TODO: examine discrepancies in num of params between personal and PyTorch's
        
        self.blocks = nn.TransformerEncoder(self.smab, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_dim)
        
        # TODO: investigate https://paperswithcode.com/method/weight-tying
        # self.token_embedding.weight = self.linear.weight
        # val loss doesn't seem to be better from experiments
        
        self.device = device # TODO: check if possible to .to(device) outside init and reference self.device
        self.to(device)

    def forward(self, x): # (N, L)
        # N is batch, L is length of time series, E is embedding dim
        N, L = x.shape
        token_embeddings = self.token_embedding(x) # (N, L, E)
        position_embeddings = self.position_embedding(torch.arange(L).to(x.device)) # (L, E) # T <= sequence_dim
        x = token_embeddings + position_embeddings # (N, L, E) +  (-, L, E) -> (N, L, E)
        x = self.dropout(x)  # TODO: test with and without
        x = self.blocks(x, mask=self.attn_mask, is_causal=True) # (N, L, E)
        x = self.ln(x) # (N, L, E)
        logits = self.linear(x) # (N, L, E)
        return logits

    def get_loss(self, logits, y_true):
        N, L, E = logits.shape # (N, L, E)
        logits = logits.view(N*L, E) # (N * L, E)
        y_true = y_true.view(N*L) # # (N, L) -> (N * L)
        loss = F.cross_entropy(logits, y_true)
        return loss
    
    def fit(self, dataset_train, optimizer, batch_size, train_steps):
        self.train()
        dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        assert train_steps <= len(dl_train)
        for steps, (x, y) in enumerate(dl_train):
            if steps >= train_steps:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self(x)
            loss = self.get_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    @torch.no_grad() # avoids cuda out of memory error due to grads
    def evaluate(self, datasets, batch_size, val_steps):
        self.eval()
        losses = []
        for dataset in datasets:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            if val_steps >= len(dataloader):
                warnings.warn(f"valsteps {val_steps} >= len(dataloader) {len(dataloader)} | Using whole dataset.")
                losses_ = torch.full((len(dataloader),), float('nan'), device=self.device) # slightly faster if same device
            else:
                losses_ = torch.full((val_steps,), float('nan'), device=self.device) # slightly faster if same device
            for steps, (x, y) in enumerate(dataloader):
                if steps >= val_steps:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self(x)
                loss = self.get_loss(logits, y)
                losses_[steps] = loss.item() # .item() avoids cuda out of memory error due to grads
            losses.append(losses_.mean().item())
        return losses
    
    def save(self, path, optimizer_state_dict=None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict, # to resume training
        }, path)
    
    def load(self, path, optimizer=None, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())