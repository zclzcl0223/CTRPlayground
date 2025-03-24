import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_activation

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.bias = config.bias
        self.q_proj = nn.Linear(self.n_hidden, self.n_hidden, bias=config.bias)
        self.k_proj = nn.Linear(self.n_hidden, self.n_hidden, bias=config.bias)
        self.v_proj = nn.Linear(self.n_hidden, self.n_hidden, bias=config.bias)
        self.o_proj = nn.Linear(self.n_hidden, self.n_hidden, bias=config.bias)

    def forward(self, x):
        # x: [batch, sparse, hidden]
        batch, seq_len, hidden = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x.transpose(1, 2).contiguous().reshape(batch, seq_len, hidden)
        x = self.o_proj(x)
        return x

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.bias = config.bias
        self.up_proj = nn.Linear(self.n_hidden, 4 * self.n_hidden, bias=self.bias)
        self.activation = get_activation(config.activation)
        self.down_proj = nn.Linear(4 * self.n_hidden, self.n_hidden, bias=self.bias)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.att = MHA(config)
        self.att_norm = RMSNorm(self.n_hidden)
        self.ffn = FFN(config)
        self.ffn_norm = RMSNorm(self.n_hidden)
    
    def forward(self, x):
        x = x + self.att(self.att_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_hidden = config.n_hidden
        self.bias = config.bias
        self.n_dense = config.n_dense
        self.n_sparse = config.n_sparse

        # feature mixing
        vocab_sizes = [len(vocab) for vocab in tokenizer.values()]
        # multi-dim embed
        self.token_embedding = nn.Embedding(sum(vocab_sizes), self.n_embed)
        # to remove for loop
        self.register_buffer('offsets', torch.cumsum(torch.tensor([0] + vocab_sizes[:-1]), dim=0).unsqueeze(0))

        # up_proj
        self.up_proj = nn.Linear(self.n_embed, self.n_hidden, bias=False)
        # attention block
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # cls token
        #self.cls = nn.Parameter(torch.empty(1, self.n_embed).normal_(mean=0, std=1e-4))
        # proj_head
        self.proj_head = nn.Linear(self.n_hidden, 1, bias=False)

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1e-4)

    def forward(self, x):
        # x: [batch, sparse]
        batch, seq_len = x.shape
        x = x + self.offsets
        # [batch, sparse, embed]
        x = self.token_embedding(x)
        #x = torch.cat([self.cls[:, None, :].repeat(batch, 1, 1), x], dim=1)
        x = self.up_proj(x)

        for block in self.blocks:
            x = block(x)
        
        logits = self.proj_head(x.mean(dim=1))
        #logits = self.proj_head(x[:, 0])

        return logits
