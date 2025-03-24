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

class FFN(nn.Module):
    def __init__(self, n_hidden, n_inter, bias, activation):
        super().__init__()
        self.up_proj = nn.Linear(n_hidden, n_inter, bias=bias)
        self.activation = get_activation(activation)
        self.down_proj = nn.Linear(n_inter, n_hidden, bias=bias)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x

class MLPMixerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_feature = FFN(config.n_dense + config.n_sparse, config.n_hidden * 2, config.bias, config.activation)
        self.ffn_hidden = FFN(config.n_hidden, config.n_hidden * 4, config.bias, config.activation)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ffn_feature(x) + x
        x = x.transpose(1, 2)
        x = self.ffn_hidden(x) + x
        return x

class MLPMixer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed
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
        # mlp block
        self.blocks = nn.ModuleList([MLPMixerBlock(config) for _ in range(config.n_layer)])
        # proj_head
        self.proj_head = nn.Linear(self.n_hidden * (config.n_dense + config.n_sparse), 1, bias=False)

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
        # [batch, sparse, hidden]
        x = self.up_proj(x)

        for block in self.blocks:
            x = block(x)
        
        logits = self.proj_head(x.flatten(start_dim=1))

        return logits
