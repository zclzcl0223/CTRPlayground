import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_activation

class DeepFM(nn.Module):

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
        # one dim embed
        self.fm1_embedding = nn.Embedding(sum(vocab_sizes), 1)
        # multi-dim embed
        self.fm2_embedding = nn.Embedding(sum(vocab_sizes), config.n_embed)
        # to remove for loop
        self.register_buffer('offsets', torch.cumsum(torch.tensor([0] + vocab_sizes[:-1]), dim=0).unsqueeze(0))
        # if bias
        self.bias = nn.Parameter(torch.zeros(1)) if self.bias else None

        # sparse features only dnn
        self.hidden_proj = nn.Linear(self.n_embed * (self.n_sparse + self.n_dense), self.n_hidden, bias=self.bias)
        self.ffn = nn.Sequential(
            get_activation(self.config.activation),
            nn.Linear(self.n_hidden, self.n_hidden, bias=self.bias),
            get_activation(self.config.activation),
            nn.Linear(self.n_hidden, self.n_hidden, bias=self.bias),
            get_activation(self.config.activation),
        )
        self.out_proj = nn.Linear(self.n_hidden, 1, bias=self.bias)

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
        x = x + self.offsets
        # [batch, sparse]
        fm1_embed = self.fm1_embedding(x).squeeze(-1)
        # [batch, sparse, embed]
        fm2_embed = self.fm2_embedding(x)

        # lr for sparse features, [batch, sparse] -> [batch, 1]
        logits = torch.sum(fm1_embed, dim=1, keepdim=True)
        # fm for sparse features, [batch, sparse, embed] -> [batch, sparse]
        fm2_out = 0.5 * (torch.pow(torch.sum(fm2_embed, dim=-1), 2) - torch.sum(torch.pow(fm2_embed, 2), dim=-1))
        logits += fm2_out.sum(dim=1, keepdim=True)
        
        # dnn part
        dnn_out = self.hidden_proj(fm2_embed.flatten(start_dim=1))
        dnn_out = self.ffn(dnn_out)
        dnn_out = self.out_proj(dnn_out)
        logits += dnn_out

        if self.bias is not None:
            logits += self.bias
        return logits
